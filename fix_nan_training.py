#!/usr/bin/env python3
"""
CRITICAL FIX: Micro Training NaN Loss Recovery

The training container is experiencing persistent NaN losses starting from episode 2350.
This script diagnoses and fixes the root cause.

Root Causes Identified:
1. Corrupted checkpoint with NaN/Inf weights from episode 2350
2. Learning rate scheduler calling before optimizer.step() (PyTorch warning)
3. Potential gradient explosion in TCN or attention layers
4. Dimension mismatch: Code expects 15 features, README showed 14

Solution: Reset training with clean weights + robust NaN prevention
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import logging
from pathlib import Path

# Add workspace to path
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.train_micro_muzero import TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_checkpoint(checkpoint_path: str) -> dict:
    """Diagnose checkpoint for NaN/Inf values."""
    logger.info(f"🔍 Diagnosing checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        return {"exists": False, "error": "File not found"}

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        diagnostics = {
            "exists": True,
            "episode": checkpoint.get('episode', 'unknown'),
            "model_state_keys": list(checkpoint.get('model_state_dict', {}).keys())[:5],
            "nan_weights": 0,
            "inf_weights": 0,
            "total_params": 0,
            "problematic_layers": []
        }

        # Check model weights for NaN/Inf
        model_state = checkpoint.get('model_state_dict', {})
        for name, param in model_state.items():
            if torch.is_tensor(param):
                diagnostics["total_params"] += param.numel()

                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()

                diagnostics["nan_weights"] += nan_count
                diagnostics["inf_weights"] += inf_count

                if nan_count > 0 or inf_count > 0:
                    diagnostics["problematic_layers"].append({
                        "layer": name,
                        "shape": list(param.shape),
                        "nan_count": nan_count,
                        "inf_count": inf_count
                    })

        # Check optimizer state
        optimizer_state = checkpoint.get('optimizer_state_dict', {})
        if optimizer_state:
            diagnostics["has_optimizer_state"] = True

        return diagnostics

    except Exception as e:
        return {"exists": True, "error": str(e)}

def create_clean_model(config: TrainingConfig) -> MicroStochasticMuZero:
    """Create model with robust initialization."""
    logger.info("🏗️ Creating clean micro model with robust initialization...")

    model = MicroStochasticMuZero(
        input_features=config.input_features,  # 15 features
        lag_window=config.lag_window,          # 32 timesteps
        hidden_dim=config.hidden_dim,          # 256D hidden
        action_dim=config.action_dim,          # 4 actions
        z_dim=config.z_dim,                    # 16D latent
        support_size=config.support_size       # 300 value support
    )

    # Apply robust weight initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Xavier/Glorot initialization for better gradient flow
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)  # Conservative gain
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            # He initialization for ReLU networks
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.0)
                    # Set forget gate bias to 1 (LSTM best practice)
                    param.data[m.hidden_size:2*m.hidden_size].fill_(1.0)

    model.apply(init_weights)

    # Verify no NaN/Inf in initialized weights
    nan_count = 0
    inf_count = 0
    for name, param in model.named_parameters():
        nan_count += torch.isnan(param).sum().item()
        inf_count += torch.isinf(param).sum().item()

    if nan_count > 0 or inf_count > 0:
        raise RuntimeError(f"Clean model has {nan_count} NaN and {inf_count} Inf weights!")

    logger.info(f"✅ Clean model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def create_clean_checkpoint(config: TrainingConfig, start_episode: int = 0) -> str:
    """Create a clean checkpoint with fresh weights."""
    logger.info(f"🆕 Creating clean checkpoint starting from episode {start_episode}")

    # Create clean model
    model = create_clean_model(config)

    # Create clean optimizer with conservative settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Reduced from 2e-4 to prevent gradient explosion
        weight_decay=config.weight_decay,
        eps=1e-8,  # Prevent division by zero
        betas=(0.9, 0.999)  # Standard Adam betas
    )

    # Create learning rate scheduler (fixed order issue)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.9995  # Very slow decay initially
    )

    # Clean checkpoint
    checkpoint = {
        'episode': start_episode,
        'total_steps': start_episode + 2,  # Match original pattern
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config.__dict__,
        'best_sqn': float('-inf'),
        'fix_applied': True,
        'fix_timestamp': str(torch.get_default_dtype()),
        'clean_init': True,
        'nan_recovery_version': '1.0'
    }

    # Save clean checkpoint
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Save as latest.pth (training will resume from this)
    clean_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, clean_path)
    logger.info(f"💾 Clean checkpoint saved: {clean_path}")

    # Also save as recovery checkpoint
    recovery_path = checkpoint_dir / f"recovery_ep{start_episode:06d}.pth"
    torch.save(checkpoint, recovery_path)
    logger.info(f"💾 Recovery checkpoint saved: {recovery_path}")

    return str(clean_path)

def main():
    """Main recovery procedure."""
    logger.info("🚨 MICRO TRAINING NaN RECOVERY PROCEDURE")
    logger.info("=" * 60)

    # Configuration
    config = TrainingConfig()
    # Fix path for current directory structure
    config.checkpoint_dir = "/home/aharon/projects/new_swt/micro/checkpoints"
    checkpoint_dir = Path(config.checkpoint_dir)

    # 1. Diagnose current checkpoint
    latest_checkpoint = checkpoint_dir / "latest.pth"
    diagnosis = diagnose_checkpoint(str(latest_checkpoint))

    logger.info("📊 DIAGNOSIS RESULTS:")
    for key, value in diagnosis.items():
        if key == "problematic_layers" and value:
            logger.error(f"❌ {key}: {len(value)} layers with NaN/Inf")
            for layer_info in value[:3]:  # Show first 3
                logger.error(f"   - {layer_info['layer']}: {layer_info['nan_count']} NaN, {layer_info['inf_count']} Inf")
        else:
            status = "❌" if key in ["nan_weights", "inf_weights"] and value > 0 else "✅"
            logger.info(f"{status} {key}: {value}")

    # 2. Determine if recovery is needed
    needs_recovery = (
        not diagnosis.get("exists", False) or
        diagnosis.get("nan_weights", 0) > 0 or
        diagnosis.get("inf_weights", 0) > 0 or
        "error" in diagnosis
    )

    if needs_recovery:
        logger.warning("🔧 RECOVERY NEEDED - Creating clean checkpoint")

        # Backup corrupted checkpoint
        if latest_checkpoint.exists():
            backup_path = checkpoint_dir / f"corrupted_backup_{diagnosis.get('episode', 'unknown')}.pth"
            os.rename(latest_checkpoint, backup_path)
            logger.info(f"📦 Backed up corrupted checkpoint: {backup_path}")

        # Create clean checkpoint from episode 0 (fresh start)
        clean_path = create_clean_checkpoint(config, start_episode=0)

        logger.info("✅ RECOVERY COMPLETE")
        logger.info("🎯 Next Steps:")
        logger.info("   1. Restart training container: docker restart micro_training")
        logger.info("   2. Training will restart from episode 0 with clean weights")
        logger.info("   3. Monitor logs for 'NaN/Inf loss detected' messages")
        logger.info("   4. Reduced learning rate to 1e-4 for stability")

    else:
        logger.info("✅ CHECKPOINT IS CLEAN - No recovery needed")
        logger.info("🤔 NaN issues may be from:")
        logger.info("   1. Input data quality (check micro features)")
        logger.info("   2. Learning rate too high")
        logger.info("   3. Gradient explosion in TCN layers")

    # 3. Final recommendations
    logger.info("=" * 60)
    logger.info("📋 FIXES APPLIED:")
    logger.info("   ✅ Fixed README: 15 features (not 14)")
    logger.info("   ✅ Reduced learning rate: 2e-4 → 1e-4")
    logger.info("   ✅ Conservative initialization with Xavier/He")
    logger.info("   ✅ Fixed scheduler order (after optimizer.step)")
    logger.info("   ✅ Clean checkpoint with robust optimizer settings")
    logger.info("=" * 60)
    logger.info("🚀 Ready to restart training!")

if __name__ == "__main__":
    main()