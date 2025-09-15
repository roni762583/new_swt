#!/usr/bin/env python3
"""
Production training script for 1,000,000 sessions
Sends best checkpoints to validation container automatically
"""

import os
import sys
import json
import time
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training configuration for 1M sessions
TRAINING_CONFIG = {
    # Session configuration
    'total_sessions': 1_000_000,
    'sessions_per_checkpoint': 1000,  # Save every 1000 sessions
    'validation_interval': 5000,       # Validate every 5000 sessions

    # Model architecture
    'hidden_dim': 256,
    'support_size': 300,
    'num_layers': 8,
    'num_heads': 8,

    # Training parameters
    'batch_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'gradient_clip': 5.0,

    # Experience replay
    'buffer_size': 100_000,
    'min_buffer_size': 10_000,
    'priority_alpha': 0.6,
    'priority_beta': 0.4,

    # Session sampling
    'min_session_hours': 6,
    'max_gap_minutes': 10,
    'session_overlap_hours': 0,

    # MCTS parameters
    'num_simulations': 50,
    'c_puct': 1.25,
    'dirichlet_alpha': 0.3,
    'exploration_fraction': 0.25,

    # Device configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,

    # Validation settings
    'validation_samples': 10000,
    'validation_container': 'swt_validation_container',
    'min_improvement_for_transfer': 0.01,  # 1% improvement threshold
}

class MillionSessionTrainer:
    """Trainer for 1 million session production run"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration"""
        self.config = config
        self.device = torch.device(config['device'])

        # Tracking
        self.current_session = 0
        self.best_validation_score = -float('inf')
        self.checkpoint_history = []

        # Paths
        self.checkpoint_dir = Path('/workspace/checkpoints')
        self.validation_dir = Path('/workspace/validation_queue')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.validation_dir.mkdir(exist_ok=True)

        logger.info("="*60)
        logger.info("🚀 MILLION SESSION TRAINING INITIALIZED")
        logger.info("="*60)
        logger.info(f"Total Sessions: {config['total_sessions']:,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint Interval: {config['sessions_per_checkpoint']:,}")
        logger.info(f"Validation Interval: {config['validation_interval']:,}")
        logger.info("="*60)

    def initialize_model(self):
        """Initialize or load model"""
        logger.info("🧠 Initializing model architecture...")

        # Import model components
        sys.path.insert(0, '/workspace')
        from swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetwork, SWTStochasticMuZeroConfig

        # Create config
        model_config = SWTStochasticMuZeroConfig(
            hidden_dim=self.config['hidden_dim'],
            support_size=self.config['support_size'],
            representation_blocks=self.config['num_layers'],
            dynamics_blocks=self.config['num_layers'],
            prediction_blocks=self.config['num_layers']
        )

        # Create model
        self.networks = SWTStochasticMuZeroNetwork(
            config=model_config,
            observation_shape=(137,),
            action_space_size=4
        ).to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.networks.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10000,  # Initial restart interval
            T_mult=2,   # Double interval after each restart
            eta_min=1e-6
        )

        logger.info(f"✅ Model initialized with {sum(p.numel() for p in self.networks.parameters()):,} parameters")

    def train_session_batch(self, session_batch: int) -> Dict[str, float]:
        """Train a batch of sessions"""
        batch_losses = []

        for session in range(self.config['sessions_per_checkpoint']):
            self.current_session += 1

            # Simulate training (replace with actual training logic)
            loss = self._train_single_session()
            batch_losses.append(loss)

            # Log progress
            if self.current_session % 100 == 0:
                avg_loss = np.mean(batch_losses[-100:])
                logger.info(f"Session {self.current_session:,}/{self.config['total_sessions']:,} | Loss: {avg_loss:.4f}")

        return {
            'avg_loss': float(np.mean(batch_losses)),
            'min_loss': float(np.min(batch_losses)),
            'max_loss': float(np.max(batch_losses))
        }

    def _train_single_session(self) -> float:
        """Train on a single session (placeholder - implement actual training)"""
        # This is where you'd implement actual training logic
        # For now, simulating with decreasing loss
        base_loss = 1.0 / (1 + self.current_session / 10000)
        noise = np.random.normal(0, 0.1)
        return max(0.01, base_loss + noise)

    def save_checkpoint(self, metrics: Dict[str, Any]):
        """Save checkpoint with metadata"""
        checkpoint_path = self.checkpoint_dir / f"swt_session_{self.current_session}.pth"
        metadata_path = self.checkpoint_dir / f"swt_session_{self.current_session}.json"

        # Save model
        checkpoint = {
            'session': self.current_session,
            'model_state': self.networks.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {
            'session': self.current_session,
            'timestamp': time.time(),
            'metrics': metrics,
            'best_validation_score': self.best_validation_score
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")

        # Track checkpoint
        self.checkpoint_history.append({
            'session': self.current_session,
            'path': str(checkpoint_path),
            'metrics': metrics
        })

        # Keep only last 10 checkpoints
        if len(self.checkpoint_history) > 10:
            old_checkpoint = self.checkpoint_history.pop(0)
            old_path = Path(old_checkpoint['path'])
            if old_path.exists():
                old_path.unlink()
            old_meta = old_path.with_suffix('.json')
            if old_meta.exists():
                old_meta.unlink()

    def validate_and_transfer(self):
        """Validate current model and transfer if improved"""
        logger.info("🔍 Running validation...")

        # Simulate validation (replace with actual validation)
        validation_score = self._run_validation()

        improvement = validation_score - self.best_validation_score
        improvement_pct = (improvement / max(abs(self.best_validation_score), 1)) * 100

        logger.info(f"📊 Validation Score: {validation_score:.4f} (Previous best: {self.best_validation_score:.4f})")

        if improvement > self.config['min_improvement_for_transfer']:
            logger.info(f"🎯 New best! Improvement: {improvement_pct:+.2f}%")
            self.best_validation_score = validation_score

            # Transfer to validation container
            self._transfer_to_validation()
        else:
            logger.info(f"📈 No significant improvement ({improvement_pct:+.2f}%)")

    def _run_validation(self) -> float:
        """Run validation (placeholder - implement actual validation)"""
        # Simulate improving validation score
        base_score = 10 + self.current_session / 50000
        noise = np.random.normal(0, 1)
        return base_score + noise

    def _transfer_to_validation(self):
        """Transfer best checkpoint to validation container"""
        try:
            latest_checkpoint = self.checkpoint_dir / f"swt_session_{self.current_session}.pth"
            latest_metadata = self.checkpoint_dir / f"swt_session_{self.current_session}.json"

            # Copy to validation queue
            validation_checkpoint = self.validation_dir / f"best_session_{self.current_session}.pth"
            validation_metadata = self.validation_dir / f"best_session_{self.current_session}.json"

            shutil.copy2(latest_checkpoint, validation_checkpoint)
            shutil.copy2(latest_metadata, validation_metadata)

            # Notify validation container (using Docker exec)
            cmd = f"docker exec {self.config['validation_container']} python3 /workspace/validate_checkpoint.py --checkpoint {validation_checkpoint.name}"
            subprocess.run(cmd, shell=True, check=False)

            logger.info(f"✅ Transferred checkpoint to validation: {validation_checkpoint.name}")

        except Exception as e:
            logger.error(f"❌ Failed to transfer checkpoint: {e}")

    def run(self):
        """Run the million session training"""
        try:
            # Initialize model
            self.initialize_model()

            # Training loop
            logger.info("\n" + "="*60)
            logger.info("🏁 STARTING TRAINING")
            logger.info("="*60 + "\n")

            start_time = time.time()

            while self.current_session < self.config['total_sessions']:
                # Train batch of sessions
                batch_metrics = self.train_session_batch(
                    self.current_session // self.config['sessions_per_checkpoint']
                )

                # Save checkpoint
                self.save_checkpoint(batch_metrics)

                # Validate and transfer if needed
                if self.current_session % self.config['validation_interval'] == 0:
                    self.validate_and_transfer()

                # Progress report
                elapsed = time.time() - start_time
                sessions_per_hour = self.current_session / (elapsed / 3600)
                eta_hours = (self.config['total_sessions'] - self.current_session) / sessions_per_hour

                logger.info("-"*60)
                logger.info(f"📊 Progress: {self.current_session:,}/{self.config['total_sessions']:,} ({self.current_session/self.config['total_sessions']*100:.1f}%)")
                logger.info(f"⏱️  Speed: {sessions_per_hour:.0f} sessions/hour")
                logger.info(f"🕐 ETA: {eta_hours:.1f} hours")
                logger.info("-"*60 + "\n")

            # Final summary
            total_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("🎉 TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info(f"Total Sessions: {self.current_session:,}")
            logger.info(f"Total Time: {total_time/3600:.1f} hours")
            logger.info(f"Best Validation Score: {self.best_validation_score:.4f}")
            logger.info(f"Final Checkpoints: {len(self.checkpoint_history)}")
            logger.info("="*60)

        except KeyboardInterrupt:
            logger.info("\n⚠️  Training interrupted by user")
            self._save_emergency_checkpoint()
        except Exception as e:
            logger.error(f"\n❌ Training failed: {e}")
            self._save_emergency_checkpoint()
            raise

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on interruption"""
        try:
            checkpoint_path = self.checkpoint_dir / f"emergency_session_{self.current_session}.pth"
            torch.save({
                'session': self.current_session,
                'model_state': self.networks.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config
            }, checkpoint_path)
            logger.info(f"💾 Emergency checkpoint saved: {checkpoint_path.name}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train SWT for 1 million sessions')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--sessions', type=int, default=1_000_000, help='Total sessions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Update config
    TRAINING_CONFIG['total_sessions'] = args.sessions
    TRAINING_CONFIG['device'] = args.device

    # Create trainer
    trainer = MillionSessionTrainer(TRAINING_CONFIG)

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()