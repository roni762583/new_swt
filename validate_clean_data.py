#!/usr/bin/env python3
"""
Validation script using clean production data and WST features
Tests model performance on real, quality-verified forex data
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import sys
import h5py

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from swt_environments.swt_forex_env import SWTForexEnvironment
from swt_models.swt_muzero_model import SWTMuZeroNet
from swt_system.swt_config_manager import SWTConfigManager
from swt_environments.swt_precomputed_wst_loader import PrecomputedWSTLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_with_clean_data(
    checkpoint_path: str,
    wst_file: str,
    csv_file: str,
    num_sessions: int = 10
):
    """
    Run validation using clean GBPJPY data

    Args:
        checkpoint_path: Path to model checkpoint
        wst_file: Path to precomputed WST features (clean data)
        csv_file: Path to clean CSV data
        num_sessions: Number of 6-hour sessions to validate
    """
    logger.info("=" * 60)
    logger.info("🏦 SWT VALIDATION WITH CLEAN PRODUCTION DATA")
    logger.info("=" * 60)

    # Verify files exist
    if not Path(checkpoint_path).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    if not Path(wst_file).exists():
        logger.error(f"❌ WST file not found: {wst_file}")
        return

    if not Path(csv_file).exists():
        logger.error(f"❌ CSV file not found: {csv_file}")
        return

    # Quick data verification
    with h5py.File(wst_file, 'r') as f:
        n_windows = f.attrs.get('n_windows', 0)
        wst_shape = f['wst_features'].shape if 'wst_features' in f else (0, 0)
        logger.info(f"✅ WST data loaded: {n_windows:,} windows, shape {wst_shape}")

    # Load configuration
    config_manager = SWTConfigManager()
    config = config_manager.merged_config

    # Initialize environment with clean data
    logger.info(f"📊 Loading clean forex data from: {csv_file}")
    env = SWTForexEnvironment(
        data_path=csv_file,
        config=config.get('environment', {}),
        mode='validation'
    )

    # Initialize WST loader
    logger.info(f"🌊 Loading WST features from: {wst_file}")
    wst_loader = PrecomputedWSTLoader(wst_file)

    # Load model
    logger.info(f"🤖 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Initialize model
    model = SWTMuZeroNet(
        observation_size=137,  # 128 WST + 9 position features
        action_size=3,
        config=config.get('model', {})
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Run validation sessions
    logger.info(f"🚀 Running {num_sessions} validation sessions...")

    results = []
    for session_idx in range(num_sessions):
        # Reset environment for new 6-hour session
        observation, info = env.reset()

        # Get current window index
        window_index = env.current_step

        # Initialize session stats
        session_pnl = 0.0
        trades = 0
        steps = 0
        max_drawdown = 0.0
        peak_pnl = 0.0

        # Run 6-hour session (360 minutes)
        session_length = 360
        done = False

        while not done and steps < session_length:
            # Get WST features for current window
            if window_index < len(wst_loader):
                wst_features = wst_loader.get_wst_features(window_index)
                if hasattr(wst_features, 'numpy'):
                    wst_features = wst_features.numpy()
            else:
                logger.warning(f"Window index {window_index} out of range")
                break

            # Prepare observation
            position_features = observation[1] if isinstance(observation, tuple) else observation
            combined_features = np.concatenate([wst_features, position_features])

            # Get model prediction
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                policy_logits, value = model.initial_inference(obs_tensor)
                action_probs = torch.softmax(policy_logits, dim=1)
                action = torch.argmax(action_probs, dim=1).item()

            # Take action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update stats
            steps += 1
            window_index = env.current_step

            if info.get('trade_closed'):
                trades += 1
                session_pnl += info.get('pnl_pips', 0)

                # Track drawdown
                if session_pnl > peak_pnl:
                    peak_pnl = session_pnl
                drawdown = peak_pnl - session_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        # Session complete
        expectancy = session_pnl / trades if trades > 0 else 0
        sqn = (expectancy * np.sqrt(trades)) / np.sqrt(max_drawdown + 1e-6)

        logger.info(f"Session {session_idx + 1}/{num_sessions}:")
        logger.info(f"  PnL: {session_pnl:+.1f} pips")
        logger.info(f"  Trades: {trades}")
        logger.info(f"  Expectancy: {expectancy:+.2f} pips/trade")
        logger.info(f"  Max DD: {max_drawdown:.1f} pips")
        logger.info(f"  SQN: {sqn:.2f}")

        results.append({
            'session': session_idx + 1,
            'pnl': session_pnl,
            'trades': trades,
            'expectancy': expectancy,
            'max_dd': max_drawdown,
            'sqn': sqn
        })

    # Summary statistics
    total_pnl = sum(r['pnl'] for r in results)
    total_trades = sum(r['trades'] for r in results)
    avg_expectancy = np.mean([r['expectancy'] for r in results])
    avg_sqn = np.mean([r['sqn'] for r in results])

    logger.info("=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info(f"Total PnL: {total_pnl:+.1f} pips")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Avg Expectancy: {avg_expectancy:+.2f} pips/trade")
    logger.info(f"Avg SQN: {avg_sqn:.2f}")

    if avg_sqn > 2.0:
        logger.info("✅ Model performance: EXCELLENT (SQN > 2.0)")
    elif avg_sqn > 1.5:
        logger.info("✅ Model performance: GOOD (SQN > 1.5)")
    elif avg_sqn > 1.0:
        logger.info("⚠️ Model performance: ACCEPTABLE (SQN > 1.0)")
    else:
        logger.info("❌ Model performance: POOR (SQN < 1.0)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate SWT model with clean data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_checkpoint.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--wst-file",
        type=str,
        default="precomputed_wst/GBPJPY_WST_CLEAN_2022-2025.h5",
        help="Path to clean WST features"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="data/GBPJPY_M1_REAL_2022-2025.csv",
        help="Path to clean CSV data"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of validation sessions"
    )

    args = parser.parse_args()

    validate_with_clean_data(
        checkpoint_path=args.checkpoint,
        wst_file=args.wst_file,
        csv_file=args.csv_file,
        num_sessions=args.sessions
    )


if __name__ == "__main__":
    main()