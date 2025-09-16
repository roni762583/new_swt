#!/usr/bin/env python3
"""
Validation script that uses pre-computed WST features from HDF5
Uses the actual SWTForexEnv trading environment for consistent simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import argparse
from dataclasses import dataclass
import time
from datetime import datetime, timedelta

# Import our actual trading environment and components
from swt_environments.swt_forex_env import SWTForexEnvironment as SWTForexEnv, SWTAction, SWTPositionState
from swt_features.feature_processor import FeatureProcessor
from swt_features.precomputed_wst_loader import PrecomputedWSTLoader
from swt_core.config_manager import ConfigManager
from swt_core.types import PositionState, PositionType
from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics for a checkpoint - using pips after spread costs"""
    total_pips: float  # Total pips after spread costs
    avg_pips_per_trade: float  # Average pips per trade
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pips: float  # Max drawdown in pips
    total_return_pct: float = 0.0  # Percentage return for comparison


class PrecomputedWSTValidator:
    """Validator that uses the actual SWTForexEnv with pre-computed WST features"""

    def __init__(
        self,
        wst_file: str = "precomputed_wst/GBPJPY_WST_3.5years_streaming.h5",
        csv_file: str = "data/GBPJPY_M1_3.5years_20250912.csv"
    ):
        """
        Initialize validator with actual trading environment

        Args:
            wst_file: Path to pre-computed WST HDF5 file
            csv_file: Path to CSV with price data
        """
        self.wst_file = Path(wst_file)
        self.csv_file = Path(csv_file)

        if not self.wst_file.exists():
            raise FileNotFoundError(f"Pre-computed WST file not found: {wst_file}")

        # Load configuration
        self.config_manager = ConfigManager('config')
        self.config_manager.load_all_configs()

        # Create precomputed WST loader
        self.wst_loader = PrecomputedWSTLoader(
            str(self.wst_file),
            cache_size=10000
        )

        # Create feature processor with precomputed WST
        self.feature_processor = FeatureProcessor(
            self.config_manager,
            precomputed_loader=self.wst_loader
        )

        # Create the actual trading environment
        self.env = SWTForexEnv(
            data_path=str(self.csv_file),
            config_dict=self.config_manager.merged_config  # Use merged_config from ConfigManager
        )

        logger.info(f"Initialized trading environment with {self.csv_file}")
        logger.info(f"Using precomputed WST from {self.wst_file}")


    def validate_checkpoint(self, checkpoint_path: str, num_runs: int = 10) -> ValidationMetrics:
        """
        Validate a checkpoint using the actual trading environment

        Args:
            checkpoint_path: Path to checkpoint file
            num_runs: Number of Monte Carlo runs (6-hour sessions)

        Returns:
            ValidationMetrics with results in pips after spread costs
        """
        logger.info(f"\nValidating checkpoint: {checkpoint_path}")
        logger.info(f"Using actual SWTForexEnv with 4 pip spread cost")

        # Load checkpoint using the fixed loader function
        checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
        network = checkpoint_data['networks']
        network.eval()

        # Run validation with multiple 6-hour sessions
        SESSION_LENGTH = 360  # 6 hours = 360 1-minute bars
        all_trades = []  # Store all completed trades
        session_results = []  # Store per-session results

        with torch.no_grad():
            for session_num in range(num_runs):
                logger.info(f"  Session {session_num + 1}/{num_runs}...")

                # Reset environment for new session
                observation = self.env.reset()

                session_trades = []
                session_pips = 0
                done = False
                steps = 0

                while not done and steps < SESSION_LENGTH:
                    # Get current market price and position state from environment
                    market_prices = observation['market_prices'] if isinstance(observation, dict) else observation[:256]
                    current_price = market_prices[-1]  # Last price in the series
                    position_state = self.env.position  # Get actual position state from environment

                    # For precomputed WST, use the current step as window index
                    window_index = self.env.current_step

                    # Convert SWTPositionState to PositionState for feature processor
                    if position_state.is_long:
                        pos_type = PositionType.LONG
                    elif position_state.is_short:
                        pos_type = PositionType.SHORT
                    else:
                        pos_type = PositionType.FLAT

                    # Create immutable PositionState
                    feature_position_state = PositionState(
                        position_type=pos_type,
                        entry_price=position_state.entry_price,
                        unrealized_pnl_pips=position_state.unrealized_pnl_pips,
                        duration_minutes=position_state.duration_bars  # Using bars as minutes proxy
                    )

                    # Process observation with position state and current price
                    processed_obs = self.feature_processor.process_observation(
                        position_state=feature_position_state,
                        current_price=current_price,
                        window_index=window_index  # Pass window index for precomputed WST
                    )

                    # Get combined features (128 WST + 9 position)
                    wst_features = processed_obs.combined_features  # Already 137 features
                    obs_tensor = torch.FloatTensor(wst_features).unsqueeze(0)

                    # Get model prediction using initial_inference
                    inference_result = network.initial_inference(obs_tensor)
                    policy_logits = inference_result['policy_logits']

                    # Select action (greedy for validation)
                    action = policy_logits.argmax().item()

                    # Step environment - THIS PROPERLY HANDLES SPREAD COSTS
                    observation, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    steps += 1

                    # Collect completed trades with actual pip results after spread
                    if 'completed_trades' in info and info['completed_trades']:
                        for trade in info['completed_trades']:
                            trade_pips = trade.pnl_pips  # Actual pips after 4 pip spread
                            session_trades.append(trade_pips)
                            all_trades.append(trade_pips)
                            session_pips += trade_pips

                # Store session results
                session_results.append({
                    'total_pips': session_pips,
                    'num_trades': len(session_trades),
                    'trades': session_trades
                })

        # Calculate metrics based on actual pip results after spread costs
        if len(all_trades) > 0:
            trades_array = np.array(all_trades)

            # Total and average pips (after 4 pip spread cost)
            total_pips = np.sum(trades_array)
            avg_pips_per_trade = np.mean(trades_array)

            # Win rate (trades that made money after spread)
            win_rate = np.mean(trades_array > 0)

            # Sharpe ratio based on per-session pip results
            if len(session_results) > 1:
                session_pips = [s['total_pips'] for s in session_results]
                sharpe = np.mean(session_pips) / (np.std(session_pips) + 1e-8)
            else:
                sharpe = 0

            # Max drawdown in pips
            if len(trades_array) > 10:
                cumsum = np.cumsum(trades_array)
                running_max = np.maximum.accumulate(cumsum)
                drawdown_pips = cumsum - running_max
                max_dd_pips = abs(np.min(drawdown_pips))
            else:
                max_dd_pips = 0

            # Calculate percentage return for comparison (optional)
            # Assuming starting capital and pip value
            initial_capital = 10000  # Example starting capital
            pip_value = 0.01  # Example pip value for GBPJPY
            total_return_pct = (total_pips * pip_value) / initial_capital * 100
        else:
            total_pips = 0
            avg_pips_per_trade = 0
            win_rate = 0
            sharpe = 0
            max_dd_pips = 0
            total_return_pct = 0

        metrics = ValidationMetrics(
            total_pips=total_pips,
            avg_pips_per_trade=avg_pips_per_trade,
            total_trades=len(all_trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown_pips=max_dd_pips,
            total_return_pct=total_return_pct
        )

        return metrics

    def compare_checkpoints(self, checkpoint_paths: List[str]) -> Dict[str, ValidationMetrics]:
        """Compare multiple checkpoints"""
        results = {}

        for path in checkpoint_paths:
            name = Path(path).stem
            metrics = self.validate_checkpoint(path)
            results[name] = metrics

            logger.info(f"\n{name} Results (Pips After Spread):")
            logger.info(f"  Total Trades: {metrics.total_trades}")
            logger.info(f"  Total Pips: {metrics.total_pips:.1f}")
            logger.info(f"  Avg Pips/Trade: {metrics.avg_pips_per_trade:.2f}")
            logger.info(f"  Win Rate: {metrics.win_rate:.2%}")
            logger.info(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max DD: {metrics.max_drawdown_pips:.1f} pips")
            logger.info(f"  Return %: {metrics.total_return_pct:.2f}%")

        return results


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate checkpoints with pre-computed WST")
    parser.add_argument("--checkpoints", nargs="+",
                       default=["checkpoints/episode_10_best.pth",
                               "checkpoints/episode_775_aggressive.pth"],
                       help="Checkpoint paths to validate")
    parser.add_argument("--wst-file",
                       default="precomputed_wst/GBPJPY_WST_3.5years_streaming.h5",
                       help="Pre-computed WST HDF5 file")
    parser.add_argument("--csv-file",
                       default="data/GBPJPY_M1_3.5years_20250912.csv",
                       help="CSV price data file")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of Monte Carlo runs")

    args = parser.parse_args()

    # Start timing
    total_start_time = time.time()
    print(f"\n⏱️  VALIDATION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Create validator
    validator = PrecomputedWSTValidator(
        wst_file=args.wst_file,
        csv_file=args.csv_file
    )

    # Validate checkpoints
    results = {}
    for checkpoint in args.checkpoints:
        if Path(checkpoint).exists():
            name = Path(checkpoint).stem
            logger.info(f"\n{'='*60}")
            logger.info(f"Validating: {name}")
            logger.info('='*60)

            metrics = validator.validate_checkpoint(checkpoint, num_runs=args.runs)
            results[name] = metrics

            # Print results in pips after spread costs
            print(f"\n📊 {name} Validation Results (ACTUAL PIPS AFTER SPREAD):")
            print(f"  ✅ Total Trades: {metrics.total_trades}")
            print(f"  ✅ Total Pips: {metrics.total_pips:.1f} pips (after 4 pip spread)")
            print(f"  ✅ Avg per Trade: {metrics.avg_pips_per_trade:.2f} pips")
            print(f"  ✅ Win Rate: {metrics.win_rate:.2%}")
            print(f"  ✅ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"  ✅ Max Drawdown: {metrics.max_drawdown_pips:.1f} pips")
            print(f"  ✅ Return %: {metrics.total_return_pct:.2f}%")

    # Compare results
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📈 COMPARISON")
        print('='*60)

        best_pips = max(results.items(), key=lambda x: x[1].total_pips)
        print(f"🏆 Best Total Pips: {best_pips[0]} with {best_pips[1].total_pips:.1f} pips")

        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        print(f"🏆 Best Sharpe: {best_sharpe[0]} with {best_sharpe[1].sharpe_ratio:.2f}")

    # End timing
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n{'='*60}")
    print(f"⏱️  VALIDATION COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  TOTAL DURATION: {timedelta(seconds=int(total_duration))}")
    print(f"⏱️  Processing speed: {len(args.checkpoints)} checkpoints in {total_duration:.1f} seconds")
    print(f"⏱️  Average per checkpoint: {total_duration/len(args.checkpoints):.1f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()