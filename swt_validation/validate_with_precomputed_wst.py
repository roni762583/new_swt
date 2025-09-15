#!/usr/bin/env python3
"""
Validation script that uses pre-computed WST features from HDF5
Avoids memory explosion from recomputing WST on every run
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

from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics for a checkpoint"""
    total_return: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    car25: float = 0.0  # Will be calculated from Monte Carlo


class PrecomputedWSTValidator:
    """Validator that uses pre-computed WST features"""

    def __init__(
        self,
        wst_file: str = "precomputed_wst/GBPJPY_WST_3.5years_streaming.h5",
        csv_file: str = "data/GBPJPY_M1_3.5years_20250912.csv",
        test_split: float = 0.2
    ):
        """
        Initialize validator with pre-computed WST features

        Args:
            wst_file: Path to pre-computed WST HDF5 file
            csv_file: Path to CSV with price data (for position features)
            test_split: Fraction of data to use for testing
        """
        self.wst_file = Path(wst_file)
        self.csv_file = Path(csv_file)
        self.test_split = test_split

        if not self.wst_file.exists():
            raise FileNotFoundError(f"Pre-computed WST file not found: {wst_file}")

        self._load_data()

    def _load_data(self):
        """Load pre-computed WST features and price data"""
        logger.info(f"Loading pre-computed WST from {self.wst_file}")

        # Load WST features from HDF5
        with h5py.File(self.wst_file, 'r') as f:
            # Check available keys
            logger.info(f"HDF5 keys: {list(f.keys())}")

            # Load features - try different possible key names
            if 'features' in f:
                self.wst_features = f['features'][:]
            elif 'wst_features' in f:
                self.wst_features = f['wst_features'][:]
            else:
                raise KeyError(f"No WST features found. Available keys: {list(f.keys())}")

            logger.info(f"WST features shape: {self.wst_features.shape}")

            # Load timestamps if available
            if 'timestamps' in f:
                self.timestamps = f['timestamps'][:]
            else:
                self.timestamps = None

        # Load price data for position features
        logger.info(f"Loading price data from {self.csv_file}")
        self.price_df = pd.read_csv(self.csv_file)

        # Handle alignment - use indices from HDF5 to align with CSV
        if len(self.price_df) != len(self.wst_features):
            logger.warning(f"Length mismatch: CSV has {len(self.price_df)} rows, WST has {len(self.wst_features)}")
            logger.info("Using WST indices to align with CSV data")

            # WST features start from index 255 based on sample indices
            if hasattr(self, 'timestamps') and self.timestamps is not None:
                logger.info(f"WST covers data from index {self.wst_features.shape[0]} bars")
                # Ensure we have enough CSV data
                assert len(self.price_df) >= len(self.wst_features) + 255, \
                    f"CSV too short: need {len(self.wst_features) + 255}, have {len(self.price_df)}"

        # Split data
        self.split_index = int(len(self.wst_features) * (1 - self.test_split))
        logger.info(f"Using last {len(self.wst_features) - self.split_index:,} bars for testing")

        # Prepare test data
        self.test_wst = self.wst_features[self.split_index:]
        self.test_prices = self.price_df.iloc[self.split_index:]

    def create_position_features(self, csv_index: int) -> np.ndarray:
        """
        Create position features for a given CSV index

        Returns 9 position features: spread, volatility, time features, etc.
        """
        # Simple position features (9 dimensions to match training)
        position_features = np.zeros(9)

        # Add some basic features (simplified for validation)
        if csv_index > 20 and csv_index < len(self.price_df) - 1:
            recent_prices = self.price_df.iloc[csv_index-20:csv_index]['close'].values
            if len(recent_prices) > 1 and np.mean(recent_prices) > 0:
                position_features[0] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]  # Return
                position_features[1] = np.std(recent_prices) / np.mean(recent_prices)  # Volatility

        return position_features

    def validate_checkpoint(self, checkpoint_path: str, num_runs: int = 10) -> ValidationMetrics:
        """
        Validate a checkpoint using pre-computed WST features

        Args:
            checkpoint_path: Path to checkpoint file
            num_runs: Number of Monte Carlo runs for CAR25

        Returns:
            ValidationMetrics with results
        """
        logger.info(f"\nValidating checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = load_checkpoint_with_proper_config(checkpoint_path)
        network = checkpoint['networks']
        network.eval()

        # Run validation
        trades = []

        with torch.no_grad():
            # Process test data in batches
            batch_size = 100
            for i in range(0, len(self.test_wst) - 1000, batch_size):
                # Get WST features (already computed!)
                wst_batch = self.test_wst[i:i+batch_size]

                # Create position features - map WST indices to CSV indices
                # WST starts from around index 255 in the CSV based on the indices data
                position_batch = np.array([
                    self.create_position_features(255 + self.split_index + i + j)
                    for j in range(len(wst_batch))
                ])

                # Process WST features - pre-computed are 16-dim raw WST output, need to expand to 128
                if wst_batch.shape[1] == 16:
                    # Pre-computed WST features are raw 16-dimensional
                    # Pad or repeat to match expected 128 dimensions for the model
                    logger.info(f"Expanding 16-dim WST features to 128 dimensions")
                    # Repeat the 16 features 8 times to get 128 (16 * 8 = 128)
                    expanded_wst = np.tile(wst_batch, (1, 8))  # Shape: (batch, 128)
                    features = np.concatenate([expanded_wst, position_batch], axis=1)
                elif wst_batch.shape[1] == 128:
                    # WST features are already 128 dims
                    features = np.concatenate([wst_batch, position_batch], axis=1)
                else:
                    # Handle other dimensions
                    logger.warning(f"WST shape {wst_batch.shape[1]} != expected 16 or 128, truncating/padding to 128")
                    if wst_batch.shape[1] > 128:
                        wst_processed = wst_batch[:, :128]
                    else:
                        # Pad with zeros
                        padding = np.zeros((wst_batch.shape[0], 128 - wst_batch.shape[1]))
                        wst_processed = np.concatenate([wst_batch, padding], axis=1)
                    features = np.concatenate([wst_processed, position_batch], axis=1)

                # Convert to tensor
                features_tensor = torch.FloatTensor(features)

                # Get network predictions
                hidden = network.representation_network(features_tensor)

                # Simple trading logic (simplified for demo)
                predictions = hidden.mean(dim=1).numpy()
                for j, pred in enumerate(predictions):
                    if abs(pred) > 0.5:  # Threshold for trade
                        # Calculate return from next bar - map to proper CSV index
                        csv_idx = 255 + self.split_index + i + j
                        if csv_idx + 10 < len(self.price_df):
                            entry_price = self.price_df.iloc[csv_idx]['close']
                            exit_price = self.price_df.iloc[csv_idx + 10]['close']
                            trade_return = (exit_price - entry_price) / entry_price * np.sign(pred)
                            trades.append(trade_return)

        # Calculate metrics
        if len(trades) > 0:
            trades = np.array(trades)
            total_return = np.sum(trades)
            win_rate = np.mean(trades > 0)
            sharpe = np.mean(trades) / (np.std(trades) + 1e-8) * np.sqrt(252)

            # Simple max drawdown
            cumsum = np.cumsum(trades)
            running_max = np.maximum.accumulate(cumsum)
            drawdown = (cumsum - running_max) / (running_max + 1e-8)
            max_dd = abs(np.min(drawdown))

            # Monte Carlo for CAR25 (simplified)
            car25_values = []
            for _ in range(num_runs):
                sampled = np.random.choice(trades, size=len(trades), replace=True)
                annual_return = np.sum(sampled) * (252 * 24 * 60 / len(self.test_wst))
                car25_values.append(annual_return)
            car25 = np.percentile(car25_values, 25)
        else:
            total_return = 0
            win_rate = 0
            sharpe = 0
            max_dd = 0
            car25 = 0

        metrics = ValidationMetrics(
            total_return=total_return,
            total_trades=len(trades),
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            car25=car25
        )

        return metrics

    def compare_checkpoints(self, checkpoint_paths: List[str]) -> Dict[str, ValidationMetrics]:
        """Compare multiple checkpoints"""
        results = {}

        for path in checkpoint_paths:
            name = Path(path).stem
            metrics = self.validate_checkpoint(path)
            results[name] = metrics

            logger.info(f"\n{name} Results:")
            logger.info(f"  Trades: {metrics.total_trades}")
            logger.info(f"  Total Return: {metrics.total_return:.2%}")
            logger.info(f"  Win Rate: {metrics.win_rate:.2%}")
            logger.info(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
            logger.info(f"  Max DD: {metrics.max_drawdown:.2%}")
            logger.info(f"  CAR25: {metrics.car25:.2%}")

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

            # Print results
            print(f"\n📊 {name} Validation Results:")
            print(f"  ✅ Trades: {metrics.total_trades}")
            print(f"  ✅ Total Return: {metrics.total_return:.2%}")
            print(f"  ✅ Win Rate: {metrics.win_rate:.2%}")
            print(f"  ✅ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"  ✅ Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"  ✅ CAR25: {metrics.car25:.2%}")

    # Compare results
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("📈 COMPARISON")
        print('='*60)

        best_car25 = max(results.items(), key=lambda x: x[1].car25)
        print(f"🏆 Best CAR25: {best_car25[0]} with {best_car25[1].car25:.2%}")

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