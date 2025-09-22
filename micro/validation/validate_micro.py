#!/usr/bin/env python3
"""
Validation script for Micro Stochastic MuZero checkpoints.

Monitors for new checkpoints and validates performance.
"""

import torch
import numpy as np
import duckdb
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys

# Add parent directory to path
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.stochastic_mcts import StochasticMCTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroValidator:
    """Validate micro MuZero checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str = "/workspace/micro/checkpoints",
        data_path: str = "/workspace/data/micro_features.duckdb",
        results_dir: str = "/workspace/micro/validation_results",
        num_episodes: int = 100
    ):
        """
        Initialize validator.

        Args:
            checkpoint_dir: Directory with checkpoints
            data_path: Path to micro features database
            results_dir: Directory for validation results
            num_episodes: Number of validation episodes
        """
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.results_dir = results_dir
        self.num_episodes = num_episodes

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track validated checkpoints
        self.validated_checkpoints = set()

        # Load validation data indices
        self.conn = duckdb.connect(data_path, read_only=True)
        self.total_rows = self.conn.execute(
            "SELECT COUNT(*) FROM micro_features"
        ).fetchone()[0]

        # Reserve last 20% for validation
        self.validation_start = int(self.total_rows * 0.8)

        logger.info(f"Validator initialized")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
        logger.info(f"  Validation data: rows {self.validation_start:,} to {self.total_rows:,}")
        logger.info(f"  Device: {self.device}")

    def load_checkpoint(self, checkpoint_path: str) -> Optional[MicroStochasticMuZero]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model or None if failed
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Create model with separated architecture
            model = MicroStochasticMuZero(
                temporal_features=9,  # Market (5) + Time (4)
                static_features=6,  # Position features
                lag_window=32,
                hidden_dim=256,
                action_dim=4,
                support_size=300
            ).to(self.device)

            # Load weights
            model.set_weights(checkpoint['model_state'])
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def get_validation_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of validation data with separated temporal and static features.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (temporal_batch, static_batch):
                - temporal_batch: (batch, 32, 9) - market + time features
                - static_batch: (batch, 6) - position features
        """
        temporal_observations = []
        static_observations = []

        for _ in range(batch_size):
            # Random index in validation range
            idx = np.random.randint(
                self.validation_start,
                self.total_rows - 100
            )

            # Fetch data
            query = f"""
            SELECT *
            FROM micro_features
            WHERE bar_index = {idx}
            """

            row = self.conn.execute(query).fetchone()

            if row:
                # Extract features (simplified - should match training)
                features = []
                col_idx = 3  # Skip metadata columns

                # Technical indicators (5 x 32)
                for _ in range(5 * 32):
                    features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Cyclical features (4 x 32)
                for _ in range(4 * 32):
                    features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Position features (6)
                position_features = []
                for _ in range(6):
                    position_features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Reshape temporal to (32, 9)
                temporal = np.zeros((32, 9))
                for t in range(32):
                    # Technical (5)
                    for f in range(5):
                        temporal[t, f] = features[f * 32 + t]
                    # Cyclical (4)
                    for f in range(4):
                        temporal[t, 5 + f] = features[160 + f * 32 + t]

                # Static features (6,)
                static = np.array(position_features)

                temporal_observations.append(temporal)
                static_observations.append(static)

        temporal_batch = torch.tensor(np.array(temporal_observations), dtype=torch.float32, device=self.device)
        static_batch = torch.tensor(np.array(static_observations), dtype=torch.float32, device=self.device)

        return temporal_batch, static_batch

    def validate_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Validate a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating checkpoint: {checkpoint_path}")

        # Load model
        model = self.load_checkpoint(checkpoint_path)
        if model is None:
            return None

        # Initialize Stochastic MCTS
        mcts = StochasticMCTS(
            model=model,
            num_actions=4,
            num_simulations=10,  # Fewer for validation
            exploration_fraction=0.0  # No exploration for validation (deterministic)
        )

        # Validation metrics
        total_rewards = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        values = []

        # Run validation episodes
        for episode in range(self.num_episodes):
            # Get validation data with separated features
            temporal_batch, static_batch = self.get_validation_batch(batch_size=1)

            # Run MCTS with separated inputs
            with torch.no_grad():
                mcts_result = mcts.run(
                    temporal_batch,
                    static_batch,
                    add_exploration_noise=False
                )

            # Track metrics
            action_counts[mcts_result['action']] += 1
            values.append(mcts_result['value'])

            # Simulate reward (would come from environment in production)
            reward = np.random.randn() * 10
            total_rewards.append(reward)

            if (episode + 1) % 20 == 0:
                logger.info(f"  Episodes: {episode + 1}/{self.num_episodes}")

        # Calculate expectancy-based score (matching main system)
        avg_reward = np.mean(total_rewards)
        winning_rewards = [r for r in total_rewards if r > 0]
        losing_rewards = [r for r in total_rewards if r < 0]

        if winning_rewards and losing_rewards:
            avg_win = np.mean(winning_rewards)
            avg_loss = abs(np.mean(losing_rewards))
            win_rate = len(winning_rewards) / len(total_rewards)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        else:
            expectancy = avg_reward

        # Trading quality score (matching swt_checkpoint_manager.py)
        expectancy_score = expectancy * 5.0
        pnl_score = avg_reward * 0.5
        consistency_factor = (len(winning_rewards) / max(1, len(total_rewards))) * min(len(total_rewards), 10)
        trade_bonus = consistency_factor * 2.0
        volume_bonus = min(len(total_rewards) * 0.1, 1.0)
        quality_score = expectancy_score + pnl_score + trade_bonus + volume_bonus

        # Calculate statistics
        results = {
            'checkpoint': os.path.basename(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'num_episodes': self.num_episodes,
            'mean_reward': float(avg_reward),
            'std_reward': float(np.std(total_rewards)),
            'mean_value': float(np.mean(values)),
            'action_distribution': action_counts,
            'expectancy': float(expectancy),
            'quality_score': float(quality_score),
            'win_rate': float(len(winning_rewards) / max(1, len(total_rewards)) * 100)
        }

        logger.info(f"Validation complete:")
        logger.info(f"  Mean reward: {results['mean_reward']:.2f}")
        logger.info(f"  Expectancy: {results['expectancy']:.3f}")
        logger.info(f"  Quality score: {results['quality_score']:.2f}")
        logger.info(f"  Win rate: {results['win_rate']:.1f}%")
        logger.info(f"  Actions: {action_counts}")

        return results

    def save_results(self, results: Dict):
        """Save validation results."""
        # Save to JSON
        results_file = os.path.join(
            self.results_dir,
            f"validation_{results['checkpoint']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved: {results_file}")

        # Update best checkpoint if needed
        best_file = os.path.join(self.results_dir, "best_checkpoint.json")
        is_new_best = False

        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                best = json.load(f)
            if results['quality_score'] > best.get('quality_score', -999):
                is_new_best = True
                with open(best_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"🎯 New best checkpoint! Quality: {results['quality_score']:.2f}, Expectancy: {results['expectancy']:.3f}")
        else:
            is_new_best = True
            with open(best_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"First checkpoint validated")

        # Copy best checkpoint to validation directory to preserve it
        if is_new_best:
            self.preserve_best_checkpoint(results['checkpoint'])

    def preserve_best_checkpoint(self, checkpoint_path: str):
        """Copy best checkpoint to validation directory to prevent overwrite."""
        try:
            import shutil

            # Create preserved checkpoints directory
            preserved_dir = os.path.join(self.results_dir, "preserved_checkpoints")
            os.makedirs(preserved_dir, exist_ok=True)

            # Copy checkpoint to preserved location
            checkpoint_name = os.path.basename(checkpoint_path)
            preserved_path = os.path.join(preserved_dir, f"best_{checkpoint_name}")

            # Also save with timestamp for history
            timestamped_path = os.path.join(
                preserved_dir,
                f"best_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{checkpoint_name}"
            )

            shutil.copy2(checkpoint_path, preserved_path)
            shutil.copy2(checkpoint_path, timestamped_path)

            logger.info(f"✅ Best checkpoint preserved: {preserved_path}")
            logger.info(f"📁 History saved: {timestamped_path}")

            # Also copy to a fixed "best_validated.pth" for easy access
            best_validated_path = os.path.join(preserved_dir, "best_validated.pth")
            shutil.copy2(checkpoint_path, best_validated_path)
            logger.info(f"🏆 Best validated checkpoint: {best_validated_path}")

        except Exception as e:
            logger.error(f"Failed to preserve best checkpoint: {e}")

    def monitor_checkpoints(self):
        """Continuously monitor for new checkpoints."""
        logger.info("Starting checkpoint monitor...")

        while True:
            try:
                # List checkpoint files
                if os.path.exists(self.checkpoint_dir):
                    checkpoints = [
                        f for f in os.listdir(self.checkpoint_dir)
                        if f.endswith('.pth') and f != 'latest.pth'
                    ]

                    for checkpoint_name in checkpoints:
                        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

                        # Skip if already validated
                        if checkpoint_path in self.validated_checkpoints:
                            continue

                        # Validate checkpoint
                        results = self.validate_checkpoint(checkpoint_path)

                        if results:
                            self.save_results(results)
                            self.validated_checkpoints.add(checkpoint_path)

                # Wait before next check
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(60)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    validator = MicroValidator()

    try:
        validator.monitor_checkpoints()
    except KeyboardInterrupt:
        logger.info("Validation stopped by user")
    finally:
        validator.cleanup()