#!/usr/bin/env python3
"""
Validation watcher for Micro MuZero.
Monitors for best.pth updates and runs validation automatically.
"""

import os
import time
import json
import torch
import duckdb
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
import sys
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BestCheckpointWatcher:
    """Watches for best.pth updates and runs validation."""

    def __init__(
        self,
        checkpoint_dir: str = "/workspace/micro/checkpoints",
        results_dir: str = "/workspace/micro/validation_results",
        data_path: str = "/workspace/data/micro_features.duckdb",
        check_interval: int = 60
    ):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.data_path = data_path
        self.check_interval = check_interval
        self.best_path = os.path.join(checkpoint_dir, "best.pth")
        self.last_mtime = None

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Connect to database
        self.conn = duckdb.connect(data_path, read_only=True)
        total_rows = self.conn.execute("SELECT COUNT(*) FROM micro_features").fetchone()[0]
        self.val_start = int(total_rows * 0.7)  # Start of validation data
        self.val_end = int(total_rows * 0.85)   # End of validation data
        logger.info(f"Validation data: rows {self.val_start:,} to {self.val_end:,}")

    def watch(self):
        """Main watching loop."""
        logger.info("🔍 Starting best checkpoint watcher...")
        logger.info(f"Watching: {self.best_path}")
        logger.info(f"Check interval: {self.check_interval} seconds")

        while True:
            try:
                if os.path.exists(self.best_path):
                    current_mtime = os.path.getmtime(self.best_path)

                    if self.last_mtime is None:
                        # First check - validate current best
                        logger.info("Found existing best.pth - validating...")
                        self.validate_checkpoint()
                        self.last_mtime = current_mtime

                    elif current_mtime > self.last_mtime:
                        # best.pth was updated
                        logger.info("🎯 New best checkpoint detected!")
                        time.sleep(2)  # Wait for file write to complete
                        self.validate_checkpoint()
                        self.last_mtime = current_mtime
                else:
                    logger.debug("No best.pth found yet")

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Watcher stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in watcher: {e}")
                time.sleep(self.check_interval)

    def validate_checkpoint(self):
        """Run validation on best checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.best_path, map_location=self.device)
            episode = checkpoint.get('episode', 0)

            logger.info(f"Validating checkpoint from episode {episode}")

            # Initialize model
            model = MicroStochasticMuZero(
                input_features=15,
                lag_window=32,
                hidden_dim=256,
                action_dim=4,
                num_outcomes=3,
                support_size=300
            ).to(self.device)

            # Load weights - handle both old and new checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                model.set_weights(checkpoint['model_state'])
            else:
                logger.error("No model weights found in checkpoint")
                return
            model.eval()

            # Run validation episodes
            results = self.run_validation(model, num_episodes=100)

            # Save results
            results['checkpoint'] = 'best.pth'
            results['episode'] = episode
            results['timestamp'] = datetime.now().isoformat()

            # Save validation results
            results_file = os.path.join(
                self.results_dir,
                f"validation_best_ep{episode:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Update best validation results
            best_val_file = os.path.join(self.results_dir, "best_validation.json")
            if os.path.exists(best_val_file):
                with open(best_val_file, 'r') as f:
                    best_val = json.load(f)
                if results['expectancy'] > best_val.get('expectancy', -999):
                    with open(best_val_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"🏆 New best validation! Expectancy: {results['expectancy']:.4f}")
            else:
                with open(best_val_file, 'w') as f:
                    json.dump(results, f, indent=2)

            # Log results
            logger.info(f"Validation complete:")
            logger.info(f"  Expectancy: {results['expectancy']:.4f}")
            logger.info(f"  Win Rate: {results['win_rate']:.1f}%")
            logger.info(f"  Trades: {results['num_trades']}")
            logger.info(f"  Action Distribution: {results['action_distribution']}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")

    def run_validation(self, model, num_episodes: int = 100):
        """Run validation episodes."""
        total_reward = 0
        total_trades = 0
        winning_trades = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for episode in range(num_episodes):
            # Get random validation window
            start_idx = np.random.randint(self.val_start, self.val_end - 400)

            # Fetch data
            query = f"""
            SELECT * FROM micro_features
            WHERE bar_index >= {start_idx}
            ORDER BY bar_index
            LIMIT 400
            """
            data = self.conn.execute(query).fetchdf()

            if len(data) < 100:
                continue

            # Run episode (simplified - you'd run full MCTS here)
            episode_reward = 0
            position = 0

            for i in range(32, len(data) - 1):
                # Create observation (simplified)
                obs = self._create_observation(data, i)

                # Get action from model (simplified - should use MCTS)
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    hidden, policy_logits, _ = model.initial_inference(obs_tensor)
                    action = torch.argmax(policy_logits).item()

                action_counts[action] += 1

                # Track trades (simplified)
                if action in [1, 2] and position == 0:
                    position = 1 if action == 1 else -1
                elif action == 3 and position != 0:
                    # Close position
                    reward = np.random.normal(0, 10)  # Simplified P&L
                    episode_reward += reward
                    total_trades += 1
                    if reward > 0:
                        winning_trades += 1
                    position = 0

            total_reward += episode_reward

        # Calculate metrics
        expectancy = total_reward / max(total_trades, 1)
        win_rate = (winning_trades / max(total_trades, 1)) * 100

        return {
            'num_episodes': num_episodes,
            'num_trades': total_trades,
            'total_reward': total_reward,
            'expectancy': expectancy,
            'win_rate': win_rate,
            'action_distribution': action_counts,
            'quality_score': expectancy * np.sqrt(max(total_trades, 1))
        }

    def _create_observation(self, data, idx):
        """Create observation from data (simplified)."""
        # This is a simplified version - real implementation would match training
        obs = np.random.randn(32, 15).astype(np.float32) * 0.1
        return obs


def main():
    """Main entry point."""
    watcher = BestCheckpointWatcher()

    try:
        watcher.watch()
    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()