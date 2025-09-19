#!/usr/bin/env python3
"""
Training monitor that tracks action distributions and key metrics in real-time.
Can be run alongside training to monitor progress without stopping.
"""

import time
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from collections import deque
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training progress and action distributions."""

    def __init__(self, checkpoint_dir="/workspace/micro/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.action_history = deque(maxlen=1000)  # Track last 1000 actions
        self.metrics_history = []

    def load_checkpoint(self, path="latest.pth"):
        """Load checkpoint and extract metrics."""
        checkpoint_path = self.checkpoint_dir / path

        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def analyze_buffer_actions(self, checkpoint):
        """Analyze action distribution from buffer if available."""
        # This would need access to the actual buffer
        # For now, we'll extract from training stats if saved
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if 'action_distributions' in stats and stats['action_distributions']:
                return stats['action_distributions'][-1]
        return None

    def monitor_loop(self, interval=10):
        """Main monitoring loop."""
        logger.info("Starting training monitor...")
        logger.info("Press Ctrl+C to stop")

        last_episode = -1

        while True:
            try:
                # Load latest checkpoint
                checkpoint = self.load_checkpoint()

                if checkpoint:
                    episode = checkpoint.get('episode', 0)

                    # Only update if new data
                    if episode > last_episode:
                        last_episode = episode

                        # Extract metrics
                        total_steps = checkpoint.get('total_steps', 0)
                        temperature = checkpoint.get('temperature', 1.0)

                        # Print status
                        print("\n" + "="*60)
                        print(f"TRAINING STATUS - {datetime.now().strftime('%H:%M:%S')}")
                        print("="*60)
                        print(f"Episode: {episode:,}")
                        print(f"Total Steps: {total_steps:,}")
                        print(f"Temperature: {temperature:.3f}")

                        # Check for training stats
                        if 'training_stats' in checkpoint:
                            stats = checkpoint['training_stats']

                            # Recent expectancy
                            if 'expectancies' in stats and stats['expectancies']:
                                recent = stats['expectancies'][-100:]
                                avg_exp = np.mean(recent) if recent else 0
                                print(f"Expectancy (100-ep avg): {avg_exp:.4f} pips")

                            # Recent win rate
                            if 'win_rates' in stats and stats['win_rates']:
                                recent = stats['win_rates'][-100:]
                                avg_wr = np.mean(recent) if recent else 0
                                print(f"Win Rate (100-ep avg): {avg_wr:.1f}%")

                            # Recent trade counts
                            if 'trade_counts' in stats and stats['trade_counts']:
                                recent = stats['trade_counts'][-100:]
                                avg_trades = np.mean(recent) if recent else 0
                                print(f"Trades/Episode (100-ep avg): {avg_trades:.1f}")

                        # Action distribution
                        action_dist = self.analyze_buffer_actions(checkpoint)
                        if action_dist:
                            print("\nAction Distribution:")
                            print(f"  HOLD:  {action_dist.get('HOLD', 0)*100:5.1f}%")
                            print(f"  BUY:   {action_dist.get('BUY', 0)*100:5.1f}%")
                            print(f"  SELL:  {action_dist.get('SELL', 0)*100:5.1f}%")
                            print(f"  CLOSE: {action_dist.get('CLOSE', 0)*100:5.1f}%")

                            # Check for collapse
                            hold_pct = action_dist.get('HOLD', 0) * 100
                            if hold_pct > 90:
                                print("⚠️  WARNING: Model may be collapsing to HOLD-only!")
                            elif hold_pct > 70:
                                print("⚠️  CAUTION: High HOLD percentage")

                        print("="*60)

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(interval)


def main():
    """Run the training monitor."""
    monitor = TrainingMonitor()
    monitor.monitor_loop(interval=10)  # Check every 10 seconds


if __name__ == "__main__":
    main()