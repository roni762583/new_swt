#!/usr/bin/env python3
"""
Validation watcher for Micro MuZero.
Monitors for best.pth updates and runs validation automatically.
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import subprocess

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
        self.last_validated_episode = None

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Track validation history
        self.validation_history_file = os.path.join(results_dir, "validation_history.json")
        if os.path.exists(self.validation_history_file):
            with open(self.validation_history_file, 'r') as f:
                self.validation_history = json.load(f)
        else:
            self.validation_history = {}

        logger.info(f"Watching directory: {self.checkpoint_dir}")
        logger.info(f"Results directory: {self.results_dir}")

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
        """Run validation on best checkpoint using validate_micro.py."""
        try:
            # Load checkpoint to get episode number
            checkpoint = torch.load(self.best_path, map_location='cpu')
            episode = checkpoint.get('episode', 0)

            # Skip if already validated this episode
            if self.last_validated_episode == episode:
                logger.debug(f"Episode {episode} already validated")
                return

            logger.info(f"🎯 Validating checkpoint from episode {episode}")

            # Copy best.pth to a temporary validation checkpoint
            val_checkpoint_path = os.path.join(
                "/tmp",
                f"validation_ep{episode:06d}.pth"
            )
            subprocess.run(["cp", self.best_path, val_checkpoint_path], check=True)

            # Run the full validation script
            cmd = [
                "python3",
                "/workspace/micro/validation/validate_micro.py",
                "--checkpoint", val_checkpoint_path,
                "--num-runs", "1000",  # Monte Carlo runs
                "--output-dir", self.results_dir
            ]

            logger.info("Starting validation with Monte Carlo simulation (1000 runs)...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Validation completed successfully")

                # Parse validation results from stdout
                for line in result.stdout.split('\n'):
                    if 'CAR' in line or 'Safe-f' in line or 'Expectancy' in line:
                        logger.info(f"  {line.strip()}")

                # Update tracking
                self.last_validated_episode = episode
                self.validation_history[str(episode)] = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }

                # Save history
                with open(self.validation_history_file, 'w') as f:
                    json.dump(self.validation_history, f, indent=2)

                # Clean up temporary checkpoint
                os.remove(val_checkpoint_path)

            else:
                logger.error(f"Validation failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                self.validation_history[str(episode)] = {
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed',
                    'error': result.stderr[:500]
                }

        except subprocess.TimeoutExpired:
            logger.error("Validation timed out after 10 minutes")
        except Exception as e:
            logger.error(f"Validation failed: {e}")



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