#!/usr/bin/env python3
"""
Idle wait script for live trading container.
Waits for proper checkpoint before starting live trading.
"""

import os
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main idle loop."""
    checkpoint_dir = "/workspace/micro/checkpoints"
    best_checkpoint = os.path.join(checkpoint_dir, "best.pth")

    logger.info("🤖 Live trading container started in IDLE mode")
    logger.info("Waiting for validated checkpoint before starting live trading...")
    logger.info(f"Monitoring: {best_checkpoint}")

    check_count = 0
    while True:
        try:
            # Check every minute
            time.sleep(60)
            check_count += 1

            if os.path.exists(best_checkpoint):
                # Get file info
                mtime = os.path.getmtime(best_checkpoint)
                size = os.path.getsize(best_checkpoint) / (1024 * 1024)  # MB
                age_minutes = (time.time() - mtime) / 60

                logger.info(f"[Check #{check_count}] Best checkpoint found:")
                logger.info(f"  Size: {size:.1f} MB")
                logger.info(f"  Age: {age_minutes:.0f} minutes")
                logger.info(f"  Ready for live trading when validated")
            else:
                if check_count % 10 == 0:  # Log every 10 minutes
                    logger.info(f"[Check #{check_count}] No best checkpoint yet, waiting...")

            # Show we're alive every 30 minutes
            if check_count % 30 == 0:
                logger.info(f"💓 Heartbeat - Live container healthy, idle for {check_count} minutes")

        except KeyboardInterrupt:
            logger.info("Shutting down idle wait...")
            break
        except Exception as e:
            logger.error(f"Error in idle loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()