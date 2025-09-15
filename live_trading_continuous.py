#!/usr/bin/env python3
"""
Continuous Live Trading Wrapper
Runs indefinitely with automatic restarts and weekend handling
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContinuousLiveTradingWrapper:
    """Wrapper to run live trading indefinitely with proper handling"""

    def __init__(self, checkpoint_path: str = "checkpoints/episode_10_best.pth"):
        self.checkpoint_path = checkpoint_path
        self.running = True
        self.current_checkpoint = checkpoint_path

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"⛔ Received signal {signum}, shutting down gracefully...")
        self.running = False

    def is_market_open(self) -> bool:
        """Check if forex market is open (Sunday 5PM EST to Friday 5PM EST)"""
        now = datetime.now()
        weekday = now.weekday()

        # Forex is closed from Friday 5PM EST to Sunday 5PM EST
        # weekday: 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

        if weekday == 5:  # Saturday - always closed
            return False
        elif weekday == 6:  # Sunday - check if after 5PM EST
            # Simplified check - you'd need proper timezone handling in production
            if now.hour >= 17:  # After 5PM
                return True
            return False
        elif weekday == 4:  # Friday - check if before 5PM EST
            if now.hour < 17:  # Before 5PM
                return True
            return False
        else:  # Monday-Thursday - always open
            return True

    def check_for_new_checkpoint(self) -> str:
        """Check if a newer validated checkpoint is available"""
        # Check for best validated checkpoint
        best_checkpoint = Path("/shared/checkpoints/best_validated.pth")
        if best_checkpoint.exists():
            return str(best_checkpoint)

        # Check for latest checkpoint
        latest_checkpoint = Path("/shared/checkpoints/latest.pth")
        if latest_checkpoint.exists():
            return str(latest_checkpoint)

        return self.checkpoint_path

    async def run_trading_session(self):
        """Run continuous trading (no session limit)"""
        logger.info(f"🚀 Starting continuous trading with checkpoint: {self.current_checkpoint}")

        try:
            # Import here to avoid issues if module isn't available
            from live_trading_main import SWTLiveTradingOrchestrator

            # Create and run orchestrator
            orchestrator = SWTLiveTradingOrchestrator(
                config_path="config/live.yaml",
                checkpoint_path=self.current_checkpoint
            )
            await orchestrator.initialize_trading_system()
            await orchestrator.run_live_trading_loop()

        except ImportError:
            # Fallback simulation mode
            logger.warning("Live trading module not found, running in simulation mode")
            await self.simulate_trading()

        except Exception as e:
            logger.error(f"Trading session error: {e}")
            await asyncio.sleep(60)  # Wait before retry

    async def simulate_trading(self):
        """Simulate trading for testing when live module isn't available"""
        logger.info("📊 Running simulated trading session...")

        for i in range(60):  # Simulate 60 minutes
            if not self.running:
                break

            # Simulate trading activity
            if i % 10 == 0:
                logger.info(f"   Trading minute {i}: Analyzing market...")

            await asyncio.sleep(1)  # Speed up for testing (normally would be 60)

    async def run_forever(self):
        """Main loop that runs indefinitely"""
        logger.info("🔄 Starting continuous live trading wrapper")
        logger.info("   Press Ctrl+C to stop")

        session_count = 0

        while self.running:
            session_count += 1

            # Check if market is open - pause trading but keep container running
            if not self.is_market_open():
                logger.info("🔴 Market is closed (weekend). Pausing trading activity...")
                logger.info("   Container remains active, will resume when market opens")
                await asyncio.sleep(300)  # Check every 5 minutes
                continue

            logger.info(f"🟢 Market is open. Starting session #{session_count}")

            # Check for updated checkpoint
            new_checkpoint = self.check_for_new_checkpoint()
            if new_checkpoint != self.current_checkpoint:
                logger.info(f"🔄 Switching to new checkpoint: {new_checkpoint}")
                self.current_checkpoint = new_checkpoint

            # Run continuous trading (no session restarts)
            try:
                await self.run_trading_session()
            except Exception as e:
                logger.error(f"Trading error: {e}")
                logger.info("⚠️ Restarting trading system in 30 seconds...")
                await asyncio.sleep(30)  # Wait before retry

        logger.info("👋 Continuous trading wrapper stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Continuous Live Trading")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/episode_10_best.pth",
        help="Initial checkpoint to use"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Create and run wrapper
    wrapper = ContinuousLiveTradingWrapper(checkpoint_path=args.checkpoint)

    try:
        asyncio.run(wrapper.run_forever())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()