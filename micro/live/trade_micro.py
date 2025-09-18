#!/usr/bin/env python3
"""
Live trading with Micro Stochastic MuZero.

Uses incremental feature builder for real-time feature generation.
"""

import torch
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
import sys

# Add parent directories to path
sys.path.append('/workspace')

from micro.live.micro_feature_builder import MicroFeatureBuilder
from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.mcts_micro import MCTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroLiveTrader:
    """Live trading with Micro MuZero."""

    def __init__(
        self,
        checkpoint_path: str = None,
        instrument: str = "GBP_JPY",
        max_position_size: int = 1
        # REMOVED is_demo parameter - ALWAYS LIVE TRADING
    ):
        """
        Initialize live trader.

        Args:
            checkpoint_path: Path to model checkpoint (None for auto-select best)
            instrument: Trading instrument
            max_position_size: Maximum position size
        """
        # Auto-select best checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()

        self.checkpoint_path = checkpoint_path
        self.instrument = instrument
        self.max_position_size = max_position_size
        # HARD-CODED: ALWAYS LIVE TRADING - NEVER DEMO
        self.is_live_trading = True  # Permanently true

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self.load_model()
        if self.model is None:
            raise RuntimeError("Failed to load model")

        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            num_actions=4,
            discount=0.997,
            num_simulations=10  # Fewer for real-time
        )

        # Initialize feature builder with proper warmup
        self.feature_builder = MicroFeatureBuilder(
            instrument=instrument,
            lag_window=32,
            warmup_bars=100  # Need sufficient history for indicators
        )

        # Initialize with historical data to avoid NaNs
        if not self.feature_builder.initialize_from_history():
            logger.error("Failed to initialize feature builder with historical data")
            raise RuntimeError("Cannot start without proper historical data")

        # Position tracking
        self.position = {
            'side': 0,  # -1: short, 0: flat, 1: long
            'entry_price': 0.0,
            'entry_bar': 0,
            'peak_pnl': 0.0,
            'current_pnl': 0.0,
            'max_dd': 0.0,
            'accumulated_dd': 0.0
        }

        # Trading stats
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'bars_processed': 0
        }

        # Action mapping
        self.action_names = {
            0: 'HOLD',
            1: 'BUY',
            2: 'SELL',
            3: 'CLOSE'
        }

        logger.info(f"Live trader initialized")
        logger.info(f"  Model: {checkpoint_path}")
        logger.info(f"  Instrument: {instrument}")
        logger.info(f"  ⚠️ LIVE TRADING MODE: ACTIVE ⚠️")  # Hard-coded live status
        logger.info(f"  Device: {self.device}")

    def _find_best_checkpoint(self) -> str:
        """Find the best checkpoint based on validation results or use specific checkpoint."""
        # Check if we want to use specific checkpoint (Episode 3400)
        specific_checkpoint = "/workspace/micro/checkpoints/micro_checkpoint_ep003400.pth"
        if os.path.exists(specific_checkpoint):
            logger.info(f"Using specific checkpoint: Episode 3400")
            return specific_checkpoint

        # Otherwise use the best.pth
        best_path = "/workspace/micro/checkpoints/best.pth"
        if os.path.exists(best_path):
            logger.info(f"Using best checkpoint")
            return best_path

        # Fallback to latest
        latest_path = "/workspace/micro/checkpoints/latest.pth"
        logger.info(f"Using latest checkpoint as fallback")
        return latest_path

    def load_model(self) -> Optional[MicroStochasticMuZero]:
        """Load model from checkpoint."""
        try:
            if not os.path.exists(self.checkpoint_path):
                logger.error(f"Checkpoint not found: {self.checkpoint_path}")
                return None

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Create model
            model = MicroStochasticMuZero(
                input_features=15,
                lag_window=32,
                hidden_dim=256,
                action_dim=4,
                z_dim=16,
                support_size=300
            ).to(self.device)

            # Load weights
            model.set_weights(checkpoint['model_state'])
            model.eval()

            logger.info(f"Model loaded from episode {checkpoint.get('episode', 'unknown')}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def update_position_state(self, current_price: float):
        """
        Update position tracking.

        Args:
            current_price: Current market price
        """
        if self.position['side'] != 0:
            # Calculate P&L
            if self.position['side'] == 1:  # Long
                self.position['current_pnl'] = (current_price - self.position['entry_price']) * 100
            else:  # Short
                self.position['current_pnl'] = (self.position['entry_price'] - current_price) * 100

            # Update peak and drawdown
            self.position['peak_pnl'] = max(self.position['peak_pnl'], self.position['current_pnl'])
            current_dd = self.position['peak_pnl'] - self.position['current_pnl']
            self.position['max_dd'] = max(self.position['max_dd'], current_dd)

            # Accumulate drawdown
            if current_dd > 0:
                self.position['accumulated_dd'] += current_dd * 0.01  # Small increment

        # Update feature builder with position state
        self.feature_builder.update_position_state(
            position_side=self.position['side'],
            position_pips=self.position['current_pnl'],
            bars_since_entry=self.stats['bars_processed'] - self.position['entry_bar'],
            pips_from_peak=self.position['current_pnl'] - self.position['peak_pnl'],
            max_drawdown_pips=-abs(self.position['max_dd']),
            accumulated_dd=self.position['accumulated_dd']
        )

    def execute_action(self, action: int, current_price: float):
        """
        Execute trading action.

        Args:
            action: Action index (0-3)
            current_price: Current market price
        """
        action_name = self.action_names[action]
        logger.info(f"Action: {action_name} at {current_price:.3f}")

        # NEVER USE DEMO MODE - ALWAYS EXECUTE REAL TRADES
        if False:  # self.is_demo DISABLED - Always live trading
            # Demo mode - DISABLED
            if action == 1 and self.position['side'] <= 0:  # BUY
                if self.position['side'] == -1:  # Close short first
                    self.close_position(current_price)
                # Open long
                self.position['side'] = 1
                self.position['entry_price'] = current_price
                self.position['entry_bar'] = self.stats['bars_processed']
                self.position['peak_pnl'] = 0
                self.position['max_dd'] = 0
                self.stats['total_trades'] += 1
                logger.info(f"  📈 Opened LONG at {current_price:.3f}")

            elif action == 2 and self.position['side'] >= 0:  # SELL
                if self.position['side'] == 1:  # Close long first
                    self.close_position(current_price)
                # Open short
                self.position['side'] = -1
                self.position['entry_price'] = current_price
                self.position['entry_bar'] = self.stats['bars_processed']
                self.position['peak_pnl'] = 0
                self.position['max_dd'] = 0
                self.stats['total_trades'] += 1
                logger.info(f"  📉 Opened SHORT at {current_price:.3f}")

            elif action == 3 and self.position['side'] != 0:  # CLOSE
                self.close_position(current_price)

        else:
            # ALWAYS LIVE TRADING - Execute real OANDA orders
            logger.warning(f"⚠️ LIVE TRADE EXECUTION: {action_name} at {current_price:.3f}")
            # TODO: Implement OANDA API calls here
            logger.error("OANDA API integration needed for live execution")

    def close_position(self, current_price: float):
        """Close current position."""
        if self.position['side'] == 0:
            return

        # Calculate final P&L
        if self.position['side'] == 1:
            final_pnl = (current_price - self.position['entry_price']) * 100
        else:
            final_pnl = (self.position['entry_price'] - current_price) * 100

        # Update stats
        self.stats['total_pnl'] += final_pnl
        if final_pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1

        logger.info(f"  💰 Closed position: P&L = {final_pnl:.1f} pips")

        # Reset position
        self.position['side'] = 0
        self.position['current_pnl'] = 0
        self.position['accumulated_dd'] = 0

    def process_new_bar(self, close_price: float, timestamp: datetime) -> int:
        """
        Process new bar and decide action.

        Args:
            close_price: Bar close price
            timestamp: Bar timestamp

        Returns:
            Selected action (0-3)
        """
        # Update position state
        self.update_position_state(close_price)

        # Add new bar to feature builder
        if not self.feature_builder.add_new_bar(close_price, timestamp):
            logger.warning("Failed to add new bar")
            return 0  # HOLD

        # Get feature vector (32, 15) for model
        model_input = self.feature_builder.get_feature_vector()
        if model_input is None:
            logger.warning("Insufficient feature history for model input")
            return 0  # HOLD

        # Convert to tensor
        observation = torch.tensor(
            model_input,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Run MCTS
        with torch.no_grad():
            mcts_result = self.mcts.run(
                observation,
                add_exploration_noise=False,
                temperature=0  # Deterministic for live trading
            )

        action = mcts_result['action']
        value = mcts_result['value']

        logger.info(f"Bar {self.stats['bars_processed']}: Close={close_price:.3f}, Action={self.action_names[action]}, Value={value:.2f}")

        # Execute action
        self.execute_action(action, close_price)

        self.stats['bars_processed'] += 1
        return action

    def print_stats(self):
        """Print current trading statistics."""
        win_rate = self.stats['winning_trades'] / max(1, self.stats['total_trades']) * 100

        logger.info("=" * 50)
        logger.info("Trading Statistics:")
        logger.info(f"  Bars processed: {self.stats['bars_processed']}")
        logger.info(f"  Total trades: {self.stats['total_trades']}")
        logger.info(f"  Win rate: {win_rate:.1f}%")
        logger.info(f"  Total P&L: {self.stats['total_pnl']:.1f} pips")
        logger.info(f"  Current position: {self.position['side']}")
        logger.info("=" * 50)

    def run(self):
        """Main trading loop."""
        logger.info("Starting live trading...")

        # Initialize feature builder with history
        if not self.feature_builder.initialize_from_history():
            logger.error("Failed to initialize feature builder")
            return

        logger.info("Feature builder initialized. Starting main loop...")

        # Main trading loop
        last_price = None

        while True:
            try:
                # Get latest price
                current_price = self.feature_builder.data_puller.get_latest_m1_close()

                if current_price and current_price != last_price:
                    # New bar detected
                    timestamp = datetime.now()
                    self.process_new_bar(current_price, timestamp)
                    last_price = current_price

                    # Print stats every 10 bars
                    if self.stats['bars_processed'] % 10 == 0:
                        self.print_stats()

                # Wait before next check
                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                logger.info("Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error

        # Final stats
        self.print_stats()
        logger.info("Trading session ended")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live trading with Micro MuZero")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint path (default: auto-select best, use 'ep3400' for Episode 3400)"
    )
    parser.add_argument(
        "--instrument",
        default="GBP_JPY",
        help="Trading instrument"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: demo)"
    )

    args = parser.parse_args()

    # Handle special checkpoint names
    checkpoint_path = args.checkpoint
    if checkpoint_path == 'ep3400':
        checkpoint_path = "/workspace/micro/checkpoints/micro_checkpoint_ep003400.pth"
        logger.info("Using Episode 3400 checkpoint (best validation performance)")
    elif checkpoint_path == 'best':
        checkpoint_path = "/workspace/micro/checkpoints/best.pth"
    elif checkpoint_path == 'latest':
        checkpoint_path = "/workspace/micro/checkpoints/latest.pth"

    trader = MicroLiveTrader(
        checkpoint_path=checkpoint_path,
        instrument=args.instrument
        # NO is_demo parameter - PERMANENTLY LIVE TRADING
    )

    trader.run()