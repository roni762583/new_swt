"""
Improved live trading script with rolling std gating and safety controls.
DEMO/SIMULATION - Requires OANDA API for real trading.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
from datetime import datetime
import json
import time
from typing import Dict, Optional, Tuple
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_improved import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LiveTradingSystem:
    """Live trading system with safety controls and gating."""

    def __init__(
        self,
        model_path: str,
        instrument: str = "GBPJPY",
        max_daily_loss: float = 0.02,
        max_drawdown: float = 0.05,
        position_size: int = 1000,
        demo_mode: bool = True
    ):
        self.model_path = model_path
        self.instrument = instrument
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.position_size = position_size
        self.demo_mode = demo_mode

        # Trading state
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance

        # Daily tracking
        self.daily_start_balance = self.balance
        self.daily_trades = []
        self.consecutive_losses = 0

        # Price history for rolling std
        self.price_history = deque(maxlen=GATING_CONFIG["sigma_window"] * 2)
        self.sigma_window = GATING_CONFIG["sigma_window"]
        self.k_threshold = GATING_CONFIG["k_threshold_end"]  # Use strict for live
        self.m_spread = GATING_CONFIG["m_spread"]
        self.min_threshold_pips = GATING_CONFIG["min_threshold_pips"]

        # Performance tracking
        self.trade_history = []
        self.gate_history = []
        self.total_gates = 0
        self.false_rejects = 0

        # Safety flags
        self.trading_enabled = True
        self.halt_reason = None

        # Load model
        self.model = self._load_model()

        # Get instrument config
        config = get_instrument_config(instrument)
        self.pip_value = config["pip_value"]
        self.spread = config["spread"]

        logger.info(f"Live trading system initialized")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Instrument: {instrument}")
        logger.info(f"  Position Size: {position_size}")
        logger.info(f"  Demo Mode: {demo_mode}")
        logger.info(f"  Gating: σ-based with k={self.k_threshold}")

    def _load_model(self) -> PPO:
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model (simplified - real implementation needs proper env setup)
        model = PPO.load(self.model_path)
        logger.info(f"Model loaded from: {self.model_path}")
        return model

    def _calculate_rolling_std(self) -> Tuple[float, float]:
        """Calculate rolling standard deviation and threshold."""
        if len(self.price_history) < self.sigma_window:
            return 0.0, self.min_threshold_pips

        prices = list(self.price_history)[-self.sigma_window:]
        returns_pips = np.diff(prices) * 100  # Convert to pips

        if len(returns_pips) > 0:
            rolling_std = np.std(returns_pips)
        else:
            rolling_std = 0.0

        # Calculate threshold
        threshold = max(
            self.k_threshold * rolling_std,
            self.m_spread * self.spread,
            self.min_threshold_pips
        )

        return rolling_std, threshold

    def _check_gate(self, action: int, recent_move_pips: float) -> bool:
        """Check if trade should be gated."""
        if action == 0 or self.position != 0:  # Only gate new entries
            return True

        rolling_std, threshold = self._calculate_rolling_std()
        gate_allowed = abs(recent_move_pips) >= threshold

        # Track gating
        self.total_gates += 1
        if not gate_allowed:
            self.false_rejects += 1
            logger.info(f"Gate blocked: move={recent_move_pips:.2f} < threshold={threshold:.2f} (σ={rolling_std:.2f})")

        self.gate_history.append({
            'timestamp': datetime.now().isoformat(),
            'allowed': gate_allowed,
            'move': recent_move_pips,
            'threshold': threshold,
            'rolling_std': rolling_std
        })

        return gate_allowed

    def _check_safety_limits(self) -> bool:
        """Check if safety limits are breached."""
        if not self.trading_enabled:
            return False

        # Check daily loss
        daily_loss = (self.daily_start_balance - self.balance) / self.daily_start_balance
        if daily_loss > self.max_daily_loss:
            self.trading_enabled = False
            self.halt_reason = f"Daily loss limit exceeded: {daily_loss:.2%}"
            logger.error(self.halt_reason)
            return False

        # Check max drawdown
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.trading_enabled = False
            self.halt_reason = f"Max drawdown exceeded: {current_dd:.2%}"
            logger.error(self.halt_reason)
            return False

        # Check consecutive losses
        if self.consecutive_losses >= INSTRUMENTS[self.instrument].get("max_consecutive_losses", 5):
            self.trading_enabled = False
            self.halt_reason = f"Max consecutive losses: {self.consecutive_losses}"
            logger.error(self.halt_reason)
            return False

        return True

    def _execute_trade(self, action: int, current_price: float):
        """Execute trade action."""
        if action == 0:  # Hold/Close
            if self.position != 0:
                # Close position
                exit_price = current_price - (self.spread * self.pip_value) * np.sign(self.position)
                pnl = (exit_price - self.entry_price) * self.position * self.position_size
                self.balance += pnl
                self.equity = self.balance

                # Track trade
                trade_pips = (exit_price - self.entry_price) * self.position * 100
                self.trade_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'close',
                    'pnl': pnl,
                    'pips': trade_pips
                })
                self.daily_trades.append(pnl)

                # Update consecutive losses
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                logger.info(f"Position closed: {trade_pips:.1f} pips, PnL: ${pnl:.2f}")

                # Reset position
                self.position = 0
                self.entry_price = 0.0

        elif action == 1:  # Buy
            if self.position <= 0:
                if self.position < 0:
                    # Close short first
                    self._execute_trade(0, current_price)

                # Open long
                self.position = 1
                self.entry_price = current_price + self.spread * self.pip_value
                logger.info(f"Long position opened at {self.entry_price:.5f}")

        elif action == 2:  # Sell
            if self.position >= 0:
                if self.position > 0:
                    # Close long first
                    self._execute_trade(0, current_price)

                # Open short
                self.position = -1
                self.entry_price = current_price - self.spread * self.pip_value
                logger.info(f"Short position opened at {self.entry_price:.5f}")

        # Update equity for open positions
        if self.position != 0:
            if self.position > 0:
                self.equity = self.balance + (current_price - self.entry_price) * self.position_size
            else:
                self.equity = self.balance + (self.entry_price - current_price) * self.position_size

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_current_features(self, market_data: Dict) -> np.ndarray:
        """Convert market data to model features."""
        # This would need to match the exact feature engineering from training
        # Simplified for demo
        features = np.zeros(20)  # 20 features in improved env

        # Market features (would come from real market data)
        features[0:8] = market_data.get('market_features', np.zeros(8))

        # Position features
        features[8] = float(self.position)
        if self.position != 0:
            pnl_pips = (market_data['price'] - self.entry_price) * self.position * 100
            features[9] = pnl_pips / 100  # Scaled

        # Gating features
        rolling_std, threshold = self._calculate_rolling_std()
        features[17] = rolling_std / 10  # Scaled
        features[18] = threshold / 10  # Scaled
        features[19] = 1.0  # Gate allowed (will be checked separately)

        return features

    def run_tick(self, market_data: Dict):
        """Process one market tick."""
        if not self.trading_enabled:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return

        current_price = market_data['price']
        self.price_history.append(current_price)

        # Calculate recent move
        if len(self.price_history) >= 2:
            recent_move_pips = (self.price_history[-1] - self.price_history[-2]) * 100
        else:
            recent_move_pips = 0.0

        # Get features and predict action
        features = self.get_current_features(market_data)
        action, _ = self.model.predict(features.reshape(1, -1), deterministic=True)
        action = int(action[0])

        # Check gate
        if self._check_gate(action, recent_move_pips):
            # Check safety limits
            if self._check_safety_limits():
                # Execute trade
                self._execute_trade(action, current_price)
        else:
            logger.debug(f"Action {action} gated")

        # Log status periodically
        if len(self.trade_history) > 0 and len(self.trade_history) % 10 == 0:
            self._log_performance()

    def _log_performance(self):
        """Log current performance metrics."""
        total_trades = len(self.trade_history)
        profitable = sum(1 for t in self.trade_history if t['pips'] > 0)
        total_pips = sum(t['pips'] for t in self.trade_history)

        logger.info("="*50)
        logger.info(f"Performance Update:")
        logger.info(f"  Trades: {total_trades}")
        logger.info(f"  Win Rate: {profitable/total_trades*100:.1f}%")
        logger.info(f"  Total Pips: {total_pips:.1f}")
        logger.info(f"  Equity: ${self.equity:.2f}")
        logger.info(f"  Drawdown: {(self.peak_equity - self.equity)/self.peak_equity*100:.1f}%")
        logger.info(f"  Gate Rate: {self.false_rejects}/{self.total_gates} blocked")
        logger.info("="*50)

    def shutdown(self):
        """Shutdown trading system."""
        # Close any open positions
        if self.position != 0:
            logger.info("Closing open position before shutdown")
            # Use current market price (would need real price in production)
            self._execute_trade(0, self.entry_price)

        # Save performance report
        report = {
            'start_time': datetime.now().isoformat(),
            'trades': self.trade_history,
            'final_equity': self.equity,
            'total_pips': sum(t['pips'] for t in self.trade_history),
            'win_rate': sum(1 for t in self.trade_history if t['pips'] > 0) / max(len(self.trade_history), 1),
            'gate_stats': {
                'total': self.total_gates,
                'blocked': self.false_rejects,
                'rate': self.false_rejects / max(self.total_gates, 1)
            }
        }

        report_file = f"live_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Performance report saved: {report_file}")


def simulate_live_trading():
    """Simulate live trading with fake market data."""
    logger.info("Starting live trading simulation")

    # Initialize trading system
    trader = LiveTradingSystem(
        model_path="models/best_model.zip",
        instrument="GBPJPY",
        demo_mode=True
    )

    # Simulate market ticks (would be real data stream in production)
    base_price = 150.000  # GBPJPY example
    for tick in range(1000):
        # Generate fake market movement
        noise = np.random.randn() * 0.01  # 1 pip std dev
        price = base_price + np.sin(tick * 0.1) * 0.05 + noise

        market_data = {
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'market_features': np.random.randn(8)  # Fake features
        }

        # Process tick
        trader.run_tick(market_data)

        # Simulate tick delay
        time.sleep(0.1)  # 100ms between ticks

        # Check for keyboard interrupt
        if tick % 100 == 0:
            logger.info(f"Processed {tick} ticks")

    # Shutdown
    trader.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live trading with improved PPO model")
    parser.add_argument("--model", type=str, default="models/best_model.zip",
                       help="Path to model file")
    parser.add_argument("--instrument", type=str, default="GBPJPY",
                       help="Trading instrument")
    parser.add_argument("--demo", action="store_true",
                       help="Run in demo mode with simulated data")
    parser.add_argument("--position-size", type=int, default=1000,
                       help="Fixed position size")

    args = parser.parse_args()

    if args.demo:
        simulate_live_trading()
    else:
        logger.error("Real trading requires OANDA API integration - not yet implemented")
        logger.info("Use --demo flag for simulation")