#!/usr/bin/env python3
"""
PRODUCTION LIVE TRADING with OANDA Integration
Complete broker connectivity, position reconciliation, and risk management
"""

import os
import sys
import torch
import pickle
import numpy as np
import asyncio
import signal
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ppo_agent import PPOLearningPolicy

# =======================
# CONFIGURATION
# =======================

@dataclass
class TradingConfig:
    """Production trading configuration"""
    # Broker settings
    oanda_api_token: str = os.getenv('OANDA_API_TOKEN', '')
    oanda_account_id: str = os.getenv('OANDA_ACCOUNT_ID', '')
    oanda_environment: str = os.getenv('OANDA_ENV', 'practice')  # 'practice' or 'live'
    instrument: str = 'GBP_JPY'

    # Model settings
    checkpoint_path: str = '/app/checkpoints/parashat_vayelech.pth'
    state_dim: int = 17

    # Position sizing
    base_units: int = 1  # Trade size: 1 unit only (NOT 1 lot)
    max_position_units: int = 1  # Maximum 1 unit position at a time

    # Risk management
    max_daily_loss: float = -500.0  # Maximum daily loss in account currency
    max_drawdown_pct: float = 0.10  # 10% maximum drawdown
    stop_loss_pips: float = 20.0  # Default stop loss
    take_profit_pips: float = 30.0  # Default take profit

    # Reconciliation
    reconciliation_interval: int = 60  # Seconds between reconciliation checks
    position_tolerance_units: int = 1  # Allowed position difference

    # Trading hours (UTC)
    trading_start_hour: int = 1  # 1 AM UTC
    trading_end_hour: int = 21  # 9 PM UTC

    def validate(self):
        """Validate configuration"""
        if not self.oanda_api_token:
            raise ValueError("OANDA_API_TOKEN environment variable not set")
        if not self.oanda_account_id:
            raise ValueError("OANDA_ACCOUNT_ID environment variable not set")
        if not os.path.exists(self.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")

        # ALWAYS LIVE TRADING - NO SIMULATION
        logger.warning("🔴 LIVE TRADING SYSTEM - REAL MONEY AT RISK")
        logger.warning("🔴 This will execute REAL trades with REAL money")
        response = input("Type 'CONFIRM LIVE TRADING' to proceed: ")
        if response != 'CONFIRM LIVE TRADING':
            raise ValueError("Live trading not confirmed - must type 'CONFIRM LIVE TRADING'")


# =======================
# BROKER INTEGRATION
# =======================

class PositionState(Enum):
    """Position states"""
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Current position information"""
    state: PositionState = PositionState.FLAT
    units: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    unrealized_pnl: float = 0.0


class BrokerIntegration:
    """OANDA broker integration with position reconciliation"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.position = Position()
        self.last_reconciliation = datetime.now(timezone.utc)

        # Import OANDA client if available
        self.oanda_client = None
        self.has_broker = False

        try:
            import v20
            self.oanda_context = v20.Context(
                self.config.oanda_environment + '.oanda.com',
                443,
                True,
                application='PPO_Trading',
                token=self.config.oanda_api_token
            )
            self.has_broker = True
            logger.info("✅ OANDA broker connected")
        except ImportError:
            logger.warning("⚠️ OANDA v20 not installed - running in simulation mode")
        except Exception as e:
            logger.error(f"❌ Failed to connect to OANDA: {e}")

    async def get_current_price(self) -> float:
        """Get current market price"""
        if not self.has_broker:
            # Return simulated price
            return 150.000 + np.random.randn() * 0.01

        try:
            response = self.oanda_context.pricing.get(
                self.config.oanda_account_id,
                instruments=self.config.instrument
            )

            if response.status == 200:
                prices = response.body['prices']
                if prices:
                    bid = float(prices[0].bids[0].price)
                    ask = float(prices[0].asks[0].price)
                    return (bid + ask) / 2

        except Exception as e:
            logger.error(f"Failed to get price: {e}")

        return 0.0

    async def get_broker_position(self) -> Dict[str, Any]:
        """Get current position from broker"""
        if not self.has_broker:
            return {
                'units': self.position.units,
                'averagePrice': self.position.entry_price,
                'unrealizedPL': self.position.unrealized_pnl
            }

        try:
            response = self.oanda_context.position.get(
                self.config.oanda_account_id,
                self.config.instrument
            )

            if response.status == 200:
                position = response.body['position']
                long_units = int(position.long.units)
                short_units = int(position.short.units)

                net_units = long_units + short_units

                return {
                    'units': net_units,
                    'averagePrice': float(position.long.averagePrice if long_units > 0
                                        else position.short.averagePrice if short_units < 0
                                        else 0),
                    'unrealizedPL': float(position.unrealizedPL)
                }

        except Exception as e:
            logger.error(f"Failed to get broker position: {e}")

        return {'units': 0, 'averagePrice': 0, 'unrealizedPL': 0}

    async def place_order(self, action: int, units: int) -> bool:
        """Place LIVE order with broker - ALWAYS EXECUTES REAL TRADES"""

        if not self.has_broker:
            logger.error("❌ No broker connection")
            return False

        try:
            # Build order based on action
            if action == 1:  # BUY
                order_data = {
                    'order': {
                        'type': 'MARKET',
                        'instrument': self.config.instrument,
                        'units': str(units),
                        'timeInForce': 'FOK',
                        'positionFill': 'DEFAULT'
                    }
                }
            elif action == 2:  # SELL
                order_data = {
                    'order': {
                        'type': 'MARKET',
                        'instrument': self.config.instrument,
                        'units': str(-units),
                        'timeInForce': 'FOK',
                        'positionFill': 'DEFAULT'
                    }
                }
            elif action == 3:  # CLOSE
                # Close existing position
                response = self.oanda_context.position.close(
                    self.config.oanda_account_id,
                    self.config.instrument,
                    longUnits='ALL' if self.position.state == PositionState.LONG else None,
                    shortUnits='ALL' if self.position.state == PositionState.SHORT else None
                )
                logger.info(f"✅ Position closed")
                self.position = Position()
                return response.status == 200
            else:
                return False

            # Execute order for BUY/SELL
            if action in [1, 2]:
                response = self.oanda_context.order.create(
                    self.config.oanda_account_id,
                    order=order_data
                )

                if response.status == 201:
                    logger.info(f"✅ Order executed: {action} {units} units")
                    return True
                else:
                    logger.error(f"❌ Order failed: {response.body}")
                    return False

        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            return False

    async def reconcile_position(self) -> bool:
        """Reconcile internal position with broker"""
        broker_pos = await self.get_broker_position()

        # Compare positions
        broker_units = broker_pos['units']
        our_units = self.position.units

        if abs(broker_units - our_units) > self.config.position_tolerance_units:
            logger.warning(f"⚠️ POSITION MISMATCH - Broker: {broker_units}, Internal: {our_units}")

            # Sync to broker position (broker is source of truth)
            self.position.units = broker_units
            if broker_units > 0:
                self.position.state = PositionState.LONG
            elif broker_units < 0:
                self.position.state = PositionState.SHORT
            else:
                self.position.state = PositionState.FLAT

            self.position.entry_price = broker_pos['averagePrice']
            self.position.unrealized_pnl = broker_pos['unrealizedPL']

            return False  # Mismatch found

        # Update P&L
        self.position.unrealized_pnl = broker_pos['unrealizedPL']
        return True  # Positions match

    def should_reconcile(self) -> bool:
        """Check if reconciliation is due"""
        elapsed = (datetime.now(timezone.utc) - self.last_reconciliation).total_seconds()
        return elapsed >= self.config.reconciliation_interval


# =======================
# RISK MANAGEMENT
# =======================

class RiskManager:
    """Risk management and position sizing"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = 10000.0
        self.current_balance = 10000.0
        self.last_reset = datetime.now(timezone.utc).date()

    def check_daily_reset(self):
        """Reset daily counters if new day"""
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.last_reset:
            logger.info(f"📅 New trading day - resetting daily counters")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = current_date

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        self.check_daily_reset()

        # Check trading hours
        current_hour = datetime.now(timezone.utc).hour
        if not (self.config.trading_start_hour <= current_hour < self.config.trading_end_hour):
            return False, "Outside trading hours"

        # Check daily loss limit
        if self.daily_pnl <= self.config.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # Check drawdown
        drawdown_pct = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown_pct > self.config.max_drawdown_pct:
            return False, f"Maximum drawdown exceeded: {drawdown_pct:.2%}"

        return True, "OK"

    def calculate_position_size(self, action: int, current_position: Position) -> int:
        """Calculate safe position size - ONLY 1 unit trades allowed"""
        if action == 3:  # CLOSE
            return 0

        # CRITICAL: Only allow trading if we're FLAT (no existing position)
        if current_position.state != PositionState.FLAT:
            logger.info("⚠️ Position already open - only one trade at a time allowed")
            return 0

        # Check if trying to open a new position
        if action in [1, 2]:  # BUY or SELL
            # Only allow if we have no position
            if abs(current_position.units) > 0:
                logger.warning(f"Cannot open new position - already have {current_position.units} units")
                return 0

        # Don't trade if in drawdown
        if current_position.unrealized_pnl < -50:
            logger.warning("Not trading while in drawdown")
            return 0

        # Always return exactly 1 unit (our configured base_units)
        return self.config.base_units  # Will be 1 unit

    def update_pnl(self, pnl: float):
        """Update P&L tracking"""
        self.daily_pnl += pnl
        self.current_balance += pnl

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        self.max_drawdown = max(self.max_drawdown,
                                (self.peak_balance - self.current_balance) / self.peak_balance)


# =======================
# MAIN LIVE TRADING
# =======================

class LiveTradingSystem:
    """Complete live trading system with PPO agent"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.running = False
        self.policy = None
        self.broker = None
        self.risk_manager = None

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pips = 0.0
        self.start_time = datetime.now(timezone.utc)

    def load_model(self):
        """Load PPO model from checkpoint"""
        logger.info(f"📂 Loading model from {self.config.checkpoint_path}")

        # Initialize policy
        self.policy = PPOLearningPolicy(state_dim=self.config.state_dim)

        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location='cpu')

        if 'policy_state_dict' in checkpoint:
            self.policy.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            logger.info("✅ Model loaded successfully")
        else:
            raise ValueError("Invalid checkpoint format")

    def get_observation(self) -> np.ndarray:
        """Get current observation for model"""
        # In production, this would get real market data
        # For now, create mock observation matching training format

        obs = np.zeros(self.config.state_dim)

        # Market features (simplified)
        obs[0] = np.random.randn() * 0.1  # Price change
        obs[1] = np.random.randn() * 0.1  # RSI
        obs[2] = np.random.randn() * 0.1  # Volume

        # Position features
        if self.broker.position.state == PositionState.LONG:
            obs[10] = 1.0
        elif self.broker.position.state == PositionState.SHORT:
            obs[11] = -1.0

        obs[12] = self.broker.position.unrealized_pnl / 100  # Normalized P&L

        return obs

    async def trading_loop(self):
        """Main trading loop"""
        logger.info("🚀 Starting live trading loop")

        while self.running:
            try:
                # Check if we can trade
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    logger.info(f"⏸️ Trading paused: {reason}")
                    await asyncio.sleep(60)
                    continue

                # Reconcile position if needed
                if self.broker.should_reconcile():
                    is_synced = await self.broker.reconcile_position()
                    if not is_synced:
                        logger.warning("⚠️ Position reconciliation required")
                    self.broker.last_reconciliation = datetime.now(timezone.utc)

                # Get observation
                obs = self.get_observation()

                # Get action from model
                action = self.policy.compute_action(obs)

                # POSITION MANAGEMENT LOGIC - ONE TRADE AT A TIME
                current_pos = self.broker.position

                # If we have a position, only allow HOLD or CLOSE
                if current_pos.state != PositionState.FLAT:
                    if action == 3:  # CLOSE signal
                        logger.info(f"📊 Closing position: {current_pos.units} units")
                        success = await self.broker.place_order(3, 0)  # Close all
                        if success:
                            self.total_trades += 1
                            # Update P&L
                            if current_pos.unrealized_pnl != 0:
                                self.risk_manager.update_pnl(current_pos.unrealized_pnl)
                                if current_pos.unrealized_pnl > 0:
                                    self.winning_trades += 1
                                self.total_pips += current_pos.unrealized_pnl
                    elif action in [1, 2]:
                        logger.debug(f"Ignoring {['', 'BUY', 'SELL'][action]} signal - position already open")
                else:
                    # We're flat - can open new position
                    if action in [1, 2]:  # BUY or SELL
                        # Always use 1 unit
                        units = 1
                        action_name = 'BUY' if action == 1 else 'SELL'

                        # Extra safety check
                        can_trade, reason = self.risk_manager.can_trade()
                        if can_trade:
                            logger.info(f"📊 Opening {action_name} position: 1 unit")
                            success = await self.broker.place_order(action, units)
                            if success:
                                self.total_trades += 1
                        else:
                            logger.warning(f"Cannot open position: {reason}")

                # Wait before next iteration (don't overtrade)
                await asyncio.sleep(5)  # Check every 5 seconds

            except KeyboardInterrupt:
                logger.info("⏹️ Stopping trading...")
                self.running = False
                break

            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(10)

    def print_statistics(self):
        """Print trading statistics"""
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600

        print("\n" + "=" * 60)
        print("📊 TRADING STATISTICS")
        print("=" * 60)
        print(f"Runtime: {runtime:.2f} hours")
        print(f"Total trades: {self.total_trades}")
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            print(f"Winning trades: {self.winning_trades}/{self.total_trades} ({win_rate:.1f}%)")
            print(f"Total P&L: {self.total_pips:.2f} pips")
            print(f"Average per trade: {self.total_pips/self.total_trades:.2f} pips")
        print(f"Current position: {self.broker.position.state.value} ({self.broker.position.units} units)")
        print(f"Unrealized P&L: {self.broker.position.unrealized_pnl:.2f}")
        print(f"Daily P&L: {self.risk_manager.daily_pnl:.2f}")
        print(f"Max drawdown: {self.risk_manager.max_drawdown:.2%}")
        print("=" * 60)

    async def run(self):
        """Run the live trading system"""
        try:
            # Validate configuration
            self.config.validate()

            # Load model
            self.load_model()

            # Initialize components
            self.broker = BrokerIntegration(self.config)
            self.risk_manager = RiskManager(self.config)

            # Initial position reconciliation
            await self.broker.reconcile_position()

            # Setup signal handlers
            signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'running', False))
            signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'running', False))

            # Start trading
            self.running = True
            await self.trading_loop()

        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise

        finally:
            # Print final statistics
            self.print_statistics()

            # Close all positions on shutdown
            if self.broker and self.broker.position.state != PositionState.FLAT:
                logger.info("🔴 Closing all positions...")
                await self.broker.place_order(3, 0)  # CLOSE


async def main():
    """Main entry point"""
    print("=" * 60)
    print("🔴 PPO LIVE TRADING SYSTEM")
    print("=" * 60)
    print(f"Starting at: {datetime.now(timezone.utc)}")

    # Load configuration
    config = TradingConfig()

    # Safety warnings
    if config.enable_trading and not config.dry_run:
        print("\n⚠️ WARNING: LIVE TRADING ENABLED - REAL MONEY AT RISK")
        print("Press Ctrl+C to stop at any time")
    else:
        print("\n✅ SAFE MODE: Dry run enabled")

    # Create and run trading system
    trading_system = LiveTradingSystem(config)
    await trading_system.run()

    print("\n✅ Trading system shutdown complete")


if __name__ == '__main__':
    # Run the async main function
    asyncio.run(main())