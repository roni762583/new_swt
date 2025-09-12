#!/usr/bin/env python3
"""
OANDA Trade Executor - Live Trading Implementation
Executes actual trades through OANDA V20 API for SWT live trading system
"""

import v20
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for OANDA trading"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"  
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRequest:
    """Trade execution request"""
    direction: TradeDirection
    units: int
    instrument: str
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "FOK"  # Fill or Kill
    client_extensions: Optional[Dict] = None


@dataclass 
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_units: Optional[int] = None
    execution_time: Optional[datetime] = None
    error_message: Optional[str] = None
    transaction_id: Optional[str] = None


@dataclass
class PositionCloseResult:
    """Position close result"""
    success: bool
    units_closed: int
    close_price: float
    realized_pnl: float
    transaction_id: Optional[str] = None
    error_message: Optional[str] = None


class OANDATradeExecutor:
    """
    OANDA V20 API trade executor for live SWT trading
    Handles actual trade execution, position management and order placement
    """
    
    def __init__(
        self,
        api_token: str,
        account_id: str,
        environment: str = "practice",
        max_slippage_pips: float = 2.0,
        execution_timeout: float = 30.0
    ):
        """
        Initialize OANDA trade executor
        
        Args:
            api_token: OANDA V20 API token
            account_id: OANDA account ID
            environment: "practice" or "live" 
            max_slippage_pips: Maximum allowed slippage in pips
            execution_timeout: Maximum time to wait for execution in seconds
        """
        self.api_token = api_token
        self.account_id = account_id
        self.environment = environment
        self.max_slippage_pips = max_slippage_pips
        self.execution_timeout = execution_timeout
        
        # OANDA V20 API setup
        environment_param = "practice" if environment == "practice" else "live"
        
        self.api = v20.Context(
            hostname="api-fxtrade.oanda.com" if environment == "live" else "api-fxpractice.oanda.com",
            token=api_token
        )
        
        # Execution statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_slippage_pips = 0.0
        self.last_execution_time = None
        
        # Order failure tracking to prevent spam
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.failure_cooldown_seconds = 30  # Start with 30 second cooldown
        self.max_consecutive_failures = 5   # Max failures before longer cooldown
        
        # Risk management
        self.max_units_per_trade = 100000  # Maximum position size
        self.min_balance_required = 10.0  # Minimum account balance (lowered for demo)
        
        logger.info(f"🔧 OANDA Trade Executor initialized")
        logger.info(f"   Account: {account_id} ({environment})")
        logger.info(f"   Max slippage: {max_slippage_pips} pips")
        logger.info(f"   Execution timeout: {execution_timeout}s")
        
        # Validate API connection
        self._validate_trading_permissions()
        
    def _validate_trading_permissions(self) -> None:
        """Validate API has trading permissions"""
        try:
            response = self.api.account.get(self.account_id)
            
            if response.status != 200:
                raise ValueError(f"Cannot access account: {response.body}")
                
            account = response.body['account']
            
            # Check if account allows trading
            if not hasattr(account, 'marginCallExtensionCount'):
                logger.warning("⚠️ Account may not support margin trading")
                
            balance = float(account.balance)
            if balance < self.min_balance_required:
                raise ValueError(f"Insufficient balance: {balance} < {self.min_balance_required}")
                
            logger.info(f"✅ Trading permissions validated - Balance: {balance} {account.currency}")
            
        except Exception as e:
            logger.error(f"❌ Trading validation failed: {e}")
            raise
            
    def execute_market_order(
        self,
        direction: TradeDirection,
        units: int,
        instrument: str,
        stop_loss_pips: Optional[float] = None,
        take_profit_pips: Optional[float] = None,
        client_tag: Optional[str] = None
    ) -> TradeResult:
        """
        Execute market order for immediate execution
        
        Args:
            direction: LONG or SHORT
            units: Number of units to trade (positive number)
            instrument: Trading instrument (e.g. "GBP_JPY")
            stop_loss_pips: Stop loss in pips from entry price
            take_profit_pips: Take profit in pips from entry price
            client_tag: Optional client reference tag
            
        Returns:
            TradeResult with execution details
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check order failure cooldown to prevent spam
            if self._is_in_failure_cooldown():
                return TradeResult(
                    success=False,
                    error_message=f"Order cooldown active: {self._get_cooldown_remaining():.1f}s remaining after {self.consecutive_failures} consecutive failures"
                )
            
            # Check if forex markets are likely open (basic check)
            if not self._is_likely_market_hours():
                self._handle_order_failure("Market likely closed (weekend or low liquidity hours)")
                return TradeResult(
                    success=False,
                    error_message="Market likely closed (weekend or low liquidity hours)"
                )
            
            # Validate inputs
            if units <= 0:
                return TradeResult(
                    success=False,
                    error_message="Units must be positive"
                )
                
            if units > self.max_units_per_trade:
                return TradeResult(
                    success=False, 
                    error_message=f"Units exceed maximum: {units} > {self.max_units_per_trade}"
                )
            
            # Convert to signed units for OANDA API
            signed_units = units if direction == TradeDirection.LONG else -units
            
            # Get current price for stop loss/take profit calculation
            current_price = self._get_current_price(instrument)
            if not current_price:
                return TradeResult(
                    success=False,
                    error_message="Cannot get current price"
                )
                
            # Build order request with proper handling to prevent order spam
            order_body = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(signed_units),
                    "timeInForce": "IOC",  # Immediate or Cancel - less aggressive than FOK
                    "positionFill": "DEFAULT"
                }
            }
            
            # Add client extensions if provided
            if client_tag:
                order_body["order"]["clientExtensions"] = {
                    "id": client_tag,
                    "tag": "SWT_LIVE_TRADING",
                    "comment": f"SWT {direction.value.upper()} order"
                }
                
            # Add stop loss if specified
            if stop_loss_pips:
                if direction == TradeDirection.LONG:
                    sl_price = current_price - (stop_loss_pips * self._get_pip_value(instrument))
                else:
                    sl_price = current_price + (stop_loss_pips * self._get_pip_value(instrument))
                    
                order_body["order"]["stopLossOnFill"] = {
                    "price": f"{sl_price:.5f}",
                    "timeInForce": "GTC"
                }
                
            # Add take profit if specified  
            if take_profit_pips:
                if direction == TradeDirection.LONG:
                    tp_price = current_price + (take_profit_pips * self._get_pip_value(instrument))
                else:
                    tp_price = current_price - (take_profit_pips * self._get_pip_value(instrument))
                    
                order_body["order"]["takeProfitOnFill"] = {
                    "price": f"{tp_price:.5f}",
                    "timeInForce": "GTC"
                }
                
            # Execute order
            logger.info(f"🔄 Executing {direction.value.upper()} order: {units} units {instrument}")
            
            # Execute order using v20 API - correct method is api.order.market()
            response = self.api.order.market(self.account_id, **order_body["order"])
            
            self.total_orders += 1
            execution_time = datetime.now(timezone.utc)
            
            if response.status == 201:
                # Success - parse response
                order_create_transaction = response.body.get("orderCreateTransaction")
                order_fill_transaction = response.body.get("orderFillTransaction")
                
                if order_fill_transaction:
                    # Order was filled
                    fill_price = float(order_fill_transaction.price)
                    fill_units = int(order_fill_transaction.units)
                    trade_id = order_fill_transaction.tradeOpened.tradeID if order_fill_transaction.tradeOpened else None
                    
                    # Calculate slippage
                    slippage = abs(fill_price - current_price)
                    slippage_pips = slippage / self._get_pip_value(instrument)
                    self.total_slippage_pips += slippage_pips
                    
                    self.successful_orders += 1
                    self.last_execution_time = execution_time
                    # Reset failure tracking on success
                    self.consecutive_failures = 0
                    self.last_failure_time = None
                    
                    logger.info(f"✅ Order filled: {fill_units} units @ {fill_price:.5f}")
                    logger.info(f"📊 Slippage: {slippage_pips:.1f} pips, Trade ID: {trade_id}")
                    
                    return TradeResult(
                        success=True,
                        order_id=order_create_transaction.id if order_create_transaction else None,
                        trade_id=trade_id,
                        fill_price=fill_price,
                        fill_units=abs(fill_units),
                        execution_time=execution_time,
                        transaction_id=order_fill_transaction.id
                    )
                    
                else:
                    # Order created but not filled - track consecutive failures
                    self.failed_orders += 1
                    self._handle_order_failure("Order created but not filled (market closed or insufficient liquidity)")
                    return TradeResult(
                        success=False,
                        order_id=order_create_transaction.id if order_create_transaction else None,
                        error_message="Order created but not filled (market closed or insufficient liquidity)"
                    )
                    
            else:
                # Order failed - track consecutive failures
                self.failed_orders += 1
                error_msg = response.body.get("errorMessage", f"HTTP {response.status}")
                
                logger.error(f"❌ Order failed: {error_msg}")
                self._handle_order_failure(error_msg)
                
                return TradeResult(
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            self.failed_orders += 1
            logger.error(f"❌ Order execution exception: {e}")
            self._handle_order_failure(str(e))
            
            return TradeResult(
                success=False,
                error_message=str(e)
            )
            
    def close_position(
        self,
        instrument: str,
        units: Optional[int] = None,
        client_tag: Optional[str] = None
    ) -> PositionCloseResult:
        """
        Close position (partial or full)
        
        Args:
            instrument: Instrument to close
            units: Units to close (None for full position)
            client_tag: Optional client reference
            
        Returns:
            PositionCloseResult with close details
        """
        try:
            # Get current position
            position_response = self.api.position.get(
                accountID=self.account_id,
                instrument=instrument
            )
            
            if position_response.status != 200:
                return PositionCloseResult(
                    success=False,
                    units_closed=0,
                    close_price=0.0,
                    realized_pnl=0.0,
                    error_message="Cannot get position info"
                )
                
            position = position_response.body["position"]
            long_units = int(position.long.units) if position.long.units else 0
            short_units = int(position.short.units) if position.short.units else 0
            net_units = long_units + short_units
            
            if net_units == 0:
                return PositionCloseResult(
                    success=False,
                    units_closed=0,
                    close_price=0.0,
                    realized_pnl=0.0,
                    error_message="No position to close"
                )
                
            # Determine units to close
            if units is None:
                # Close entire position
                close_body = {"longUnits": "ALL", "shortUnits": "ALL"}
                logger.info(f"🔄 Closing full position: {net_units} units {instrument}")
            else:
                # Close partial position
                if net_units > 0:  # Long position
                    close_body = {"longUnits": str(min(units, long_units))}
                else:  # Short position
                    close_body = {"shortUnits": str(min(units, abs(short_units)))}
                logger.info(f"🔄 Closing partial position: {units} units {instrument}")
                
            # Add client extensions
            if client_tag:
                close_body["clientExtensions"] = {
                    "id": client_tag,
                    "tag": "SWT_LIVE_TRADING",
                    "comment": "SWT position close"
                }
                
            # Execute close
            response = self.api.position.close(
                accountID=self.account_id,
                instrument=instrument,
                body=close_body
            )
            
            if response.status == 200:
                # Parse close result
                long_close = response.body.get("longOrderFillTransaction")
                short_close = response.body.get("shortOrderFillTransaction")
                
                close_transaction = long_close or short_close
                
                if close_transaction:
                    units_closed = abs(int(close_transaction.units))
                    close_price = float(close_transaction.price)
                    realized_pnl = float(close_transaction.pl)
                    
                    logger.info(f"✅ Position closed: {units_closed} units @ {close_price:.5f}")
                    logger.info(f"💰 Realized PnL: {realized_pnl:.2f}")
                    
                    return PositionCloseResult(
                        success=True,
                        units_closed=units_closed,
                        close_price=close_price,
                        realized_pnl=realized_pnl,
                        transaction_id=close_transaction.id
                    )
                else:
                    return PositionCloseResult(
                        success=False,
                        units_closed=0,
                        close_price=0.0,
                        realized_pnl=0.0,
                        error_message="Close transaction not found in response"
                    )
            else:
                error_msg = response.body.get("errorMessage", f"HTTP {response.status}")
                return PositionCloseResult(
                    success=False,
                    units_closed=0,
                    close_price=0.0,
                    realized_pnl=0.0,
                    error_message=error_msg
                )
                
        except Exception as e:
            logger.error(f"❌ Position close exception: {e}")
            return PositionCloseResult(
                success=False,
                units_closed=0,
                close_price=0.0,
                realized_pnl=0.0,
                error_message=str(e)
            )
            
    def _get_current_price(self, instrument: str) -> Optional[float]:
        """Get current mid price for instrument"""
        try:
            response = self.api.pricing.get(
                accountID=self.account_id,
                instruments=instrument
            )
            
            if response.status == 200:
                prices = response.body["prices"]
                if prices:
                    price_data = prices[0]
                    bid = float(price_data.bids[0].price)
                    ask = float(price_data.asks[0].price)
                    return (bid + ask) / 2
                    
        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
            
        return None
        
    def _get_pip_value(self, instrument: str) -> float:
        """Get pip value for instrument"""
        if "JPY" in instrument:
            return 0.01  # JPY pairs
        else:
            return 0.0001  # Most other pairs
    
    def _handle_order_failure(self, error_message: str) -> None:
        """Handle order failure and update cooldown tracking"""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        # Exponential backoff: 30s, 60s, 120s, 240s, 300s (max 5 min)
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.failure_cooldown_seconds = min(300, 30 * (2 ** (self.consecutive_failures - 1)))
            
        logger.warning(f"⚠️ Order failure #{self.consecutive_failures}: {error_message}")
        logger.warning(f"⏳ Next order allowed in {self.failure_cooldown_seconds}s")
    
    def _is_in_failure_cooldown(self) -> bool:
        """Check if we're in failure cooldown period"""
        if self.last_failure_time is None:
            return False
            
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return time_since_failure < self.failure_cooldown_seconds
    
    def _get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds"""
        if self.last_failure_time is None:
            return 0.0
            
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return max(0.0, self.failure_cooldown_seconds - time_since_failure)
    
    def _is_likely_market_hours(self) -> bool:
        """Basic check if forex markets are likely open"""
        now_utc = datetime.now(timezone.utc)
        
        # Check if weekend (forex closed Saturday 22:00 UTC to Sunday 22:00 UTC)
        weekday = now_utc.weekday()  # 0=Monday, 6=Sunday
        hour = now_utc.hour
        
        # Saturday after 22:00 UTC or Sunday before 22:00 UTC
        if weekday == 5 and hour >= 22:  # Saturday evening
            return False
        if weekday == 6 and hour < 22:   # Sunday before evening
            return False
            
        # Major holidays (simplified - in production use proper calendar)
        # This is a basic implementation
        return True
            
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        success_rate = (self.successful_orders / max(self.total_orders, 1)) * 100
        avg_slippage = self.total_slippage_pips / max(self.successful_orders, 1)
        
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate_percent": success_rate,
            "average_slippage_pips": avg_slippage,
            "total_slippage_pips": self.total_slippage_pips,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "environment": self.environment,
            "max_slippage_allowed": self.max_slippage_pips,
            "execution_timeout": self.execution_timeout
        }
        
    def validate_trade_request(self, request: TradeRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate trade request before execution
        
        Args:
            request: Trade request to validate
            
        Returns:
            (is_valid, error_message)
        """
        # Check units
        if request.units <= 0:
            return False, "Units must be positive"
            
        if request.units > self.max_units_per_trade:
            return False, f"Units exceed maximum: {request.units} > {self.max_units_per_trade}"
            
        # Check account balance
        try:
            response = self.api.account.get(self.account_id)
            if response.status != 200:
                return False, "Cannot check account balance"
                
            account = response.body["account"]
            balance = float(account.balance)
            margin_available = float(account.marginAvailable)
            
            if balance < self.min_balance_required:
                return False, f"Insufficient balance: {balance} < {self.min_balance_required}"
                
            # Rough margin check (instrument-specific margin rates would be better)
            estimated_margin = request.units * 0.02  # Assume 2% margin requirement
            if margin_available < estimated_margin:
                return False, f"Insufficient margin: {margin_available} < {estimated_margin}"
                
        except Exception as e:
            return False, f"Account validation error: {e}"
            
        return True, None


def create_trade_executor(
    api_token: str,
    account_id: str,
    environment: str = "practice"
) -> OANDATradeExecutor:
    """Factory function to create OANDA trade executor"""
    return OANDATradeExecutor(
        api_token=api_token,
        account_id=account_id,
        environment=environment
    )


if __name__ == "__main__":
    # Test trade executor
    import os
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Get credentials from environment
    api_token = os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    environment = os.getenv("OANDA_ENVIRONMENT", "practice")
    
    if not (api_token and account_id):
        print("❌ OANDA credentials not found in environment")
        print("   Set OANDA_API_KEY and OANDA_ACCOUNT_ID")
        exit(1)
        
    try:
        # Create trade executor
        executor = create_trade_executor(
            api_token=api_token,
            account_id=account_id,
            environment=environment
        )
        
        # Test validation
        request = TradeRequest(
            direction=TradeDirection.LONG,
            units=1000,
            instrument="GBP_JPY"
        )
        
        is_valid, error = executor.validate_trade_request(request)
        print(f"📊 Trade validation: {'✅' if is_valid else '❌'}")
        if error:
            print(f"   Error: {error}")
            
        # Get statistics
        stats = executor.get_execution_statistics()
        print(f"\n📈 Execution Statistics:")
        print(f"   Total orders: {stats['total_orders']}")
        print(f"   Success rate: {stats['success_rate_percent']:.1f}%")
        print(f"   Environment: {stats['environment']}")
        
        # Note: Actual trade execution would require user confirmation
        print(f"\n⚠️ Live trading test complete - No actual trades executed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        exit(1)