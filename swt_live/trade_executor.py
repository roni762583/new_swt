"""
SWT Trade Executor
Production-grade order execution system for live trading

This module provides robust order execution with:
- Async order placement with retry logic
- Slippage tracking and validation
- Position size management
- Execution confirmation and error handling
- Risk management controls
- Performance monitoring

CRITICAL: Integrates with shared position management ensuring
consistency with training system expectations.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

import aiohttp
from swt_core.types import TradingAction, PositionType, TradeResult
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import TradeExecutionError, ValidationError, RiskManagementError
from swt_features.market_features import MarketDataPoint

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


class TimeInForce(Enum):
    """Time in force enumeration"""
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTC = "GTC"  # Good Till Cancelled


@dataclass
class OrderRequest:
    """Order request specification"""
    action: TradingAction
    instrument: str
    units: int
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.IOC
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_tag: Optional[str] = None
    
    def __post_init__(self):
        """Validate order request"""
        if self.action == TradingAction.HOLD:
            raise ValidationError("Cannot execute HOLD action")
        
        if self.units == 0:
            raise ValidationError("Order units cannot be zero")
        
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValidationError("Limit orders require price")


@dataclass
class ExecutionReport:
    """Order execution report"""
    order_id: str
    transaction_id: str
    status: OrderStatus
    filled_units: int
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    slippage_pips: Optional[float] = None
    commission: Optional[float] = None
    financing: Optional[float] = None
    pl_realized: Optional[float] = None
    error_message: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


class SWTTradeExecutor:
    """
    Production-grade trade execution system
    
    Features:
    - Async order placement with configurable timeouts
    - Automatic retry logic with exponential backoff
    - Slippage monitoring and alerting
    - Position size validation and risk controls
    - Comprehensive execution reporting
    - Performance tracking and diagnostics
    """
    
    def __init__(self, config: SWTConfig):
        """
        Initialize trade executor
        
        Args:
            config: SWT configuration containing OANDA credentials and trading params
        """
        self.config = config
        
        # OANDA API configuration
        self.account_id = config.live_trading_config.oanda_account_id
        self.api_token = config.live_trading_config.oanda_api_token
        self.api_environment = config.live_trading_config.oanda_environment
        self.instrument = config.live_trading_config.instrument
        
        # Trading configuration
        self.max_position_size = config.trading_config.max_position_size
        self.max_daily_trades = config.trading_config.max_daily_trades
        self.max_slippage_pips = config.trading_config.max_slippage_pips
        
        # Execution settings
        self.order_timeout = 10.0  # 10 second timeout
        self.max_retries = 3
        self.retry_delay = 0.5  # Start with 500ms
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.orders_sent = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.total_slippage_pips = 0.0
        self.daily_trades = 0
        self.last_trade_reset = datetime.now(timezone.utc).date()
        
        # Risk management
        self.current_position_size = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = config.trading_config.max_daily_loss
        
        logger.info(f"⚡ SWTTradeExecutor initialized for {self.instrument}")
    
    async def start(self) -> None:
        """Initialize the trade executor"""
        if self.session:
            logger.warning("⚠️ Trade executor already started")
            return
        
        # Create HTTP session with appropriate timeouts
        timeout = aiohttp.ClientTimeout(
            total=self.order_timeout,
            connect=5.0,
            sock_read=self.order_timeout
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=10)
        )
        
        logger.info("✅ Trade executor started")
    
    async def stop(self) -> None:
        """Shutdown the trade executor gracefully"""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("🛑 Trade executor stopped")
    
    async def execute_order(self, order: OrderRequest, 
                           expected_price: Optional[float] = None) -> ExecutionReport:
        """
        Execute trading order with full error handling
        
        Args:
            order: Order specification
            expected_price: Expected execution price for slippage calculation
            
        Returns:
            Execution report with results
            
        Raises:
            TradeExecutionError: If execution fails after retries
            RiskManagementError: If risk limits are violated
        """
        if not self.session:
            raise TradeExecutionError("Trade executor not started")
        
        # Pre-execution risk checks
        self._validate_risk_limits(order)
        
        logger.info(f"📤 Executing order: {order.action.value} {order.units} {order.instrument}")
        
        # Attempt execution with retries
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                execution_report = await self._execute_single_order(order, expected_price)
                
                # Update statistics
                self._update_execution_stats(execution_report)
                
                # Post-execution validation
                self._validate_execution_result(execution_report, order, expected_price)
                
                logger.info(f"✅ Order executed successfully: {execution_report.status.value}")
                return execution_report
                
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ Execution attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ Order execution failed after {self.max_retries + 1} attempts")
        
        # If we get here, all retries failed
        self.orders_rejected += 1
        raise TradeExecutionError(
            f"Failed to execute order after {self.max_retries + 1} attempts",
            original_error=last_exception
        )
    
    async def _execute_single_order(self, order: OrderRequest, 
                                   expected_price: Optional[float]) -> ExecutionReport:
        """Execute single order attempt"""
        start_time = time.time()
        
        # Build order payload
        order_payload = self._build_order_payload(order)
        
        # Execute via OANDA API
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        base_url = self._get_api_base_url()
        order_url = f"{base_url}/v3/accounts/{self.account_id}/orders"
        
        async with self.session.post(order_url, json=order_payload, headers=headers) as response:
            response_data = await response.json()
            
            if response.status not in [200, 201]:
                error_msg = self._extract_error_message(response_data)
                raise TradeExecutionError(f"OANDA API error {response.status}: {error_msg}")
            
            # Parse execution result
            execution_report = self._parse_execution_response(
                response_data, order, expected_price, start_time
            )
            
            return execution_report
    
    def _build_order_payload(self, order: OrderRequest) -> Dict[str, Any]:
        """Build OANDA order payload"""
        # Convert SWT action to OANDA units
        if order.action == TradingAction.BUY:
            units = abs(order.units)
        elif order.action == TradingAction.SELL:
            units = -abs(order.units)
        elif order.action == TradingAction.CLOSE:
            # For close, we need to determine direction based on current position
            # This should be handled by the caller, but we'll handle it safely
            units = -self.current_position_size if self.current_position_size != 0 else 0
        else:
            raise ValidationError(f"Unsupported action for execution: {order.action}")
        
        payload = {
            "order": {
                "type": order.order_type.value,
                "instrument": order.instrument,
                "units": str(units),
                "timeInForce": order.time_in_force.value
            }
        }
        
        # Add optional parameters
        if order.price is not None:
            payload["order"]["price"] = str(order.price)
        
        if order.client_tag:
            payload["order"]["clientTag"] = order.client_tag
        
        # Add stop loss if specified
        if order.stop_loss is not None:
            payload["order"]["stopLossOnFill"] = {
                "price": str(order.stop_loss),
                "timeInForce": "GTC"
            }
        
        # Add take profit if specified  
        if order.take_profit is not None:
            payload["order"]["takeProfitOnFill"] = {
                "price": str(order.take_profit),
                "timeInForce": "GTC"
            }
        
        return payload
    
    def _parse_execution_response(self, response_data: Dict[str, Any], 
                                 order: OrderRequest, 
                                 expected_price: Optional[float],
                                 start_time: float) -> ExecutionReport:
        """Parse OANDA execution response"""
        execution_time = datetime.now(timezone.utc)
        execution_latency = time.time() - start_time
        
        # Extract order fill transaction
        fill_transaction = None
        order_transaction = None
        
        for transaction in response_data.get("orderFillTransaction", []):
            if transaction.get("type") == "ORDER_FILL":
                fill_transaction = transaction
                break
        
        for transaction in response_data.get("orderCreateTransaction", []):
            if transaction.get("type") == "MARKET_ORDER":
                order_transaction = transaction
                break
        
        if fill_transaction:
            # Order was filled
            execution_price = float(fill_transaction.get("price", 0))
            filled_units = int(fill_transaction.get("units", 0))
            
            # Calculate slippage
            slippage_pips = None
            if expected_price is not None:
                slippage_pips = abs(execution_price - expected_price) * 100  # Convert to pips
            
            execution_report = ExecutionReport(
                order_id=fill_transaction.get("orderID", ""),
                transaction_id=fill_transaction.get("id", ""),
                status=OrderStatus.FILLED,
                filled_units=filled_units,
                execution_price=execution_price,
                execution_time=execution_time,
                slippage_pips=slippage_pips,
                commission=float(fill_transaction.get("commission", 0)),
                financing=float(fill_transaction.get("financing", 0)),
                pl_realized=float(fill_transaction.get("pl", 0)),
                raw_response=response_data
            )
            
        elif order_transaction:
            # Order was created but not filled (limit/stop orders)
            execution_report = ExecutionReport(
                order_id=order_transaction.get("id", ""),
                transaction_id=order_transaction.get("id", ""),
                status=OrderStatus.PENDING,
                filled_units=0,
                raw_response=response_data
            )
            
        else:
            # Order was rejected or failed
            error_msg = self._extract_error_message(response_data)
            execution_report = ExecutionReport(
                order_id="",
                transaction_id="",
                status=OrderStatus.REJECTED,
                filled_units=0,
                error_message=error_msg,
                raw_response=response_data
            )
        
        logger.info(f"📊 Execution latency: {execution_latency*1000:.1f}ms")
        
        return execution_report
    
    def _validate_risk_limits(self, order: OrderRequest) -> None:
        """Validate order against risk management limits"""
        # Reset daily counters if needed
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_trade_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_reset = current_date
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            raise RiskManagementError(f"Daily trade limit reached: {self.max_daily_trades}")
        
        # Check daily loss limit
        if self.daily_pnl <= -abs(self.max_daily_loss):
            raise RiskManagementError(f"Daily loss limit reached: ${abs(self.max_daily_loss)}")
        
        # Check position size limits
        if order.action in [TradingAction.BUY, TradingAction.SELL]:
            new_position_size = abs(order.units)
            if new_position_size > self.max_position_size:
                raise RiskManagementError(f"Position size exceeds limit: {new_position_size} > {self.max_position_size}")
            
            # Check if this would create excessive total exposure
            if order.action == TradingAction.BUY:
                total_exposure = max(0, self.current_position_size) + new_position_size
            else:
                total_exposure = max(0, -self.current_position_size) + new_position_size
            
            if total_exposure > self.max_position_size:
                raise RiskManagementError(f"Total exposure would exceed limit: {total_exposure}")
    
    def _validate_execution_result(self, execution: ExecutionReport, 
                                 order: OrderRequest,
                                 expected_price: Optional[float]) -> None:
        """Validate execution result and check for anomalies"""
        if execution.status == OrderStatus.FILLED:
            # Check slippage limits
            if execution.slippage_pips is not None:
                if execution.slippage_pips > self.max_slippage_pips:
                    logger.warning(f"⚠️ High slippage: {execution.slippage_pips:.1f} pips > {self.max_slippage_pips}")
                    # Note: We don't reject the trade, just warn as it's already executed
            
            # Validate execution price is reasonable
            if execution.execution_price is not None:
                if execution.execution_price <= 0:
                    raise ValidationError(f"Invalid execution price: {execution.execution_price}")
            
            # Update position tracking
            if order.action == TradingAction.BUY:
                self.current_position_size += execution.filled_units
            elif order.action == TradingAction.SELL:
                self.current_position_size -= execution.filled_units
            elif order.action == TradingAction.CLOSE:
                self.current_position_size = 0  # Assuming full close
            
            # Update daily P&L if available
            if execution.pl_realized is not None:
                self.daily_pnl += execution.pl_realized
    
    def _update_execution_stats(self, execution: ExecutionReport) -> None:
        """Update execution statistics"""
        self.orders_sent += 1
        
        if execution.status == OrderStatus.FILLED:
            self.orders_filled += 1
            self.daily_trades += 1
            
            if execution.slippage_pips is not None:
                self.total_slippage_pips += execution.slippage_pips
                
        elif execution.status == OrderStatus.REJECTED:
            self.orders_rejected += 1
    
    def _extract_error_message(self, response_data: Dict[str, Any]) -> str:
        """Extract error message from OANDA response"""
        if "errorMessage" in response_data:
            return response_data["errorMessage"]
        
        if "orderRejectTransaction" in response_data:
            reject_transaction = response_data["orderRejectTransaction"]
            return reject_transaction.get("rejectReason", "Unknown rejection reason")
        
        return str(response_data)
    
    def _get_api_base_url(self) -> str:
        """Get OANDA API base URL based on environment"""
        if self.api_environment == "live":
            return "https://api-fxtrade.oanda.com"
        else:
            return "https://api-fxpractice.oanda.com"
    
    async def get_position_status(self) -> Dict[str, Any]:
        """Get current position status from OANDA"""
        if not self.session:
            raise TradeExecutionError("Trade executor not started")
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        base_url = self._get_api_base_url()
        position_url = f"{base_url}/v3/accounts/{self.account_id}/positions/{self.instrument}"
        
        async with self.session.get(position_url, headers=headers) as response:
            if response.status != 200:
                response_data = await response.json()
                error_msg = self._extract_error_message(response_data)
                raise TradeExecutionError(f"Failed to get position: {error_msg}")
            
            position_data = await response.json()
            return position_data.get("position", {})
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        fill_rate = (self.orders_filled / max(1, self.orders_sent)) * 100
        avg_slippage = self.total_slippage_pips / max(1, self.orders_filled)
        
        return {
            "orders_sent": self.orders_sent,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "fill_rate_pct": fill_rate,
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "current_position_size": self.current_position_size,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "avg_slippage_pips": avg_slippage,
            "max_slippage_pips": self.max_slippage_pips,
            "last_trade_reset": self.last_trade_reset.isoformat()
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for troubleshooting"""
        stats = self.get_execution_stats()
        
        return {
            **stats,
            "configuration": {
                "api_environment": self.api_environment,
                "instrument": self.instrument,
                "max_position_size": self.max_position_size,
                "order_timeout": self.order_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay
            },
            "session_status": {
                "session_active": self.session is not None,
                "session_closed": self.session.closed if self.session else True
            }
        }


# Helper functions for creating common orders
def create_market_order(action: TradingAction, instrument: str, units: int, 
                       client_tag: Optional[str] = None) -> OrderRequest:
    """
    Create market order request
    
    Args:
        action: Trading action (BUY, SELL, CLOSE)
        instrument: Trading instrument
        units: Position size
        client_tag: Optional client identifier
        
    Returns:
        OrderRequest for market execution
    """
    return OrderRequest(
        action=action,
        instrument=instrument,
        units=units,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.IOC,
        client_tag=client_tag
    )


def create_limit_order(action: TradingAction, instrument: str, units: int, 
                      price: float, client_tag: Optional[str] = None) -> OrderRequest:
    """
    Create limit order request
    
    Args:
        action: Trading action (BUY, SELL)
        instrument: Trading instrument
        units: Position size
        price: Limit price
        client_tag: Optional client identifier
        
    Returns:
        OrderRequest for limit execution
    """
    if action == TradingAction.CLOSE:
        raise ValidationError("CLOSE action not supported for limit orders")
    
    return OrderRequest(
        action=action,
        instrument=instrument,
        units=units,
        order_type=OrderType.LIMIT,
        price=price,
        time_in_force=TimeInForce.GTC,
        client_tag=client_tag
    )