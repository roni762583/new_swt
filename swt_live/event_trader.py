"""
SWT Event Trader
Main trading orchestration engine for live trading

This module provides the central event-driven trading system that:
- Coordinates market data ingestion with inference decisions
- Manages the complete trading lifecycle
- Integrates all shared components (features, inference, execution)
- Provides real-time performance monitoring
- Handles all error conditions gracefully

CRITICAL: This is the main production trading engine that ties together
all system components into a cohesive live trading system.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

from swt_core.types import TradingAction, TradingDecision, PositionState, PositionType
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import TradingSystemError, InferenceError, TradeExecutionError
from swt_features.feature_processor import FeatureProcessor, ProcessedObservation
from swt_features.market_features import MarketDataPoint
from swt_inference.inference_engine import SWTInferenceEngine
from swt_inference.agent_factory import AgentFactory

from .data_feed import OANDADataFeed, StreamingBar, streaming_bar_to_market_data
from .trade_executor import SWTTradeExecutor, OrderRequest, ExecutionReport, create_market_order
from .position_reconciler import SWTPositionReconciler
from .monitoring import SWTLiveMonitor

logger = logging.getLogger(__name__)


class TradingSystemState(Enum):
    """Trading system state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TradingSession:
    """Trading session information"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    decisions_made: int = 0
    errors_encountered: int = 0


class SWTEventTrader:
    """
    Main event-driven trading orchestration engine
    
    This class coordinates all system components:
    - Market data streaming (OANDADataFeed)
    - Feature processing (FeatureProcessor) 
    - Trading decisions (SWTInferenceEngine)
    - Order execution (SWTTradeExecutor)
    - Position management (SWTPositionReconciler)
    - Performance monitoring (SWTLiveMonitor)
    
    Architecture:
    Market Data -> Feature Processing -> Inference -> Decision -> Execution -> Monitoring
    """
    
    def __init__(self, config: SWTConfig, checkpoint_path: str):
        """
        Initialize event trader
        
        Args:
            config: SWT system configuration
            checkpoint_path: Path to trained model checkpoint
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        # Trading system state
        self.state = TradingSystemState.STOPPED
        self.current_session: Optional[TradingSession] = None
        
        # Core components (initialized in start())
        self.feature_processor: Optional[FeatureProcessor] = None
        self.inference_engine: Optional[SWTInferenceEngine] = None
        self.data_feed: Optional[OANDADataFeed] = None
        self.trade_executor: Optional[SWTTradeExecutor] = None
        self.position_reconciler: Optional[SWTPositionReconciler] = None
        self.live_monitor: Optional[SWTLiveMonitor] = None
        
        # Trading state
        self.current_position = PositionState()
        self.last_decision_time: Optional[datetime] = None
        self.pending_orders: List[str] = []
        
        # Decision frequency control (1 decision per minute max)
        self.min_decision_interval = timedelta(minutes=1)
        self.last_bar_time: Optional[datetime] = None
        
        # Performance tracking
        self.total_bars_processed = 0
        self.total_decisions_made = 0
        self.total_trades_executed = 0
        self.system_start_time: Optional[datetime] = None
        
        # Error handling
        self.error_count = 0
        self.max_errors_per_hour = 100
        self.error_timestamps: List[datetime] = []
        
        # Event handlers
        self.decision_callbacks: List[Callable[[TradingDecision], None]] = []
        self.trade_callbacks: List[Callable[[ExecutionReport], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
        
        logger.info("🎯 SWTEventTrader initialized")
    
    async def start(self) -> None:
        """Start the complete trading system"""
        if self.state != TradingSystemState.STOPPED:
            logger.warning(f"⚠️ Trading system already in state: {self.state.value}")
            return
        
        logger.info("🚀 Starting SWT Event Trading System...")
        self.state = TradingSystemState.STARTING
        
        try:
            # Initialize core components
            await self._initialize_components()
            
            # Start all subsystems
            await self._start_subsystems()
            
            # Begin trading session
            self._start_new_session()
            
            self.state = TradingSystemState.RUNNING
            self.system_start_time = datetime.now(timezone.utc)
            
            logger.info("✅ SWT Event Trading System started successfully")
            logger.info(f"📊 Agent: {self.config.agent_system.value}")
            logger.info(f"🎯 Instrument: {self.config.live_trading_config.instrument}")
            logger.info(f"📈 Min confidence: {self.config.trading_config.min_confidence:.1%}")
            
        except Exception as e:
            self.state = TradingSystemState.ERROR
            logger.error(f"❌ Failed to start trading system: {e}")
            await self.stop()
            raise TradingSystemError(f"System startup failed: {str(e)}", original_error=e)
    
    async def stop(self) -> None:
        """Stop the trading system gracefully"""
        if self.state == TradingSystemState.STOPPED:
            logger.info("ℹ️ Trading system already stopped")
            return
        
        logger.info("🛑 Stopping SWT Event Trading System...")
        self.state = TradingSystemState.STOPPING
        
        try:
            # Stop data feed first (no new data)
            if self.data_feed:
                await self.data_feed.stop()
            
            # Close any pending orders
            await self._cancel_pending_orders()
            
            # Stop other subsystems
            if self.trade_executor:
                await self.trade_executor.stop()
            
            if self.live_monitor:
                await self.live_monitor.stop()
            
            # End current session
            if self.current_session:
                self.current_session.end_time = datetime.now(timezone.utc)
                await self._save_session_results()
            
            self.state = TradingSystemState.STOPPED
            logger.info("✅ SWT Event Trading System stopped")
            
        except Exception as e:
            logger.error(f"❌ Error during system shutdown: {e}")
            self.state = TradingSystemState.ERROR
    
    async def pause(self) -> None:
        """Pause trading (stop making new decisions but keep monitoring)"""
        if self.state != TradingSystemState.RUNNING:
            logger.warning(f"⚠️ Cannot pause system in state: {self.state.value}")
            return
        
        logger.info("⏸️ Pausing trading system...")
        self.state = TradingSystemState.PAUSED
        logger.info("✅ Trading system paused (monitoring continues)")
    
    async def resume(self) -> None:
        """Resume trading from paused state"""
        if self.state != TradingSystemState.PAUSED:
            logger.warning(f"⚠️ Cannot resume system in state: {self.state.value}")
            return
        
        logger.info("▶️ Resuming trading system...")
        self.state = TradingSystemState.RUNNING
        logger.info("✅ Trading system resumed")
    
    async def _initialize_components(self) -> None:
        """Initialize all trading system components"""
        logger.info("🔧 Initializing system components...")
        
        # 1. Feature Processor (shared component)
        self.feature_processor = FeatureProcessor(self.config)
        logger.info("✅ Feature processor initialized")
        
        # 2. Inference Engine (with agent factory)
        agent = AgentFactory.create_agent(self.config)
        agent.load_checkpoint(self.checkpoint_path)
        
        self.inference_engine = SWTInferenceEngine(
            agent=agent,
            feature_processor=self.feature_processor,
            config=self.config
        )
        logger.info("✅ Inference engine initialized")
        
        # 3. Trade Executor
        self.trade_executor = SWTTradeExecutor(self.config)
        await self.trade_executor.start()
        logger.info("✅ Trade executor initialized")
        
        # 4. Position Reconciler
        self.position_reconciler = SWTPositionReconciler(
            config=self.config,
            trade_executor=self.trade_executor
        )
        await self.position_reconciler.start()
        logger.info("✅ Position reconciler initialized")
        
        # 5. Live Monitor
        self.live_monitor = SWTLiveMonitor(self.config)
        await self.live_monitor.start()
        logger.info("✅ Live monitor initialized")
        
        # 6. Data Feed (initialized last as it starts streaming immediately)
        self.data_feed = OANDADataFeed(
            config=self.config,
            on_bar_callback=self._on_market_data,
            on_error_callback=self._on_data_feed_error
        )
        logger.info("✅ Data feed initialized")
    
    async def _start_subsystems(self) -> None:
        """Start all subsystems in correct order"""
        logger.info("🎬 Starting subsystems...")
        
        # Get initial position from broker
        await self.position_reconciler.reconcile_position()
        self.current_position = self.position_reconciler.get_current_position()
        logger.info(f"📊 Initial position: {self.current_position.position_type.value}")
        
        # Start data feed (begins streaming)
        await self.data_feed.start()
        logger.info("✅ All subsystems started")
    
    def _start_new_session(self) -> None:
        """Start new trading session"""
        session_id = f"swt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(timezone.utc)
        )
        
        logger.info(f"📊 Started trading session: {session_id}")
    
    async def _on_market_data(self, bar: StreamingBar) -> None:
        """
        Handle incoming market data (main trading loop)
        
        This is the core event handler that processes each market tick:
        1. Convert to standard format
        2. Update feature processor  
        3. Check if decision should be made
        4. Run inference if needed
        5. Execute trades if required
        6. Update monitoring
        """
        try:
            # Track processing
            self.total_bars_processed += 1
            self.last_bar_time = bar.timestamp
            
            # Convert to standard market data format
            market_data = streaming_bar_to_market_data(bar)
            
            # Update feature processor
            self.feature_processor.add_market_data(market_data)
            
            # Check if we should make a trading decision
            if not self._should_make_decision(bar.timestamp):
                return
            
            # Only make decisions when system is running
            if self.state != TradingSystemState.RUNNING:
                return
            
            # Make trading decision
            await self._make_trading_decision(market_data, bar.mid_price)
            
        except Exception as e:
            await self._handle_error(e, "market_data_processing")
    
    def _should_make_decision(self, current_time: datetime) -> bool:
        """
        Determine if a trading decision should be made
        
        Implements frequency control to prevent overtrading
        (maximum 1 decision per minute)
        """
        # Check if enough time has passed since last decision
        if self.last_decision_time is not None:
            time_since_last = current_time - self.last_decision_time
            if time_since_last < self.min_decision_interval:
                return False
        
        # Check if feature processor is ready
        if not self.feature_processor.is_ready():
            return False
        
        # Check if we have pending orders (wait for execution)
        if self.pending_orders:
            return False
        
        return True
    
    async def _make_trading_decision(self, market_data: MarketDataPoint, current_price: float) -> None:
        """Make and execute trading decision"""
        try:
            decision_start = time.time()
            
            # Run inference
            trading_decision = await self._run_inference(current_price)
            
            # Record decision metrics
            decision_time = time.time() - decision_start
            self.total_decisions_made += 1
            self.last_decision_time = datetime.now(timezone.utc)
            
            # Log decision
            logger.info(f"🧠 Decision: {trading_decision.action.value} "
                       f"(confidence: {trading_decision.confidence:.1%}, "
                       f"time: {decision_time*1000:.1f}ms)")
            
            # Execute trading action if needed
            if trading_decision.action != TradingAction.HOLD:
                await self._execute_trading_decision(trading_decision, current_price)
            
            # Update monitoring
            if self.live_monitor:
                await self.live_monitor.record_decision(trading_decision, decision_time)
            
            # Notify callbacks
            for callback in self.decision_callbacks:
                try:
                    callback(trading_decision)
                except Exception as e:
                    logger.warning(f"⚠️ Decision callback error: {e}")
                    
        except Exception as e:
            await self._handle_error(e, "trading_decision")
    
    async def _run_inference(self, current_price: float) -> TradingDecision:
        """Run inference to get trading decision"""
        try:
            # Get current position state
            position_state = await self.position_reconciler.get_position_state()
            
            # Process observation
            observation = self.feature_processor.process_observation(
                position_state=position_state,
                current_price=current_price
            )
            
            # Run inference
            trading_decision = await self.inference_engine.get_trading_decision(
                observation=observation,
                current_position=position_state,
                deterministic=False  # Use stochastic for live trading
            )
            
            # Apply minimum confidence filter
            min_confidence = self.config.trading_config.min_confidence
            if trading_decision.confidence < min_confidence:
                logger.debug(f"🔒 Decision filtered: confidence {trading_decision.confidence:.1%} < {min_confidence:.1%}")
                trading_decision.action = TradingAction.HOLD
                trading_decision.confidence = 0.0
            
            return trading_decision
            
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            # Return safe default
            return TradingDecision(
                action=TradingAction.HOLD,
                confidence=0.0,
                value_estimate=0.0,
                policy_distribution=[0.25, 0.25, 0.25, 0.25],  # Uniform
                agent_type=self.config.agent_system,
                model_confidence=0.0
            )
    
    async def _execute_trading_decision(self, decision: TradingDecision, current_price: float) -> None:
        """Execute trading decision via trade executor"""
        try:
            # Create order based on decision
            order = self._create_order_from_decision(decision)
            
            # Add to pending orders
            pending_id = f"decision_{self.total_decisions_made}"
            self.pending_orders.append(pending_id)
            
            logger.info(f"📤 Executing {decision.action.value} order...")
            
            # Execute order
            execution_report = await self.trade_executor.execute_order(
                order=order,
                expected_price=current_price
            )
            
            # Remove from pending
            if pending_id in self.pending_orders:
                self.pending_orders.remove(pending_id)
            
            # Update position state
            await self.position_reconciler.process_execution(execution_report)
            self.current_position = self.position_reconciler.get_current_position()
            
            # Update session statistics
            if self.current_session and execution_report.filled_units > 0:
                self.current_session.total_trades += 1
                if execution_report.pl_realized:
                    self.current_session.total_pnl += execution_report.pl_realized
            
            self.total_trades_executed += 1
            
            # Update monitoring
            if self.live_monitor:
                await self.live_monitor.record_execution(execution_report)
            
            # Notify callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(execution_report)
                except Exception as e:
                    logger.warning(f"⚠️ Trade callback error: {e}")
            
            logger.info(f"✅ Trade executed: {execution_report.status.value}")
            
        except Exception as e:
            # Remove from pending orders on failure
            if pending_id in self.pending_orders:
                self.pending_orders.remove(pending_id)
            
            await self._handle_error(e, "trade_execution")
    
    def _create_order_from_decision(self, decision: TradingDecision) -> OrderRequest:
        """Create order request from trading decision"""
        instrument = self.config.live_trading_config.instrument
        position_size = self.config.trading_config.position_size
        
        # Create client tag for tracking
        client_tag = f"swt_decision_{self.total_decisions_made}"
        
        if decision.action == TradingAction.CLOSE:
            # For close, units should match current position
            current_units = abs(self.current_position.units)
            if current_units == 0:
                raise TradingSystemError("Cannot close position - no open position")
            
            return create_market_order(
                action=TradingAction.CLOSE,
                instrument=instrument,
                units=current_units,
                client_tag=client_tag
            )
        else:
            # For BUY/SELL, use configured position size
            return create_market_order(
                action=decision.action,
                instrument=instrument,
                units=position_size,
                client_tag=client_tag
            )
    
    async def _cancel_pending_orders(self) -> None:
        """Cancel any pending orders during shutdown"""
        if not self.pending_orders:
            return
        
        logger.info(f"🚫 Cancelling {len(self.pending_orders)} pending orders...")
        
        # In a full implementation, we would call OANDA API to cancel orders
        # For now, we just clear the list
        self.pending_orders.clear()
        logger.info("✅ Pending orders cleared")
    
    async def _on_data_feed_error(self, error: Exception) -> None:
        """Handle data feed errors"""
        logger.error(f"📡 Data feed error: {error}")
        await self._handle_error(error, "data_feed")
    
    async def _handle_error(self, error: Exception, context: str) -> None:
        """Handle system errors with appropriate response"""
        self.error_count += 1
        current_time = datetime.now(timezone.utc)
        self.error_timestamps.append(current_time)
        
        # Clean old error timestamps (keep last hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.error_timestamps = [t for t in self.error_timestamps if t > cutoff_time]
        
        # Check error rate
        if len(self.error_timestamps) > self.max_errors_per_hour:
            logger.error("💥 Too many errors - stopping system")
            self.state = TradingSystemState.ERROR
            await self.stop()
            return
        
        # Log error with context
        logger.error(f"❌ System error in {context}: {error}")
        
        # Update session error count
        if self.current_session:
            self.current_session.errors_encountered += 1
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")
    
    async def _save_session_results(self) -> None:
        """Save trading session results"""
        if not self.current_session:
            return
        
        try:
            session_data = {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time.isoformat(),
                "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                "total_trades": self.current_session.total_trades,
                "total_pnl": self.current_session.total_pnl,
                "decisions_made": self.current_session.decisions_made,
                "errors_encountered": self.current_session.errors_encountered,
                "bars_processed": self.total_bars_processed,
                "system_stats": self.get_system_stats()
            }
            
            logger.info(f"💾 Session completed: {session_data['total_trades']} trades, "
                       f"${session_data['total_pnl']:.2f} P&L")
            
            # In a full implementation, save to database or file
            # For now, just log the results
            
        except Exception as e:
            logger.error(f"❌ Failed to save session results: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        uptime = None
        if self.system_start_time:
            uptime = (datetime.now(timezone.utc) - self.system_start_time).total_seconds()
        
        return {
            "state": self.state.value,
            "uptime_seconds": uptime,
            "total_bars_processed": self.total_bars_processed,
            "total_decisions_made": self.total_decisions_made,
            "total_trades_executed": self.total_trades_executed,
            "error_count": self.error_count,
            "pending_orders": len(self.pending_orders),
            "current_position": {
                "type": self.current_position.position_type.value,
                "units": self.current_position.units,
                "entry_price": self.current_position.entry_price,
                "unrealized_pnl": self.current_position.unrealized_pnl
            },
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed system diagnostics"""
        stats = self.get_system_stats()
        
        component_status = {}
        
        if self.data_feed:
            component_status["data_feed"] = self.data_feed.get_status()
        
        if self.trade_executor:
            component_status["trade_executor"] = self.trade_executor.get_execution_stats()
        
        if self.position_reconciler:
            component_status["position_reconciler"] = self.position_reconciler.get_reconciliation_stats()
        
        if self.live_monitor:
            component_status["live_monitor"] = self.live_monitor.get_monitoring_stats()
        
        return {
            **stats,
            "components": component_status,
            "session": {
                "session_id": self.current_session.session_id if self.current_session else None,
                "session_trades": self.current_session.total_trades if self.current_session else 0,
                "session_pnl": self.current_session.total_pnl if self.current_session else 0.0
            }
        }
    
    # Callback registration methods
    def add_decision_callback(self, callback: Callable[[TradingDecision], None]) -> None:
        """Add callback for trading decisions"""
        self.decision_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[ExecutionReport], None]) -> None:
        """Add callback for trade executions"""
        self.trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for system errors"""
        self.error_callbacks.append(callback)