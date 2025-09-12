"""
SWT Live Monitor
Comprehensive performance tracking and monitoring system

This module provides production-grade monitoring with:
- Real-time trading performance metrics
- Decision quality tracking and analysis
- Risk monitoring and alerting
- System health and performance diagnostics
- Alert system for anomalies and thresholds
- Historical performance analysis

CRITICAL: Provides comprehensive visibility into live trading system
performance, essential for risk management and system optimization.
"""

import asyncio
import logging
import time
import statistics
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

from swt_core.types import TradingDecision, TradingAction, PositionType
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import MonitoringError
from .trade_executor import ExecutionReport, OrderStatus

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Alert type enumeration"""
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    EXECUTION = "execution"


@dataclass
class Alert:
    """System alert information"""
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    
    # Execution metrics
    average_slippage_pips: float = 0.0
    fill_rate: float = 100.0
    average_execution_time_ms: float = 0.0
    
    # Decision metrics
    total_decisions: int = 0
    decisions_per_hour: float = 0.0
    average_confidence: float = 0.0
    confidence_vs_outcome_correlation: float = 0.0


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics"""
    # System uptime
    uptime_seconds: float = 0.0
    system_status: str = "running"
    
    # Data processing
    bars_processed: int = 0
    bars_per_minute: float = 0.0
    processing_latency_ms: float = 0.0
    
    # Memory and performance
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Network and connectivity
    data_feed_uptime_percent: float = 100.0
    broker_api_latency_ms: float = 0.0
    connection_errors: int = 0
    
    # Error rates
    total_errors: int = 0
    errors_per_hour: float = 0.0
    last_error_time: Optional[datetime] = None


class SWTLiveMonitor:
    """
    Comprehensive live trading monitoring system
    
    Features:
    - Real-time performance tracking
    - Risk monitoring and alerting
    - Decision quality analysis
    - System health monitoring
    - Alert management and notifications
    - Historical trend analysis
    - Performance diagnostics
    """
    
    def __init__(self, config: SWTConfig):
        """
        Initialize live monitor
        
        Args:
            config: SWT configuration for monitoring parameters
        """
        self.config = config
        
        # Monitoring state
        self.start_time = datetime.now(timezone.utc)
        self.session_start_time = datetime.now(timezone.utc)
        self.last_reset_time = datetime.now(timezone.utc)
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.health_metrics = SystemHealthMetrics()
        
        # Historical data (use deques for efficient rolling windows)
        self.decision_history = deque(maxlen=1000)  # Last 1000 decisions
        self.trade_history = deque(maxlen=500)      # Last 500 trades  
        self.pnl_history = deque(maxlen=1440)       # 24 hours of minute-by-minute P&L
        self.equity_curve = deque(maxlen=1440)      # 24 hours of equity
        
        # Alert system
        self.alerts: List[Alert] = []
        self.max_alert_history = 1000
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring thresholds (configurable)
        self.max_drawdown_threshold = config.trading_config.max_drawdown_percent / 100
        self.max_consecutive_losses = config.trading_config.max_consecutive_losses
        self.min_win_rate_threshold = 0.3  # 30% minimum win rate
        self.max_daily_loss = config.trading_config.max_daily_loss
        
        # Performance analysis
        self.rolling_windows = {
            "1h": deque(maxlen=60),    # 1 hour (minute data)
            "4h": deque(maxlen=240),   # 4 hours  
            "24h": deque(maxlen=1440), # 24 hours
        }
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 60.0  # Monitor every minute
        
        # Trade tracking for correlations
        self.pending_decisions: Dict[int, TradingDecision] = {}
        self.decision_outcomes: List[Tuple[TradingDecision, float]] = []  # (decision, pnl)
        
        logger.info("📊 SWTLiveMonitor initialized")
    
    async def start(self) -> None:
        """Start the monitoring system"""
        self.session_start_time = datetime.now(timezone.utc)
        
        # Start periodic monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ Live monitoring started")
    
    async def stop(self) -> None:
        """Stop the monitoring system"""
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Generate final report
        await self._generate_session_report()
        
        logger.info("🛑 Live monitoring stopped")
    
    async def record_decision(self, decision: TradingDecision, processing_time: float) -> None:
        """
        Record trading decision for analysis
        
        Args:
            decision: Trading decision made
            processing_time: Time taken to make decision (seconds)
        """
        try:
            # Record decision
            decision_record = {
                "timestamp": datetime.now(timezone.utc),
                "action": decision.action,
                "confidence": decision.confidence,
                "value_estimate": decision.value_estimate,
                "processing_time": processing_time,
                "agent_type": decision.agent_type.value
            }
            
            self.decision_history.append(decision_record)
            
            # Update metrics
            self.performance_metrics.total_decisions += 1
            
            # Track for correlation analysis (if trade executed)
            decision_id = len(self.decision_history)
            if decision.action != TradingAction.HOLD:
                self.pending_decisions[decision_id] = decision
            
            # Update decision frequency
            current_time = datetime.now(timezone.utc)
            time_window = timedelta(hours=1)
            recent_decisions = [
                d for d in self.decision_history 
                if current_time - d["timestamp"] <= time_window
            ]
            self.performance_metrics.decisions_per_hour = len(recent_decisions)
            
            # Update average confidence
            if self.performance_metrics.total_decisions > 0:
                total_confidence = sum(d["confidence"] for d in self.decision_history)
                self.performance_metrics.average_confidence = total_confidence / len(self.decision_history)
            
            # Check for decision frequency alerts
            if self.performance_metrics.decisions_per_hour > 60:  # More than 1 per minute
                await self._create_alert(
                    AlertLevel.WARNING,
                    AlertType.PERFORMANCE,
                    f"High decision frequency: {self.performance_metrics.decisions_per_hour:.1f} decisions/hour",
                    {"decisions_per_hour": self.performance_metrics.decisions_per_hour}
                )
            
            logger.debug(f"📊 Recorded decision: {decision.action.value} "
                        f"(confidence: {decision.confidence:.1%}, time: {processing_time*1000:.1f}ms)")
            
        except Exception as e:
            logger.error(f"❌ Failed to record decision: {e}")
    
    async def record_execution(self, execution: ExecutionReport) -> None:
        """
        Record trade execution for analysis
        
        Args:
            execution: Trade execution report
        """
        try:
            # Record execution
            execution_record = {
                "timestamp": datetime.now(timezone.utc),
                "order_id": execution.order_id,
                "status": execution.status,
                "filled_units": execution.filled_units,
                "execution_price": execution.execution_price,
                "slippage_pips": execution.slippage_pips,
                "pl_realized": execution.pl_realized,
                "commission": execution.commission
            }
            
            self.trade_history.append(execution_record)
            
            # Update trade metrics if filled
            if execution.status == OrderStatus.FILLED and execution.filled_units != 0:
                self.performance_metrics.total_trades += 1
                
                # Track P&L
                if execution.pl_realized is not None:
                    pnl = execution.pl_realized
                    self.performance_metrics.total_pnl += pnl
                    
                    # Update win/loss statistics  
                    if pnl > 0:
                        self.performance_metrics.winning_trades += 1
                        self.performance_metrics.gross_profit += pnl
                        self.performance_metrics.current_consecutive_losses = 0
                    elif pnl < 0:
                        self.performance_metrics.losing_trades += 1
                        self.performance_metrics.gross_loss += abs(pnl)
                        self.performance_metrics.current_consecutive_losses += 1
                        
                        # Update max consecutive losses
                        if self.performance_metrics.current_consecutive_losses > self.performance_metrics.max_consecutive_losses:
                            self.performance_metrics.max_consecutive_losses = self.performance_metrics.current_consecutive_losses
                    
                    # Update win rate
                    total_closed_trades = self.performance_metrics.winning_trades + self.performance_metrics.losing_trades
                    if total_closed_trades > 0:
                        self.performance_metrics.win_rate = self.performance_metrics.winning_trades / total_closed_trades
                    
                    # Update profit factor
                    if self.performance_metrics.gross_loss > 0:
                        self.performance_metrics.profit_factor = self.performance_metrics.gross_profit / self.performance_metrics.gross_loss
                    
                    # Update drawdown
                    self._update_drawdown_metrics(pnl)
                    
                    # Add to P&L history
                    self.pnl_history.append({
                        "timestamp": datetime.now(timezone.utc),
                        "pnl": pnl,
                        "cumulative_pnl": self.performance_metrics.total_pnl
                    })
                
                # Update slippage metrics
                if execution.slippage_pips is not None:
                    # Calculate running average slippage
                    total_slippage = 0.0
                    slippage_count = 0
                    
                    for trade in self.trade_history:
                        if trade["slippage_pips"] is not None:
                            total_slippage += trade["slippage_pips"]
                            slippage_count += 1
                    
                    if slippage_count > 0:
                        self.performance_metrics.average_slippage_pips = total_slippage / slippage_count
                
                # Check for performance alerts
                await self._check_performance_alerts()
            
            logger.info(f"📊 Recorded execution: {execution.status.value} "
                       f"({execution.filled_units} units, "
                       f"P&L: ${execution.pl_realized:.2f if execution.pl_realized else 0.0})")
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution: {e}")
    
    def _update_drawdown_metrics(self, pnl: float) -> None:
        """Update drawdown tracking metrics"""
        # Update equity curve
        current_equity = sum(p["pnl"] for p in self.pnl_history) if self.pnl_history else 0.0
        self.equity_curve.append({
            "timestamp": datetime.now(timezone.utc),
            "equity": current_equity
        })
        
        if not self.equity_curve:
            return
        
        # Calculate current drawdown
        peak_equity = max(e["equity"] for e in self.equity_curve)
        current_equity = self.equity_curve[-1]["equity"]
        
        if peak_equity > 0:
            self.performance_metrics.current_drawdown = (peak_equity - current_equity) / peak_equity
        else:
            self.performance_metrics.current_drawdown = 0.0
        
        # Update max drawdown
        if self.performance_metrics.current_drawdown > self.performance_metrics.max_drawdown:
            self.performance_metrics.max_drawdown = self.performance_metrics.current_drawdown
    
    async def _check_performance_alerts(self) -> None:
        """Check performance metrics against thresholds and create alerts"""
        # Drawdown alerts
        if self.performance_metrics.current_drawdown > self.max_drawdown_threshold:
            await self._create_alert(
                AlertLevel.CRITICAL,
                AlertType.RISK,
                f"Maximum drawdown exceeded: {self.performance_metrics.current_drawdown:.1%}",
                {"current_drawdown": self.performance_metrics.current_drawdown}
            )
        
        # Consecutive losses alert
        if self.performance_metrics.current_consecutive_losses >= self.max_consecutive_losses:
            await self._create_alert(
                AlertLevel.WARNING,
                AlertType.RISK,
                f"Consecutive losses threshold reached: {self.performance_metrics.current_consecutive_losses}",
                {"consecutive_losses": self.performance_metrics.current_consecutive_losses}
            )
        
        # Win rate alert (only after sufficient trades)
        if (self.performance_metrics.total_trades >= 20 and 
            self.performance_metrics.win_rate < self.min_win_rate_threshold):
            await self._create_alert(
                AlertLevel.WARNING,
                AlertType.PERFORMANCE,
                f"Low win rate: {self.performance_metrics.win_rate:.1%}",
                {"win_rate": self.performance_metrics.win_rate, "total_trades": self.performance_metrics.total_trades}
            )
        
        # Daily loss limit alert
        if self.performance_metrics.total_pnl <= -abs(self.max_daily_loss):
            await self._create_alert(
                AlertLevel.CRITICAL,
                AlertType.RISK,
                f"Daily loss limit reached: ${self.performance_metrics.total_pnl:.2f}",
                {"daily_pnl": self.performance_metrics.total_pnl}
            )
    
    async def _create_alert(self, level: AlertLevel, alert_type: AlertType, 
                           message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Create and process system alert"""
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level=level,
            alert_type=alert_type,
            message=message,
            details=details or {}
        )
        
        # Add to alert history
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alert_history:
            self.alerts = self.alerts[-self.max_alert_history:]
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(level, logger.info)
        
        log_level(f"🚨 ALERT [{level.value.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"❌ Alert callback error: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Periodic monitoring and health checks"""
        logger.info(f"🔄 Starting monitoring loop (interval: {self.monitoring_interval}s)")
        
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._update_system_metrics()
                await self._check_system_health()
                
            except asyncio.CancelledError:
                logger.info("📊 Monitoring loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _update_system_metrics(self) -> None:
        """Update system health metrics"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Update uptime
            self.health_metrics.uptime_seconds = (current_time - self.start_time).total_seconds()
            
            # Update processing metrics
            if len(self.decision_history) > 1:
                recent_decisions = [
                    d for d in self.decision_history 
                    if current_time - d["timestamp"] <= timedelta(minutes=1)
                ]
                
                if recent_decisions:
                    avg_processing_time = statistics.mean(d["processing_time"] for d in recent_decisions)
                    self.health_metrics.processing_latency_ms = avg_processing_time * 1000
            
            # Calculate bars per minute (approximate from decision frequency)
            if self.performance_metrics.decisions_per_hour > 0:
                # Assuming 1 bar = 1 potential decision
                self.health_metrics.bars_per_minute = self.performance_metrics.decisions_per_hour / 60
            
            # Calculate error rate
            time_window = timedelta(hours=1)
            if hasattr(self, 'error_timestamps'):
                recent_errors = [
                    t for t in self.error_timestamps 
                    if current_time - t <= time_window
                ]
                self.health_metrics.errors_per_hour = len(recent_errors)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update system metrics: {e}")
    
    async def _check_system_health(self) -> None:
        """Check system health and create alerts if needed"""
        try:
            # Check processing latency
            if self.health_metrics.processing_latency_ms > 1000:  # > 1 second
                await self._create_alert(
                    AlertLevel.WARNING,
                    AlertType.SYSTEM,
                    f"High processing latency: {self.health_metrics.processing_latency_ms:.1f}ms",
                    {"latency_ms": self.health_metrics.processing_latency_ms}
                )
            
            # Check error rate
            if self.health_metrics.errors_per_hour > 10:
                await self._create_alert(
                    AlertLevel.WARNING,
                    AlertType.SYSTEM,
                    f"High error rate: {self.health_metrics.errors_per_hour} errors/hour",
                    {"errors_per_hour": self.health_metrics.errors_per_hour}
                )
            
        except Exception as e:
            logger.warning(f"⚠️ System health check failed: {e}")
    
    async def _generate_session_report(self) -> None:
        """Generate comprehensive session report"""
        try:
            session_duration = datetime.now(timezone.utc) - self.session_start_time
            
            report = {
                "session_summary": {
                    "duration_hours": session_duration.total_seconds() / 3600,
                    "start_time": self.session_start_time.isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                },
                "trading_performance": self.get_performance_summary(),
                "system_health": self.get_health_summary(),
                "alerts_summary": {
                    "total_alerts": len(self.alerts),
                    "critical_alerts": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
                    "warning_alerts": len([a for a in self.alerts if a.level == AlertLevel.WARNING]),
                }
            }
            
            logger.info("📋 Session Report Generated:")
            logger.info(f"   Duration: {report['session_summary']['duration_hours']:.1f} hours")
            logger.info(f"   Total Trades: {report['trading_performance']['total_trades']}")
            logger.info(f"   Win Rate: {report['trading_performance']['win_rate']:.1%}")
            logger.info(f"   Total P&L: ${report['trading_performance']['total_pnl']:.2f}")
            logger.info(f"   Max Drawdown: {report['trading_performance']['max_drawdown']:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate session report: {e}")
    
    # Public API methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        return {
            "total_trades": self.performance_metrics.total_trades,
            "winning_trades": self.performance_metrics.winning_trades,
            "losing_trades": self.performance_metrics.losing_trades,
            "win_rate": self.performance_metrics.win_rate,
            "total_pnl": self.performance_metrics.total_pnl,
            "gross_profit": self.performance_metrics.gross_profit,
            "gross_loss": self.performance_metrics.gross_loss,
            "profit_factor": self.performance_metrics.profit_factor,
            "max_drawdown": self.performance_metrics.max_drawdown,
            "current_drawdown": self.performance_metrics.current_drawdown,
            "max_consecutive_losses": self.performance_metrics.max_consecutive_losses,
            "current_consecutive_losses": self.performance_metrics.current_consecutive_losses,
            "average_slippage_pips": self.performance_metrics.average_slippage_pips,
            "total_decisions": self.performance_metrics.total_decisions,
            "decisions_per_hour": self.performance_metrics.decisions_per_hour,
            "average_confidence": self.performance_metrics.average_confidence
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current system health summary"""
        return {
            "uptime_seconds": self.health_metrics.uptime_seconds,
            "uptime_hours": self.health_metrics.uptime_seconds / 3600,
            "system_status": self.health_metrics.system_status,
            "bars_processed": self.health_metrics.bars_processed,
            "bars_per_minute": self.health_metrics.bars_per_minute,
            "processing_latency_ms": self.health_metrics.processing_latency_ms,
            "total_errors": self.health_metrics.total_errors,
            "errors_per_hour": self.health_metrics.errors_per_hour,
            "last_error_time": self.health_metrics.last_error_time.isoformat() if self.health_metrics.last_error_time else None
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        recent_alerts = self.alerts[-limit:] if limit else self.alerts
        
        return [
            {
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "type": alert.alert_type.value,
                "message": alert.message,
                "details": alert.details,
                "acknowledged": alert.acknowledged
            }
            for alert in recent_alerts
        ]
    
    def get_equity_curve(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get equity curve data for specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            {
                "timestamp": point["timestamp"].isoformat(),
                "equity": point["equity"]
            }
            for point in self.equity_curve
            if point["timestamp"] >= cutoff_time
        ]
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        return {
            "performance": self.get_performance_summary(),
            "health": self.get_health_summary(),
            "history_sizes": {
                "decisions": len(self.decision_history),
                "trades": len(self.trade_history),
                "pnl_points": len(self.pnl_history),
                "equity_points": len(self.equity_curve)
            },
            "alerts": {
                "total": len(self.alerts),
                "unacknowledged": len([a for a in self.alerts if not a.acknowledged]),
                "by_level": {
                    level.value: len([a for a in self.alerts if a.level == level])
                    for level in AlertLevel
                }
            }
        }
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """Acknowledge an alert"""
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                return True
            return False
        except Exception:
            return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def reset_session_stats(self) -> None:
        """Reset session statistics (useful for new trading day)"""
        self.session_start_time = datetime.now(timezone.utc)
        self.last_reset_time = datetime.now(timezone.utc)
        
        # Reset performance metrics but keep historical data
        self.performance_metrics = PerformanceMetrics()
        
        logger.info("📊 Session statistics reset")