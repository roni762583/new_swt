"""
SWT Position Reconciler
Position management and reconciliation system

This module provides robust position tracking with:
- Real-time position synchronization with OANDA
- Discrepancy detection and resolution
- Position state management using shared types
- Risk monitoring and validation
- Comprehensive position history tracking

CRITICAL: Ensures position consistency between our system and OANDA,
preventing dangerous position mismatches in live trading.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

import aiohttp
from swt_core.types import PositionState, PositionType, TradingAction
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import PositionReconciliationError, ValidationError
from .trade_executor import SWTTradeExecutor, ExecutionReport

logger = logging.getLogger(__name__)


class ReconciliationStatus(Enum):
    """Position reconciliation status"""
    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    RECONCILING = "reconciling"
    ERROR = "error"


@dataclass
class PositionDiscrepancy:
    """Position discrepancy information"""
    timestamp: datetime
    our_position: PositionState
    broker_position: Dict[str, Any]
    discrepancy_type: str
    discrepancy_value: float
    resolved: bool = False
    resolution_action: Optional[str] = None


@dataclass
class ReconciliationStats:
    """Position reconciliation statistics"""
    total_reconciliations: int = 0
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    last_reconciliation_time: Optional[datetime] = None
    average_reconciliation_time_ms: float = 0.0
    sync_status: ReconciliationStatus = ReconciliationStatus.IN_SYNC


class SWTPositionReconciler:
    """
    Production-grade position reconciliation system
    
    Features:
    - Real-time position synchronization with OANDA
    - Automatic discrepancy detection and alerting
    - Position state management using shared types
    - Comprehensive position tracking and history
    - Risk validation and monitoring
    - Performance tracking and diagnostics
    """
    
    def __init__(self, config: SWTConfig, trade_executor: SWTTradeExecutor):
        """
        Initialize position reconciler
        
        Args:
            config: SWT configuration
            trade_executor: Trade executor for getting broker data
        """
        self.config = config
        self.trade_executor = trade_executor
        
        # OANDA API configuration
        self.account_id = config.live_trading_config.oanda_account_id
        self.api_token = config.live_trading_config.oanda_api_token
        self.api_environment = config.live_trading_config.oanda_environment
        self.instrument = config.live_trading_config.instrument
        
        # Position state
        self.current_position = PositionState()
        self.broker_position: Optional[Dict[str, Any]] = None
        self.position_history: List[PositionState] = []
        
        # Reconciliation settings
        self.reconciliation_interval = timedelta(minutes=1)  # Check every minute
        self.position_tolerance_units = 1  # Allow 1 unit difference
        self.price_tolerance_pips = 0.1  # Allow 0.1 pip difference
        
        # Session and tracking
        self.session: Optional[aiohttp.ClientSession] = None
        self.reconciliation_task: Optional[asyncio.Task] = None
        
        # Statistics and monitoring
        self.stats = ReconciliationStats()
        self.discrepancies: List[PositionDiscrepancy] = []
        self.max_discrepancy_history = 100
        
        # Performance tracking
        self.last_broker_fetch_time = 0.0
        self.broker_fetch_count = 0
        self.total_fetch_time = 0.0
        
        logger.info(f"⚖️ SWTPositionReconciler initialized for {self.instrument}")
    
    async def start(self) -> None:
        """Start the position reconciler"""
        if self.session:
            logger.warning("⚠️ Position reconciler already started")
            return
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=10.0, connect=5.0)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Perform initial reconciliation
        await self.reconcile_position()
        
        # Start periodic reconciliation task
        self.reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        
        logger.info("✅ Position reconciler started")
    
    async def stop(self) -> None:
        """Stop the position reconciler"""
        # Cancel reconciliation task
        if self.reconciliation_task:
            self.reconciliation_task.cancel()
            try:
                await self.reconciliation_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("🛑 Position reconciler stopped")
    
    async def reconcile_position(self) -> bool:
        """
        Perform position reconciliation with broker
        
        Returns:
            True if positions are in sync, False if discrepancy found
        """
        start_time = time.time()
        
        try:
            logger.debug("🔍 Reconciling position with broker...")
            
            # Fetch current position from broker
            broker_position = await self._fetch_broker_position()
            self.broker_position = broker_position
            
            # Compare positions
            discrepancy = self._compare_positions(broker_position)
            
            # Update statistics
            reconciliation_time = (time.time() - start_time) * 1000
            self.stats.total_reconciliations += 1
            self.stats.last_reconciliation_time = datetime.now(timezone.utc)
            
            # Update average reconciliation time
            if self.stats.average_reconciliation_time_ms == 0:
                self.stats.average_reconciliation_time_ms = reconciliation_time
            else:
                self.stats.average_reconciliation_time_ms = (
                    self.stats.average_reconciliation_time_ms * 0.9 + 
                    reconciliation_time * 0.1
                )
            
            if discrepancy:
                # Handle discrepancy
                self.stats.discrepancies_found += 1
                self.stats.sync_status = ReconciliationStatus.OUT_OF_SYNC
                
                await self._handle_position_discrepancy(discrepancy)
                return False
            else:
                # Positions are in sync
                self.stats.sync_status = ReconciliationStatus.IN_SYNC
                logger.debug("✅ Positions in sync")
                return True
        
        except Exception as e:
            self.stats.sync_status = ReconciliationStatus.ERROR
            logger.error(f"❌ Position reconciliation failed: {e}")
            raise PositionReconciliationError(f"Reconciliation failed: {str(e)}", original_error=e)
    
    async def process_execution(self, execution: ExecutionReport) -> None:
        """
        Process execution report and update internal position
        
        Args:
            execution: Trade execution report
        """
        if execution.filled_units == 0:
            logger.debug("📊 No position change - zero fill")
            return
        
        logger.info(f"📊 Processing execution: {execution.filled_units} units at {execution.execution_price}")
        
        # Update position based on execution
        previous_position = PositionState(
            position_type=self.current_position.position_type,
            units=self.current_position.units,
            entry_price=self.current_position.entry_price,
            entry_time=self.current_position.entry_time,
            unrealized_pnl=self.current_position.unrealized_pnl
        )
        
        # Save to history
        self.position_history.append(previous_position)
        
        # Update current position
        if self.current_position.position_type == PositionType.FLAT:
            # Opening new position
            if execution.filled_units > 0:
                self.current_position.position_type = PositionType.LONG
            else:
                self.current_position.position_type = PositionType.SHORT
            
            self.current_position.units = abs(execution.filled_units)
            self.current_position.entry_price = execution.execution_price or 0.0
            self.current_position.entry_time = execution.execution_time or datetime.now(timezone.utc)
            self.current_position.unrealized_pnl = 0.0
            
        elif ((self.current_position.position_type == PositionType.LONG and execution.filled_units < 0) or
              (self.current_position.position_type == PositionType.SHORT and execution.filled_units > 0)):
            # Closing or reducing position
            remaining_units = self.current_position.units - abs(execution.filled_units)
            
            if remaining_units <= 0:
                # Position completely closed
                self.current_position = PositionState()  # Reset to flat
            else:
                # Position partially closed
                self.current_position.units = remaining_units
        
        else:
            # Adding to existing position (same direction)
            total_cost = (self.current_position.units * self.current_position.entry_price + 
                         abs(execution.filled_units) * (execution.execution_price or 0.0))
            total_units = self.current_position.units + abs(execution.filled_units)
            
            # Update average entry price
            self.current_position.entry_price = total_cost / total_units
            self.current_position.units = total_units
        
        # Limit position history size
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-100:]
        
        logger.info(f"📊 Position updated: {self.current_position.position_type.value} "
                   f"{self.current_position.units} units @ {self.current_position.entry_price:.5f}")
        
        # Schedule immediate reconciliation to verify
        asyncio.create_task(self._delayed_reconciliation())
    
    async def _delayed_reconciliation(self, delay: float = 2.0) -> None:
        """Perform reconciliation after a delay (allows broker to update)"""
        await asyncio.sleep(delay)
        try:
            await self.reconcile_position()
        except Exception as e:
            logger.warning(f"⚠️ Delayed reconciliation failed: {e}")
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """
        Update unrealized P&L based on current market price
        
        Args:
            current_price: Current market price
        """
        if self.current_position.position_type == PositionType.FLAT:
            self.current_position.unrealized_pnl = 0.0
            return
        
        if self.current_position.entry_price <= 0:
            logger.warning("⚠️ Invalid entry price for P&L calculation")
            return
        
        # Calculate unrealized P&L
        if self.current_position.position_type == PositionType.LONG:
            pnl = (current_price - self.current_position.entry_price) * self.current_position.units
        else:  # SHORT
            pnl = (self.current_position.entry_price - current_price) * self.current_position.units
        
        self.current_position.unrealized_pnl = pnl
        
        # Update position duration
        if self.current_position.entry_time:
            duration = datetime.now(timezone.utc) - self.current_position.entry_time
            self.current_position.duration_minutes = duration.total_seconds() / 60.0
    
    async def _fetch_broker_position(self) -> Dict[str, Any]:
        """Fetch current position from OANDA broker"""
        if not self.session:
            raise PositionReconciliationError("Reconciler not started")
        
        fetch_start = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            base_url = self._get_api_base_url()
            position_url = f"{base_url}/v3/accounts/{self.account_id}/positions/{self.instrument}"
            
            async with self.session.get(position_url, headers=headers) as response:
                if response.status != 200:
                    response_data = await response.json()
                    error_msg = response_data.get("errorMessage", f"HTTP {response.status}")
                    raise PositionReconciliationError(f"Failed to fetch broker position: {error_msg}")
                
                position_data = await response.json()
                broker_position = position_data.get("position", {})
                
                # Update fetch statistics
                fetch_time = time.time() - fetch_start
                self.broker_fetch_count += 1
                self.total_fetch_time += fetch_time
                self.last_broker_fetch_time = fetch_time
                
                return broker_position
        
        except Exception as e:
            logger.error(f"❌ Failed to fetch broker position: {e}")
            raise PositionReconciliationError(f"Broker fetch failed: {str(e)}", original_error=e)
    
    def _compare_positions(self, broker_position: Dict[str, Any]) -> Optional[PositionDiscrepancy]:
        """
        Compare our position with broker position
        
        Returns:
            PositionDiscrepancy if mismatch found, None if in sync
        """
        # Extract broker position information
        long_units = int(broker_position.get("long", {}).get("units", "0"))
        short_units = int(broker_position.get("short", {}).get("units", "0"))
        
        # Calculate net position
        broker_net_units = long_units + short_units  # short_units will be negative
        
        # Determine broker position type
        if broker_net_units == 0:
            broker_position_type = PositionType.FLAT
            broker_units = 0
        elif broker_net_units > 0:
            broker_position_type = PositionType.LONG
            broker_units = broker_net_units
        else:
            broker_position_type = PositionType.SHORT
            broker_units = abs(broker_net_units)
        
        # Compare position types
        if self.current_position.position_type != broker_position_type:
            return PositionDiscrepancy(
                timestamp=datetime.now(timezone.utc),
                our_position=self.current_position,
                broker_position=broker_position,
                discrepancy_type="position_type",
                discrepancy_value=0  # Type mismatch
            )
        
        # Compare position sizes (allow small tolerance)
        our_units = self.current_position.units
        units_diff = abs(our_units - broker_units)
        
        if units_diff > self.position_tolerance_units:
            return PositionDiscrepancy(
                timestamp=datetime.now(timezone.utc),
                our_position=self.current_position,
                broker_position=broker_position,
                discrepancy_type="position_size",
                discrepancy_value=units_diff
            )
        
        # Compare average prices if we have positions
        if broker_position_type != PositionType.FLAT:
            if broker_position_type == PositionType.LONG:
                broker_avg_price = float(broker_position.get("long", {}).get("averagePrice", "0"))
            else:
                broker_avg_price = float(broker_position.get("short", {}).get("averagePrice", "0"))
            
            our_avg_price = self.current_position.entry_price
            
            if our_avg_price > 0 and broker_avg_price > 0:
                price_diff_pips = abs(our_avg_price - broker_avg_price) * 100
                
                if price_diff_pips > self.price_tolerance_pips:
                    return PositionDiscrepancy(
                        timestamp=datetime.now(timezone.utc),
                        our_position=self.current_position,
                        broker_position=broker_position,
                        discrepancy_type="average_price",
                        discrepancy_value=price_diff_pips
                    )
        
        # Positions are in sync
        return None
    
    async def _handle_position_discrepancy(self, discrepancy: PositionDiscrepancy) -> None:
        """Handle position discrepancy"""
        logger.warning(f"⚠️ Position discrepancy detected: {discrepancy.discrepancy_type}")
        logger.warning(f"   Our position: {discrepancy.our_position.position_type.value} "
                      f"{discrepancy.our_position.units} @ {discrepancy.our_position.entry_price}")
        
        # Add to discrepancy history
        self.discrepancies.append(discrepancy)
        if len(self.discrepancies) > self.max_discrepancy_history:
            self.discrepancies = self.discrepancies[-self.max_discrepancy_history:]
        
        # Attempt automatic resolution
        if discrepancy.discrepancy_type in ["position_type", "position_size"]:
            await self._attempt_position_sync(discrepancy)
        else:
            logger.warning(f"⚠️ Manual intervention may be required for {discrepancy.discrepancy_type} discrepancy")
    
    async def _attempt_position_sync(self, discrepancy: PositionDiscrepancy) -> None:
        """Attempt to synchronize position automatically"""
        logger.info("🔄 Attempting automatic position synchronization...")
        
        try:
            # Extract broker position details
            broker_position = discrepancy.broker_position
            long_units = int(broker_position.get("long", {}).get("units", "0"))
            short_units = int(broker_position.get("short", {}).get("units", "0"))
            broker_net_units = long_units + short_units
            
            # Update our position to match broker
            if broker_net_units == 0:
                self.current_position = PositionState()  # Flat
                discrepancy.resolution_action = "synced_to_flat"
            elif broker_net_units > 0:
                # Long position
                avg_price = float(broker_position.get("long", {}).get("averagePrice", "0"))
                self.current_position.position_type = PositionType.LONG
                self.current_position.units = broker_net_units
                self.current_position.entry_price = avg_price
                self.current_position.entry_time = datetime.now(timezone.utc)  # Approximate
                discrepancy.resolution_action = f"synced_to_long_{broker_net_units}"
            else:
                # Short position
                avg_price = float(broker_position.get("short", {}).get("averagePrice", "0"))
                self.current_position.position_type = PositionType.SHORT
                self.current_position.units = abs(broker_net_units)
                self.current_position.entry_price = avg_price
                self.current_position.entry_time = datetime.now(timezone.utc)  # Approximate
                discrepancy.resolution_action = f"synced_to_short_{abs(broker_net_units)}"
            
            # Mark discrepancy as resolved
            discrepancy.resolved = True
            self.stats.discrepancies_resolved += 1
            self.stats.sync_status = ReconciliationStatus.IN_SYNC
            
            logger.info(f"✅ Position synchronized: {discrepancy.resolution_action}")
            
        except Exception as e:
            logger.error(f"❌ Failed to synchronize position: {e}")
            self.stats.sync_status = ReconciliationStatus.ERROR
    
    async def _reconciliation_loop(self) -> None:
        """Periodic reconciliation loop"""
        logger.info(f"🔄 Starting reconciliation loop (interval: {self.reconciliation_interval})")
        
        while True:
            try:
                await asyncio.sleep(self.reconciliation_interval.total_seconds())
                await self.reconcile_position()
                
            except asyncio.CancelledError:
                logger.info("📊 Reconciliation loop cancelled")
                break
                
            except Exception as e:
                logger.error(f"❌ Reconciliation loop error: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    def _get_api_base_url(self) -> str:
        """Get OANDA API base URL"""
        if self.api_environment == "live":
            return "https://api-fxtrade.oanda.com"
        else:
            return "https://api-fxpractice.oanda.com"
    
    # Public accessor methods
    def get_current_position(self) -> PositionState:
        """Get current position state"""
        return self.current_position
    
    async def get_position_state(self) -> PositionState:
        """Get position state (async version for consistency)"""
        return self.current_position
    
    def get_reconciliation_stats(self) -> Dict[str, Any]:
        """Get reconciliation statistics"""
        avg_fetch_time_ms = 0.0
        if self.broker_fetch_count > 0:
            avg_fetch_time_ms = (self.total_fetch_time / self.broker_fetch_count) * 1000
        
        return {
            "sync_status": self.stats.sync_status.value,
            "total_reconciliations": self.stats.total_reconciliations,
            "discrepancies_found": self.stats.discrepancies_found,
            "discrepancies_resolved": self.stats.discrepancies_resolved,
            "last_reconciliation": self.stats.last_reconciliation_time.isoformat() if self.stats.last_reconciliation_time else None,
            "average_reconciliation_time_ms": self.stats.average_reconciliation_time_ms,
            "broker_fetch_count": self.broker_fetch_count,
            "average_fetch_time_ms": avg_fetch_time_ms,
            "position_history_count": len(self.position_history),
            "recent_discrepancies": len([d for d in self.discrepancies if not d.resolved])
        }
    
    def get_position_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent position history"""
        recent_positions = self.position_history[-limit:] if limit else self.position_history
        
        return [
            {
                "position_type": pos.position_type.value,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                "unrealized_pnl": pos.unrealized_pnl,
                "duration_minutes": pos.duration_minutes
            }
            for pos in recent_positions
        ]
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics"""
        stats = self.get_reconciliation_stats()
        
        return {
            **stats,
            "current_position": {
                "type": self.current_position.position_type.value,
                "units": self.current_position.units,
                "entry_price": self.current_position.entry_price,
                "entry_time": self.current_position.entry_time.isoformat() if self.current_position.entry_time else None,
                "unrealized_pnl": self.current_position.unrealized_pnl,
                "duration_minutes": self.current_position.duration_minutes
            },
            "broker_position": self.broker_position,
            "configuration": {
                "reconciliation_interval_seconds": self.reconciliation_interval.total_seconds(),
                "position_tolerance_units": self.position_tolerance_units,
                "price_tolerance_pips": self.price_tolerance_pips
            },
            "recent_discrepancies": [
                {
                    "timestamp": d.timestamp.isoformat(),
                    "type": d.discrepancy_type,
                    "value": d.discrepancy_value,
                    "resolved": d.resolved,
                    "resolution": d.resolution_action
                }
                for d in self.discrepancies[-5:]  # Last 5 discrepancies
            ]
        }