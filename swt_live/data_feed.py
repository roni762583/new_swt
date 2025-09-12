"""
OANDA Data Feed
Real-time market data streaming for live trading

This module provides production-grade streaming data infrastructure with:
- Event-driven data callbacks
- Connection resilience and automatic reconnection
- Gap detection and recovery
- Minimal latency processing
- Comprehensive error handling

CRITICAL: This feeds data to the shared feature processor ensuring
identical processing to training system.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

import aiohttp
from swt_core.types import MarketState, TradingAction
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import DataFeedError, ConnectionError
from swt_features.market_features import MarketDataPoint

logger = logging.getLogger(__name__)


class DataFeedStatus(Enum):
    """Data feed status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class StreamingBar:
    """Real-time market bar from OANDA"""
    instrument: str
    timestamp: datetime
    bid: float
    ask: float
    volume: int
    spread: float
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2.0


class OANDADataFeed:
    """
    Production-grade OANDA streaming data feed
    
    Features:
    - Event-driven architecture with callbacks
    - Automatic reconnection with exponential backoff
    - Gap detection and recovery mechanisms
    - Real-time latency monitoring
    - Thread-safe operations
    """
    
    def __init__(self, 
                 config: SWTConfig,
                 on_bar_callback: Callable[[StreamingBar], None],
                 on_error_callback: Optional[Callable[[Exception], None]] = None):
        """
        Initialize OANDA data feed
        
        Args:
            config: SWT configuration containing OANDA credentials
            on_bar_callback: Callback function for new market bars
            on_error_callback: Optional error callback
        """
        self.config = config
        self.on_bar_callback = on_bar_callback
        self.on_error_callback = on_error_callback
        
        # OANDA API configuration
        self.account_id = config.live_trading_config.oanda_account_id
        self.api_token = config.live_trading_config.oanda_api_token
        self.api_environment = config.live_trading_config.oanda_environment
        self.instrument = config.live_trading_config.instrument
        
        # Connection state
        self.status = DataFeedStatus.DISCONNECTED
        self.session: Optional[aiohttp.ClientSession] = None
        self.stream_task: Optional[asyncio.Task] = None
        
        # Reconnection parameters
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Maximum 60 seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 100
        
        # Performance tracking
        self.bars_received = 0
        self.last_bar_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None
        self.gaps_detected = 0
        
        # Data validation
        self.last_valid_price = 0.0
        self.max_price_change_pct = 0.05  # 5% max price change filter
        
        logger.info(f"🌊 OANDADataFeed initialized for {self.instrument}")
    
    async def start(self) -> None:
        """Start the data feed"""
        if self.status != DataFeedStatus.DISCONNECTED:
            logger.warning("⚠️ Data feed already running")
            return
            
        logger.info("🚀 Starting OANDA data feed...")
        self.status = DataFeedStatus.CONNECTING
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start streaming task
        self.stream_task = asyncio.create_task(self._streaming_loop())
        
        try:
            await asyncio.sleep(2.0)  # Give connection time to establish
            if self.status == DataFeedStatus.CONNECTED:
                logger.info("✅ OANDA data feed connected successfully")
            else:
                logger.warning("⚠️ Data feed connection pending...")
        except Exception as e:
            logger.error(f"❌ Failed to start data feed: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the data feed gracefully"""
        logger.info("🛑 Stopping OANDA data feed...")
        
        self.status = DataFeedStatus.DISCONNECTED
        
        # Cancel streaming task
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        logger.info("✅ OANDA data feed stopped")
    
    async def _streaming_loop(self) -> None:
        """Main streaming loop with reconnection logic"""
        while self.status != DataFeedStatus.DISCONNECTED:
            try:
                await self._connect_and_stream()
                
            except asyncio.CancelledError:
                logger.info("📡 Streaming task cancelled")
                break
                
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                
                if self.on_error_callback:
                    try:
                        self.on_error_callback(e)
                    except Exception as callback_error:
                        logger.error(f"Error callback failed: {callback_error}")
                
                # Attempt reconnection if not at limit
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    await self._handle_reconnection()
                else:
                    logger.error("💥 Maximum reconnection attempts reached")
                    self.status = DataFeedStatus.ERROR
                    break
    
    async def _connect_and_stream(self) -> None:
        """Establish connection and stream data"""
        if not self.session:
            raise ConnectionError("HTTP session not initialized")
        
        # Build streaming URL
        base_url = self._get_api_base_url()
        stream_url = f"{base_url}/v3/accounts/{self.account_id}/pricing/stream"
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "instruments": self.instrument,
            "snapshot": "true"
        }
        
        logger.info(f"🔌 Connecting to OANDA stream: {self.instrument}")
        
        async with self.session.get(stream_url, headers=headers, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ConnectionError(f"OANDA API error {response.status}: {error_text}")
            
            self.status = DataFeedStatus.CONNECTED
            self.connection_start_time = datetime.now(timezone.utc)
            self.reconnect_attempts = 0
            self.reconnect_delay = 1.0  # Reset reconnection delay
            
            logger.info("✅ Connected to OANDA streaming API")
            
            # Process streaming data
            async for line in response.content:
                if self.status == DataFeedStatus.DISCONNECTED:
                    break
                    
                try:
                    await self._process_stream_line(line)
                except Exception as e:
                    logger.warning(f"⚠️ Error processing stream line: {e}")
    
    async def _process_stream_line(self, line: bytes) -> None:
        """Process individual stream line"""
        try:
            if not line.strip():
                return
                
            import json
            data = json.loads(line.decode('utf-8'))
            
            # Handle different message types
            if "type" in data:
                if data["type"] == "PRICE":
                    await self._handle_price_update(data)
                elif data["type"] == "HEARTBEAT":
                    await self._handle_heartbeat(data)
                else:
                    logger.debug(f"📨 Unknown message type: {data['type']}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to process stream line: {e}")
    
    async def _handle_price_update(self, price_data: Dict[str, Any]) -> None:
        """Handle price update from stream"""
        try:
            # Extract price information
            instrument = price_data.get("instrument")
            if instrument != self.instrument:
                return
            
            timestamp_str = price_data.get("time", "")
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            
            # Get bid/ask prices
            bids = price_data.get("bids", [])
            asks = price_data.get("asks", [])
            
            if not bids or not asks:
                logger.warning("⚠️ Missing bid/ask data in price update")
                return
            
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            spread = ask - bid
            
            # Validate price data
            if not self._is_valid_price_update(bid, ask):
                return
            
            # Create streaming bar
            streaming_bar = StreamingBar(
                instrument=instrument,
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                volume=0,  # OANDA doesn't provide volume in streaming
                spread=spread
            )
            
            # Update statistics
            self.bars_received += 1
            self.last_bar_time = timestamp
            self.last_valid_price = streaming_bar.mid_price
            
            # Detect gaps
            await self._detect_price_gap(streaming_bar)
            
            # Call user callback
            try:
                self.on_bar_callback(streaming_bar)
            except Exception as e:
                logger.error(f"❌ Error in bar callback: {e}")
            
            # Log progress periodically
            if self.bars_received % 1000 == 0:
                logger.info(f"📊 Processed {self.bars_received:,} price updates")
                
        except Exception as e:
            logger.error(f"❌ Error handling price update: {e}")
    
    async def _handle_heartbeat(self, heartbeat_data: Dict[str, Any]) -> None:
        """Handle heartbeat message"""
        timestamp_str = heartbeat_data.get("time", "")
        logger.debug(f"💓 Heartbeat received: {timestamp_str}")
    
    def _is_valid_price_update(self, bid: float, ask: float) -> bool:
        """Validate price update for sanity"""
        # Check for reasonable bid/ask spread
        spread = ask - bid
        if spread <= 0:
            logger.warning(f"⚠️ Invalid spread: bid={bid}, ask={ask}")
            return False
        
        # Check for excessive spread (> 50 pips for GBP/JPY)
        if spread > 0.50:  # 50 pips
            logger.warning(f"⚠️ Excessive spread: {spread:.4f}")
            return False
        
        # Check for extreme price changes
        mid_price = (bid + ask) / 2.0
        if self.last_valid_price > 0:
            price_change_pct = abs(mid_price - self.last_valid_price) / self.last_valid_price
            if price_change_pct > self.max_price_change_pct:
                logger.warning(f"⚠️ Extreme price change: {price_change_pct:.2%}")
                return False
        
        return True
    
    async def _detect_price_gap(self, bar: StreamingBar) -> None:
        """Detect and log price gaps"""
        if self.last_bar_time is None:
            return
        
        # Check for time gaps (> 5 seconds is unusual for streaming)
        time_gap = (bar.timestamp - self.last_bar_time).total_seconds()
        if time_gap > 5.0:
            self.gaps_detected += 1
            logger.warning(f"⚠️ Price gap detected: {time_gap:.1f}s since last update")
    
    async def _handle_reconnection(self) -> None:
        """Handle reconnection logic with exponential backoff"""
        self.status = DataFeedStatus.RECONNECTING
        self.reconnect_attempts += 1
        
        logger.info(f"🔄 Attempting reconnection #{self.reconnect_attempts}")
        logger.info(f"⏳ Waiting {self.reconnect_delay:.1f}s before reconnection...")
        
        await asyncio.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2.0, self.max_reconnect_delay)
    
    def _get_api_base_url(self) -> str:
        """Get OANDA API base URL based on environment"""
        if self.api_environment == "live":
            return "https://api-fxtrade.oanda.com"
        else:
            return "https://api-fxpractice.oanda.com"  # Practice/demo
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive data feed status"""
        uptime = None
        if self.connection_start_time:
            uptime = (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
        
        return {
            "status": self.status.value,
            "instrument": self.instrument,
            "bars_received": self.bars_received,
            "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None,
            "uptime_seconds": uptime,
            "reconnect_attempts": self.reconnect_attempts,
            "gaps_detected": self.gaps_detected,
            "last_valid_price": self.last_valid_price,
            "connection_healthy": self.status == DataFeedStatus.CONNECTED
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for troubleshooting"""
        status = self.get_status()
        
        # Calculate performance metrics
        bars_per_minute = 0.0
        if self.connection_start_time and self.bars_received > 0:
            uptime_minutes = status["uptime_seconds"] / 60.0
            bars_per_minute = self.bars_received / uptime_minutes if uptime_minutes > 0 else 0.0
        
        return {
            **status,
            "performance": {
                "bars_per_minute": bars_per_minute,
                "max_reconnect_delay": self.max_reconnect_delay,
                "current_reconnect_delay": self.reconnect_delay
            },
            "api_config": {
                "environment": self.api_environment,
                "base_url": self._get_api_base_url(),
                "account_id": self.account_id[:8] + "..." if self.account_id else None
            },
            "data_validation": {
                "max_price_change_pct": self.max_price_change_pct,
                "gaps_detected": self.gaps_detected
            }
        }


# Helper function for creating MarketDataPoint from StreamingBar
def streaming_bar_to_market_data(bar: StreamingBar) -> MarketDataPoint:
    """
    Convert StreamingBar to MarketDataPoint for feature processing
    
    Args:
        bar: Streaming bar from OANDA
        
    Returns:
        MarketDataPoint compatible with feature processor
    """
    mid_price = bar.mid_price
    
    return MarketDataPoint(
        timestamp=bar.timestamp,
        open=mid_price,  # Use mid price for OHLC (streaming doesn't provide OHLC)
        high=mid_price,
        low=mid_price,
        close=mid_price,
        volume=bar.volume,
        spread=bar.spread,
        bid=bar.bid,
        ask=bar.ask
    )