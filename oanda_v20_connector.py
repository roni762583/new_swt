#!/usr/bin/env python3
"""
OANDA v20 API Data Connector
Production-ready connector for real-time market data using OANDA's official v20 REST API
"""

import os
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from collections import deque
import time

logger = logging.getLogger(__name__)

class OandaV20Connector:
    """
    OANDA v20 API connector for real-time forex data
    Uses official OANDA REST API v20 (no other ports or MT5)
    """

    def __init__(self, account_id: str = None, access_token: str = None, environment: str = "practice"):
        """
        Initialize OANDA v20 API connector

        Args:
            account_id: OANDA account ID
            access_token: OANDA API access token
            environment: 'practice' or 'live'
        """
        # Get credentials from environment or parameters
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        self.access_token = access_token or os.getenv("OANDA_ACCESS_TOKEN")
        self.environment = environment

        # Set API endpoints based on environment
        if self.environment == "live":
            self.api_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:  # practice
            self.api_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"

        # API headers
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX"
        }

        # Market data buffer
        self.price_buffer = deque(maxlen=300)  # Keep last 5 minutes of ticks
        self.candle_buffer = deque(maxlen=256)  # For WST processing
        self.last_price = None
        self.last_update_time = None

        # Connection state
        self.connected = False
        self.session = None
        self.stream_task = None

        logger.info(f"🔌 OANDA v20 connector initialized ({self.environment} environment)")

    async def connect(self):
        """Establish connection to OANDA v20 API"""
        try:
            if not self.account_id or not self.access_token:
                raise ValueError("OANDA credentials not configured. Set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN")

            # Create aiohttp session
            self.session = aiohttp.ClientSession()

            # Test connection with account endpoint
            async with self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}",
                headers=self.headers
            ) as response:
                if response.status == 200:
                    account_data = await response.json()
                    self.connected = True
                    logger.info(f"✅ Connected to OANDA v20 API - Account: {account_data['account']['alias']}")
                else:
                    error_text = await response.text()
                    raise ConnectionError(f"OANDA API error: {response.status} - {error_text}")

            # Start price stream
            await self.start_price_stream()

        except Exception as e:
            logger.error(f"❌ OANDA connection failed: {e}")
            self.connected = False
            raise

    async def start_price_stream(self, instruments: List[str] = ["GBP_JPY"]):
        """Start streaming real-time prices"""
        try:
            # Build stream URL
            instruments_str = ",".join(instruments)
            stream_endpoint = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
            params = {"instruments": instruments_str}

            # Start streaming task
            self.stream_task = asyncio.create_task(
                self._stream_prices(stream_endpoint, params)
            )
            logger.info(f"📊 Started OANDA price stream for {instruments}")

        except Exception as e:
            logger.error(f"❌ Failed to start price stream: {e}")
            raise

    async def _stream_prices(self, url: str, params: Dict):
        """Internal method to handle price streaming"""
        retry_count = 0
        max_retries = 5

        while self.connected and retry_count < max_retries:
            try:
                async with self.session.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=None)
                ) as response:
                    async for line in response.content:
                        if not self.connected:
                            break

                        try:
                            data = json.loads(line.decode('utf-8'))

                            if data.get("type") == "PRICE":
                                await self._process_price_update(data)
                            elif data.get("type") == "HEARTBEAT":
                                logger.debug("💓 OANDA heartbeat received")

                            retry_count = 0  # Reset on successful data

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing price data: {e}")

            except asyncio.CancelledError:
                logger.info("Price stream cancelled")
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"Stream error (retry {retry_count}/{max_retries}): {e}")
                await asyncio.sleep(min(2 ** retry_count, 30))

    async def _process_price_update(self, data: Dict):
        """Process incoming price update"""
        try:
            instrument = data.get("instrument")

            # Extract bid/ask prices
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if bids and asks:
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                mid = (bid + ask) / 2
                spread = ask - bid

                # Update current price
                self.last_price = {
                    "instrument": instrument,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": spread,
                    "time": datetime.now(),
                    "timestamp": data.get("time")
                }

                # Add to buffer
                self.price_buffer.append(self.last_price)
                self.last_update_time = datetime.now()

                logger.debug(f"📈 {instrument}: Bid={bid:.5f}, Ask={ask:.5f}, Spread={spread:.5f}")

        except Exception as e:
            logger.error(f"Error processing price update: {e}")

    async def get_candles(self, instrument: str = "GBP_JPY",
                         granularity: str = "M1",
                         count: int = 256) -> List[Dict]:
        """
        Fetch historical candles from OANDA

        Args:
            instrument: Currency pair (e.g., "GBP_JPY")
            granularity: Candle timeframe (M1, M5, H1, etc.)
            count: Number of candles to fetch

        Returns:
            List of candle dictionaries
        """
        try:
            endpoint = f"{self.api_url}/v3/instruments/{instrument}/candles"
            params = {
                "granularity": granularity,
                "count": count,
                "price": "MBA"  # Mid, Bid, Ask
            }

            async with self.session.get(
                endpoint,
                headers=self.headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data.get("candles", [])

                    # Process candles
                    processed_candles = []
                    for candle in candles:
                        if candle.get("complete"):
                            processed_candles.append({
                                "time": candle["time"],
                                "open": float(candle["mid"]["o"]),
                                "high": float(candle["mid"]["h"]),
                                "low": float(candle["mid"]["l"]),
                                "close": float(candle["mid"]["c"]),
                                "volume": int(candle.get("volume", 0))
                            })

                    # Update candle buffer
                    self.candle_buffer.clear()
                    self.candle_buffer.extend(processed_candles)

                    logger.info(f"📊 Fetched {len(processed_candles)} candles for {instrument}")
                    return processed_candles
                else:
                    error = await response.text()
                    logger.error(f"Failed to fetch candles: {response.status} - {error}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    async def get_current_price(self, instrument: str = "GBP_JPY") -> Optional[Dict]:
        """
        Get current price for instrument

        Returns:
            Price dictionary with bid, ask, spread
        """
        if self.last_price and (datetime.now() - self.last_update_time).seconds < 1:
            return self.last_price

        # Fetch fresh price if stream not available
        try:
            endpoint = f"{self.api_url}/v3/accounts/{self.account_id}/pricing"
            params = {"instruments": instrument}

            async with self.session.get(
                endpoint,
                headers=self.headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = data.get("prices", [])

                    if prices:
                        price_data = prices[0]
                        bids = price_data.get("bids", [])
                        asks = price_data.get("asks", [])

                        if bids and asks:
                            bid = float(bids[0]["price"])
                            ask = float(asks[0]["price"])

                            return {
                                "instrument": instrument,
                                "bid": bid,
                                "ask": ask,
                                "mid": (bid + ask) / 2,
                                "spread": ask - bid,
                                "time": datetime.now()
                            }

        except Exception as e:
            logger.error(f"Error fetching current price: {e}")

        return None

    async def place_order(self, instrument: str, units: int, side: str = "buy") -> Optional[Dict]:
        """
        Place market order via OANDA v20 API

        Args:
            instrument: Currency pair
            units: Order size (positive for buy, negative for sell)
            side: 'buy' or 'sell'

        Returns:
            Order response or None if failed
        """
        try:
            endpoint = f"{self.api_url}/v3/accounts/{self.account_id}/orders"

            # Prepare order data
            if side.lower() == "sell":
                units = -abs(units)
            else:
                units = abs(units)

            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT"
                }
            }

            async with self.session.post(
                endpoint,
                headers=self.headers,
                json=order_data
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    logger.info(f"✅ Order placed: {instrument} {units} units")
                    return result
                else:
                    error = await response.text()
                    logger.error(f"Order failed: {response.status} - {error}")
                    return None

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            endpoint = f"{self.api_url}/v3/accounts/{self.account_id}/positions"

            async with self.session.get(
                endpoint,
                headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("positions", [])
                else:
                    logger.error(f"Failed to get positions: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    async def close_position(self, instrument: str) -> bool:
        """Close position for specific instrument"""
        try:
            endpoint = f"{self.api_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"

            close_data = {
                "longUnits": "ALL",
                "shortUnits": "ALL"
            }

            async with self.session.put(
                endpoint,
                headers=self.headers,
                json=close_data
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ Position closed: {instrument}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to close position: {error}")
                    return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def disconnect(self):
        """Disconnect from OANDA API"""
        try:
            self.connected = False

            if self.stream_task:
                self.stream_task.cancel()
                await asyncio.gather(self.stream_task, return_exceptions=True)

            if self.session:
                await self.session.close()

            logger.info("🔌 Disconnected from OANDA v20 API")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def get_market_data_for_wst(self) -> Dict:
        """
        Format market data for WST feature processing

        Returns:
            Dictionary with OHLCV data ready for WST
        """
        if len(self.candle_buffer) < 256:
            logger.warning(f"Insufficient candles for WST: {len(self.candle_buffer)}/256")
            return None

        candles = list(self.candle_buffer)[-256:]

        return {
            "open": [c["open"] for c in candles],
            "high": [c["high"] for c in candles],
            "low": [c["low"] for c in candles],
            "close": [c["close"] for c in candles],
            "volume": [c.get("volume", 0) for c in candles],
            "timestamp": [c["time"] for c in candles]
        }

# Example usage
async def test_oanda_connection():
    """Test OANDA v20 connection"""
    connector = OandaV20Connector(environment="practice")

    try:
        await connector.connect()

        # Get historical candles
        candles = await connector.get_candles("GBP_JPY", "M1", 256)
        logger.info(f"Fetched {len(candles)} candles")

        # Get current price
        price = await connector.get_current_price("GBP_JPY")
        if price:
            logger.info(f"Current GBP/JPY: Bid={price['bid']:.5f}, Ask={price['ask']:.5f}")

        # Wait for streaming prices
        await asyncio.sleep(5)

    finally:
        await connector.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_oanda_connection())