#!/usr/bin/env python3
"""
Async OANDA v20 API Client for High-Performance Trading
Full async/await implementation for concurrent operations
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Tuple, AsyncIterator
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by OANDA"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"


@dataclass
class PriceUpdate:
    """Real-time price update"""
    instrument: str
    time: str
    bid: float
    ask: float
    spread: float


@dataclass
class Position:
    """Trading position data"""
    instrument: str
    units: int
    avg_price: float
    unrealized_pnl: float
    side: str  # 'long' or 'short'


class AsyncOandaClient:
    """
    Async OANDA v20 API client for production trading
    Provides high-performance concurrent operations
    """

    def __init__(self, max_connections: int = 10):
        """
        Initialize async OANDA client

        Args:
            max_connections: Maximum concurrent connections
        """
        # Get credentials from environment
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.environment = os.getenv('OANDA_ENVIRONMENT', 'practice')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        # Set up base URLs based on environment
        if self.environment == 'live':
            self.rest_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:
            self.rest_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"

        # Headers for API requests
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }

        # Connection pool configuration
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        self.session: Optional[aiohttp.ClientSession] = None

        # Price streaming
        self.price_streams: Dict[str, asyncio.Task] = {}
        self.price_queues: Dict[str, asyncio.Queue] = {}

        # Cache for frequently accessed data
        self.position_cache: Dict[str, Position] = {}
        self.account_cache: Dict[str, Any] = {}
        self.cache_ttl = 1.0  # 1 second cache TTL

        logger.info(f"🏦 Async OANDA Client initialized: {self.environment} environment")
        logger.info(f"   Account: {self.account_id}, Max connections: {max_connections}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Initialize HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("📡 Async session connected")

    async def disconnect(self):
        """Close HTTP session and cleanup"""
        # Stop all price streams
        for task in self.price_streams.values():
            task.cancel()

        # Wait for streams to stop
        if self.price_streams:
            await asyncio.gather(*self.price_streams.values(), return_exceptions=True)

        # Close session
        if self.session:
            await self.session.close()
            await asyncio.sleep(0.1)  # Allow cleanup

        logger.info("📡 Async session disconnected")

    async def get_account_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get account summary with caching

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Account summary data
        """
        # Check cache
        if use_cache and self.account_cache:
            cache_age = time.time() - self.account_cache.get('_timestamp', 0)
            if cache_age < self.cache_ttl:
                return self.account_cache

        url = f"{self.rest_url}/v3/accounts/{self.account_id}/summary"

        try:
            async with self.session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()

                account = data['account']
                summary = {
                    'balance': float(account['balance']),
                    'unrealized_pnl': float(account.get('unrealizedPL', 0)),
                    'nav': float(account['NAV']),
                    'margin_used': float(account.get('marginUsed', 0)),
                    'margin_available': float(account.get('marginAvailable', 0)),
                    'open_position_count': int(account.get('openPositionCount', 0)),
                    'open_trade_count': int(account.get('openTradeCount', 0)),
                    '_timestamp': time.time()
                }

                # Update cache
                self.account_cache = summary
                return summary

        except Exception as e:
            logger.error(f"❌ Failed to get account summary: {e}")
            raise

    async def stream_prices(self, instruments: List[str]) -> AsyncIterator[PriceUpdate]:
        """
        Stream real-time prices asynchronously

        Args:
            instruments: List of instruments to stream

        Yields:
            PriceUpdate objects
        """
        instruments_str = ','.join(instruments)
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        params = {'instruments': instruments_str}

        logger.info(f"📊 Starting async price stream for: {instruments_str}")

        try:
            async with self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=None)
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))

                            if data['type'] == 'PRICE':
                                # Process price update
                                bid = float(data['bids'][0]['price']) if data.get('bids') else None
                                ask = float(data['asks'][0]['price']) if data.get('asks') else None

                                if bid and ask:
                                    yield PriceUpdate(
                                        instrument=data['instrument'],
                                        time=data['time'],
                                        bid=bid,
                                        ask=ask,
                                        spread=ask - bid
                                    )

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse price data: {line}")
                        except Exception as e:
                            logger.error(f"Error processing price: {e}")

        except asyncio.CancelledError:
            logger.info("📊 Price stream cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Price streaming error: {e}")
            raise

    async def start_price_stream(self, instrument: str, queue_size: int = 100):
        """
        Start a background price stream for an instrument

        Args:
            instrument: Instrument to stream
            queue_size: Size of price queue
        """
        if instrument in self.price_streams:
            logger.warning(f"Price stream already active for {instrument}")
            return

        # Create queue for prices
        self.price_queues[instrument] = asyncio.Queue(maxsize=queue_size)

        # Start streaming task
        async def stream_worker():
            try:
                async for price in self.stream_prices([instrument]):
                    # Add to queue (drop old if full)
                    if self.price_queues[instrument].full():
                        try:
                            self.price_queues[instrument].get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    await self.price_queues[instrument].put(price)

            except Exception as e:
                logger.error(f"Stream worker error for {instrument}: {e}")

        self.price_streams[instrument] = asyncio.create_task(stream_worker())
        logger.info(f"📊 Started background stream for {instrument}")

    async def get_latest_price(self, instrument: str) -> Optional[PriceUpdate]:
        """
        Get latest price from stream or API

        Args:
            instrument: Currency pair

        Returns:
            Latest price or None
        """
        # Try to get from stream queue first
        if instrument in self.price_queues:
            try:
                # Get most recent price from queue
                price = None
                while not self.price_queues[instrument].empty():
                    price = await self.price_queues[instrument].get()
                if price:
                    return price
            except:
                pass

        # Fallback to API request
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/pricing"
        params = {'instruments': instrument}

        try:
            async with self.session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                if data['prices']:
                    price = data['prices'][0]
                    bid = float(price['bids'][0]['price']) if price.get('bids') else None
                    ask = float(price['asks'][0]['price']) if price.get('asks') else None

                    if bid and ask:
                        return PriceUpdate(
                            instrument=price['instrument'],
                            time=price['time'],
                            bid=bid,
                            ask=ask,
                            spread=ask - bid
                        )

        except Exception as e:
            logger.error(f"❌ Failed to get price for {instrument}: {e}")

        return None

    async def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Place market order asynchronously

        Args:
            instrument: Currency pair (e.g., 'GBP_JPY')
            units: Number of units (positive for buy, negative for sell)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order result or None if failed
        """
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/orders"

        order = {
            'order': {
                'type': 'MARKET',
                'instrument': instrument,
                'units': str(units),
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT'
            }
        }

        if stop_loss:
            order['order']['stopLossOnFill'] = {
                'price': str(stop_loss),
                'timeInForce': 'GTC'
            }

        if take_profit:
            order['order']['takeProfitOnFill'] = {
                'price': str(take_profit),
                'timeInForce': 'GTC'
            }

        try:
            async with self.session.post(url, headers=self.headers, json=order) as response:
                response.raise_for_status()
                result = await response.json()

                if 'orderFillTransaction' in result:
                    fill = result['orderFillTransaction']
                    logger.info(f"✅ Order filled: {units} units of {instrument} @ {fill.get('price')}")

                    # Invalidate position cache
                    self.position_cache.pop(instrument, None)

                    return {
                        'order_id': fill.get('orderID'),
                        'trade_id': fill.get('tradeOpened', {}).get('tradeID') if fill.get('tradeOpened') else None,
                        'instrument': instrument,
                        'units': int(fill.get('units', 0)),
                        'price': float(fill.get('price', 0)),
                        'time': fill.get('time'),
                        'pnl': float(fill.get('pl', 0))
                    }
                else:
                    logger.warning(f"Order placed but not filled: {result}")

        except aiohttp.ClientResponseError as e:
            logger.error(f"❌ Order failed: {e.message}")
        except Exception as e:
            logger.error(f"❌ Order execution error: {e}")

        return None

    async def close_position(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Close all positions for an instrument

        Args:
            instrument: Currency pair

        Returns:
            Close result or None
        """
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"

        data = {
            'longUnits': 'ALL',
            'shortUnits': 'ALL'
        }

        try:
            async with self.session.put(url, headers=self.headers, json=data) as response:
                response.raise_for_status()
                result = await response.json()

                total_pnl = 0.0
                if 'longOrderFillTransaction' in result:
                    total_pnl += float(result['longOrderFillTransaction'].get('pl', 0))
                if 'shortOrderFillTransaction' in result:
                    total_pnl += float(result['shortOrderFillTransaction'].get('pl', 0))

                # Invalidate position cache
                self.position_cache.pop(instrument, None)

                logger.info(f"✅ Closed position for {instrument}, P&L: {total_pnl}")
                return {'instrument': instrument, 'pnl': total_pnl}

        except Exception as e:
            logger.error(f"❌ Failed to close position for {instrument}: {e}")

        return None

    async def get_open_positions(self, use_cache: bool = True) -> List[Position]:
        """
        Get all open positions with caching

        Args:
            use_cache: Whether to use cached data

        Returns:
            List of open positions
        """
        # Check if we should use cache
        if use_cache and self.position_cache:
            # Simple cache check - could add TTL if needed
            return list(self.position_cache.values())

        url = f"{self.rest_url}/v3/accounts/{self.account_id}/openPositions"

        try:
            async with self.session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()

                positions = []
                for pos in data.get('positions', []):
                    # Process long positions
                    if pos.get('long', {}).get('units', '0') != '0':
                        position = Position(
                            instrument=pos['instrument'],
                            units=int(pos['long']['units']),
                            avg_price=float(pos['long']['averagePrice']),
                            unrealized_pnl=float(pos['long'].get('unrealizedPL', 0)),
                            side='long'
                        )
                        positions.append(position)

                    # Process short positions
                    if pos.get('short', {}).get('units', '0') != '0':
                        position = Position(
                            instrument=pos['instrument'],
                            units=int(pos['short']['units']),
                            avg_price=float(pos['short']['averagePrice']),
                            unrealized_pnl=float(pos['short'].get('unrealizedPL', 0)),
                            side='short'
                        )
                        positions.append(position)

                # Update cache
                self.position_cache = {p.instrument: p for p in positions}
                return positions

        except Exception as e:
            logger.error(f"❌ Failed to get open positions: {e}")
            return []

    async def get_candles_async(
        self,
        instruments: List[str],
        granularity: str = 'M1',
        count: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get candles for multiple instruments concurrently

        Args:
            instruments: List of currency pairs
            granularity: Candle granularity
            count: Number of candles

        Returns:
            Dict mapping instrument to candle data
        """
        async def fetch_candles(instrument: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
            """Fetch candles for a single instrument"""
            url = f"{self.rest_url}/v3/instruments/{instrument}/candles"
            params = {
                'granularity': granularity,
                'count': count,
                'price': 'MBA'
            }

            try:
                async with self.session.get(url, headers=self.headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    candles = []
                    for c in data.get('candles', []):
                        candle = {
                            'time': c['time'],
                            'volume': int(c['volume']),
                            'mid_o': float(c['mid']['o']),
                            'mid_h': float(c['mid']['h']),
                            'mid_l': float(c['mid']['l']),
                            'mid_c': float(c['mid']['c']),
                            'bid_c': float(c['bid']['c']),
                            'ask_c': float(c['ask']['c'])
                        }
                        candles.append(candle)

                    return instrument, candles

            except Exception as e:
                logger.error(f"Failed to get candles for {instrument}: {e}")
                return instrument, None

        # Fetch all instruments concurrently
        tasks = [fetch_candles(inst) for inst in instruments]
        results = await asyncio.gather(*tasks)

        # Convert to dict
        return {inst: candles for inst, candles in results if candles is not None}

    async def execute_trades_batch(
        self,
        trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple trades concurrently

        Args:
            trades: List of trade specifications

        Returns:
            List of trade results
        """
        async def execute_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single trade"""
            result = await self.place_market_order(
                instrument=trade['instrument'],
                units=trade['units'],
                stop_loss=trade.get('stop_loss'),
                take_profit=trade.get('take_profit')
            )
            return result or {'status': 'failed', 'trade': trade}

        # Execute all trades concurrently
        results = await asyncio.gather(*[execute_trade(t) for t in trades])

        successful = sum(1 for r in results if r.get('order_id'))
        logger.info(f"✅ Batch execution: {successful}/{len(trades)} trades successful")

        return results


async def main():
    """Example usage of async OANDA client"""
    async with AsyncOandaClient(max_connections=10) as client:
        # Get account summary
        account = await client.get_account_summary()
        print(f"Account Balance: {account['balance']}")

        # Start price streaming for GBPJPY
        await client.start_price_stream('GBP_JPY')

        # Get latest price
        price = await client.get_latest_price('GBP_JPY')
        if price:
            print(f"GBP/JPY: Bid={price.bid}, Ask={price.ask}, Spread={price.spread}")

        # Get open positions
        positions = await client.get_open_positions()
        print(f"Open positions: {len(positions)}")

        # Get candles for multiple pairs concurrently
        candles = await client.get_candles_async(
            ['GBP_JPY', 'EUR_USD', 'USD_JPY'],
            granularity='M1',
            count=10
        )
        for inst, data in candles.items():
            print(f"{inst}: {len(data)} candles")

        # Example batch trade execution
        trades = [
            {'instrument': 'GBP_JPY', 'units': 100},
            {'instrument': 'EUR_USD', 'units': -100}
        ]
        # results = await client.execute_trades_batch(trades)


if __name__ == "__main__":
    asyncio.run(main())