#!/usr/bin/env python3
"""
OANDA v20 API Client for Real Trading
Production-ready implementation for live market data and order execution
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import time
import requests
from decimal import Decimal
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class OandaV20Client:
    """
    Real OANDA v20 API client for production trading
    Handles streaming prices, order execution, and account management
    """

    def __init__(self):
        """Initialize OANDA client with credentials from environment"""
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

        # Price streaming
        self.price_stream = None
        self.price_queue = Queue(maxsize=100)
        self.streaming = False

        # Current positions and account info
        self.positions = {}
        self.account_balance = 0.0
        self.unrealized_pnl = 0.0

        logger.info(f"🏦 OANDA Client initialized: {self.environment} environment")
        logger.info(f"   Account: {self.account_id}")

    def get_account_summary(self) -> Dict[str, Any]:
        """Get real account summary from OANDA"""
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/summary"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            account = data['account']
            self.account_balance = float(account['balance'])
            self.unrealized_pnl = float(account.get('unrealizedPL', 0))

            return {
                'balance': self.account_balance,
                'unrealized_pnl': self.unrealized_pnl,
                'nav': float(account['NAV']),
                'margin_used': float(account.get('marginUsed', 0)),
                'margin_available': float(account.get('marginAvailable', 0)),
                'open_position_count': int(account.get('openPositionCount', 0)),
                'open_trade_count': int(account.get('openTradeCount', 0))
            }

        except Exception as e:
            logger.error(f"❌ Failed to get account summary: {e}")
            raise

    def start_price_streaming(self, instruments: List[str]):
        """Start streaming real-time prices from OANDA"""
        if self.streaming:
            logger.warning("Price streaming already active")
            return

        # Format instruments for API
        instruments_str = ','.join(instruments)
        url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
        params = {'instruments': instruments_str}

        def stream_worker():
            """Worker thread for price streaming"""
            try:
                logger.info(f"📊 Starting price stream for: {instruments_str}")
                response = requests.get(url, headers=self.headers, params=params, stream=True)
                response.raise_for_status()

                self.streaming = True

                for line in response.iter_lines():
                    if not self.streaming:
                        break

                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))

                            if data['type'] == 'PRICE':
                                # Process price update
                                price_data = {
                                    'instrument': data['instrument'],
                                    'time': data['time'],
                                    'bid': float(data['bids'][0]['price']) if data.get('bids') else None,
                                    'ask': float(data['asks'][0]['price']) if data.get('asks') else None,
                                    'spread': None
                                }

                                if price_data['bid'] and price_data['ask']:
                                    price_data['spread'] = price_data['ask'] - price_data['bid']

                                # Add to queue (drop old prices if full)
                                if self.price_queue.full():
                                    try:
                                        self.price_queue.get_nowait()
                                    except:
                                        pass

                                self.price_queue.put(price_data)

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse price data: {line}")
                        except Exception as e:
                            logger.error(f"Error processing price: {e}")

            except Exception as e:
                logger.error(f"❌ Price streaming error: {e}")
                self.streaming = False

        # Start streaming in background thread
        self.price_stream = threading.Thread(target=stream_worker, daemon=True)
        self.price_stream.start()

    def stop_price_streaming(self):
        """Stop price streaming"""
        self.streaming = False
        if self.price_stream:
            self.price_stream.join(timeout=2)
        logger.info("📊 Price streaming stopped")

    def get_latest_price(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get the latest price for an instrument"""
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/pricing"
        params = {'instruments': instrument}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if data['prices']:
                price = data['prices'][0]
                return {
                    'instrument': price['instrument'],
                    'time': price['time'],
                    'bid': float(price['bids'][0]['price']) if price.get('bids') else None,
                    'ask': float(price['asks'][0]['price']) if price.get('asks') else None,
                    'spread': float(price['asks'][0]['price']) - float(price['bids'][0]['price'])
                             if price.get('asks') and price.get('bids') else None
                }

        except Exception as e:
            logger.error(f"❌ Failed to get price for {instrument}: {e}")

        return None

    def place_market_order(self, instrument: str, units: int,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Place a real market order on OANDA

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
                'timeInForce': 'FOK',  # Fill or Kill
                'positionFill': 'DEFAULT'
            }
        }

        # Add stop loss if specified
        if stop_loss:
            order['order']['stopLossOnFill'] = {
                'price': str(stop_loss),
                'timeInForce': 'GTC'
            }

        # Add take profit if specified
        if take_profit:
            order['order']['takeProfitOnFill'] = {
                'price': str(take_profit),
                'timeInForce': 'GTC'
            }

        try:
            response = requests.post(url, headers=self.headers, json=order)
            response.raise_for_status()
            result = response.json()

            if 'orderFillTransaction' in result:
                fill = result['orderFillTransaction']
                logger.info(f"✅ Order filled: {units} units of {instrument} @ {fill.get('price')}")
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

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Order failed: {e.response.text}")
        except Exception as e:
            logger.error(f"❌ Order execution error: {e}")

        return None

    def close_position(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Close all positions for an instrument"""
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/positions/{instrument}/close"

        data = {
            'longUnits': 'ALL',
            'shortUnits': 'ALL'
        }

        try:
            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()

            total_pnl = 0.0
            if 'longOrderFillTransaction' in result:
                total_pnl += float(result['longOrderFillTransaction'].get('pl', 0))
            if 'shortOrderFillTransaction' in result:
                total_pnl += float(result['shortOrderFillTransaction'].get('pl', 0))

            logger.info(f"✅ Closed position for {instrument}, P&L: {total_pnl}")
            return {'instrument': instrument, 'pnl': total_pnl}

        except Exception as e:
            logger.error(f"❌ Failed to close position for {instrument}: {e}")

        return None

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        url = f"{self.rest_url}/v3/accounts/{self.account_id}/openPositions"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            positions = []
            for pos in data.get('positions', []):
                position = {
                    'instrument': pos['instrument'],
                    'long_units': int(pos.get('long', {}).get('units', 0)),
                    'long_avg_price': float(pos.get('long', {}).get('averagePrice', 0)),
                    'long_pnl': float(pos.get('long', {}).get('unrealizedPL', 0)),
                    'short_units': int(pos.get('short', {}).get('units', 0)),
                    'short_avg_price': float(pos.get('short', {}).get('averagePrice', 0)),
                    'short_pnl': float(pos.get('short', {}).get('unrealizedPL', 0)),
                    'total_pnl': float(pos.get('unrealizedPL', 0))
                }
                positions.append(position)

            self.positions = {p['instrument']: p for p in positions}
            return positions

        except Exception as e:
            logger.error(f"❌ Failed to get open positions: {e}")
            return []

    def get_candles(self, instrument: str, granularity: str = 'M1',
                    count: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Get historical candles for backtesting or analysis"""
        url = f"{self.rest_url}/v3/instruments/{instrument}/candles"

        params = {
            'granularity': granularity,
            'count': count,
            'price': 'MBA'  # Mid, Bid, Ask
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

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

            return candles

        except Exception as e:
            logger.error(f"❌ Failed to get candles for {instrument}: {e}")
            return None