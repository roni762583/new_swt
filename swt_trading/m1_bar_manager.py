#!/usr/bin/env python3
"""
M1 Bar Manager for SWT Live Trading
Maintains rolling window of 256 M1 bars for WST feature extraction
"""

import os
import logging
from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import threading
import time

logger = logging.getLogger(__name__)

class M1BarManager:
    """
    Manages M1 bar collection and maintains 256-bar rolling window
    Pulls latest completed M1 bar from OANDA every minute
    """

    def __init__(self, instrument: str = 'GBP_JPY', window_size: int = 256):
        """
        Initialize M1 bar manager

        Args:
            instrument: Trading instrument (default: GBP_JPY)
            window_size: Number of bars to maintain (default: 256 for WST)
        """
        self.instrument = instrument
        self.window_size = window_size

        # Rolling window of M1 bars
        self.price_window: Deque[Dict[str, Any]] = deque(maxlen=window_size)

        # Thread safety
        self.lock = threading.Lock()

        # Bar collection thread
        self.collection_thread = None
        self.collecting = False

        # Last bar timestamp to avoid duplicates
        self.last_bar_time = None

        # Initialize OANDA client
        from swt_trading.oanda_client import OandaV20Client
        self.oanda_client = OandaV20Client()

        logger.info(f"📊 M1 Bar Manager initialized for {instrument}")
        logger.info(f"   Window size: {window_size} bars")

    def initialize_history(self) -> bool:
        """
        Initialize with historical M1 bars from OANDA
        Fetches last 256 bars to fill the window
        """
        try:
            logger.info(f"📈 Fetching initial {self.window_size} M1 bars...")

            # Get historical candles from OANDA
            candles = self.oanda_client.get_candles(
                instrument=self.instrument,
                granularity='M1',
                count=self.window_size
            )

            if not candles or len(candles) < self.window_size:
                logger.error(f"❌ Failed to get {self.window_size} bars, got {len(candles) if candles else 0}")
                return False

            # Process candles into our format
            with self.lock:
                self.price_window.clear()
                for candle in candles:
                    bar = {
                        'timestamp': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                        'open': candle['mid_o'],
                        'high': candle['mid_h'],
                        'low': candle['mid_l'],
                        'close': candle['mid_c'],
                        'volume': candle['volume'],
                        'bid_close': candle['bid_c'],
                        'ask_close': candle['ask_c'],
                        'spread': candle['ask_c'] - candle['bid_c']
                    }
                    self.price_window.append(bar)

                # Set last bar time
                self.last_bar_time = self.price_window[-1]['timestamp']

            logger.info(f"✅ Initialized with {len(self.price_window)} M1 bars")
            logger.info(f"   Latest bar: {self.last_bar_time}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize history: {e}")
            return False

    def start_collection(self):
        """Start automatic M1 bar collection every minute"""
        if self.collecting:
            logger.warning("Bar collection already running")
            return

        def collection_worker():
            """Worker thread that pulls new M1 bars"""
            logger.info("🏃 Starting M1 bar collection thread")
            self.collecting = True

            while self.collecting:
                try:
                    # Calculate seconds until next minute
                    now = datetime.now()
                    seconds_to_wait = 60 - now.second + 5  # Wait 5 seconds after minute change

                    if seconds_to_wait > 0 and seconds_to_wait <= 65:
                        time.sleep(seconds_to_wait)

                    # Pull latest bars
                    self._fetch_latest_bars()

                except Exception as e:
                    logger.error(f"❌ Bar collection error: {e}")
                    time.sleep(10)  # Wait before retry

        self.collection_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collection_thread.start()
        logger.info("✅ M1 bar collection started")

    def stop_collection(self):
        """Stop automatic bar collection"""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("⏹️ M1 bar collection stopped")

    def _fetch_latest_bars(self):
        """Fetch latest completed M1 bars from OANDA"""
        try:
            # Get last few bars to ensure we don't miss any
            candles = self.oanda_client.get_candles(
                instrument=self.instrument,
                granularity='M1',
                count=5  # Get last 5 bars
            )

            if not candles:
                logger.warning("No candles received from OANDA")
                return

            with self.lock:
                new_bars_added = 0

                for candle in candles:
                    bar_time = datetime.fromisoformat(candle['time'].replace('Z', '+00:00'))

                    # Skip if we already have this bar
                    if self.last_bar_time and bar_time <= self.last_bar_time:
                        continue

                    # Add new bar
                    bar = {
                        'timestamp': bar_time,
                        'open': candle['mid_o'],
                        'high': candle['mid_h'],
                        'low': candle['mid_l'],
                        'close': candle['mid_c'],
                        'volume': candle['volume'],
                        'bid_close': candle['bid_c'],
                        'ask_close': candle['ask_c'],
                        'spread': candle['ask_c'] - candle['bid_c']
                    }

                    self.price_window.append(bar)
                    self.last_bar_time = bar_time
                    new_bars_added += 1

                    logger.debug(f"📊 New M1 bar: {bar_time} Close={bar['close']:.3f}")

                if new_bars_added > 0:
                    logger.info(f"📈 Added {new_bars_added} new M1 bar(s), window size: {len(self.price_window)}")

        except Exception as e:
            logger.error(f"❌ Failed to fetch latest bars: {e}")

    def get_price_series(self) -> Optional[np.ndarray]:
        """
        Get price series for WST processing
        Returns numpy array of close prices
        """
        with self.lock:
            if len(self.price_window) < self.window_size:
                logger.warning(f"Insufficient bars: {len(self.price_window)}/{self.window_size}")
                return None

            # Extract close prices
            prices = np.array([bar['close'] for bar in self.price_window])
            return prices

    def get_ohlcv_data(self) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data as pandas DataFrame
        Useful for technical indicators
        """
        with self.lock:
            if len(self.price_window) < self.window_size:
                return None

            df = pd.DataFrame(list(self.price_window))
            df.set_index('timestamp', inplace=True)
            return df

    def get_latest_bar(self) -> Optional[Dict[str, Any]]:
        """Get the most recent M1 bar"""
        with self.lock:
            if self.price_window:
                return self.price_window[-1].copy()
            return None

    def get_current_price(self) -> Optional[float]:
        """Get current price (close of latest bar)"""
        latest = self.get_latest_bar()
        return latest['close'] if latest else None

    def get_current_spread(self) -> Optional[float]:
        """Get current spread in pips"""
        latest = self.get_latest_bar()
        if latest:
            # For JPY pairs, 1 pip = 0.01
            return latest['spread'] * 100  # Convert to pips
        return None

    def is_ready(self) -> bool:
        """Check if we have enough bars for processing"""
        with self.lock:
            return len(self.price_window) >= self.window_size