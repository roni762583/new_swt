#!/usr/bin/env python3
"""
OANDA M1 Incremental Data Puller
Fetches live M1 candle data from OANDA v20 API for real-time feature updates
"""

import os
import json
import logging
import requests
from typing import List, Optional
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OandaM1Puller:
    """
    OANDA M1 candle data puller using official v20 API
    Designed for real-time feature vector updates
    """

    def __init__(self):
        """Initialize OANDA API client with credentials from .env"""
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.environment = os.getenv('OANDA_ENVIRONMENT', 'practice')

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set in .env file")

        # Set base URL based on environment
        if self.environment == 'live':
            self.api_url = "https://api-fxtrade.oanda.com"
        else:
            self.api_url = "https://api-fxpractice.oanda.com"

        # API headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }

        logger.info(f"OANDA M1 Puller initialized ({self.environment} environment)")

    def get_last_256_m1_closes(self, instrument: str = 'GBP_JPY') -> List[float]:
        """
        Fetch the last 256 M1 bar close prices from OANDA

        Args:
            instrument: Trading instrument (default: GBP_JPY)

        Returns:
            List of 256 close prices (oldest to newest)
        """
        # OANDA uses underscore format: GBP_JPY not GBPJPY
        formatted_instrument = instrument.replace('/', '_')
        if 'GBP' in instrument and 'JPY' in instrument and '_' not in instrument:
            formatted_instrument = 'GBP_JPY'

        endpoint = f"{self.api_url}/v3/instruments/{formatted_instrument}/candles"

        params = {
            'granularity': 'M1',
            'count': 257,  # Get 257 to ensure we have 256 complete candles
            'price': 'M'   # Mid prices
        }

        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            candles = data.get('candles', [])

            # Extract close prices from complete candles only
            close_prices = []
            for candle in candles:
                if candle.get('complete', False):  # Only use complete candles
                    close_price = float(candle['mid']['c'])
                    close_prices.append(close_price)

            # Ensure we have exactly 256 prices
            if len(close_prices) >= 256:
                close_prices = close_prices[-256:]  # Take last 256
            else:
                logger.warning(f"Only got {len(close_prices)} complete candles, expected 256")

            logger.info(f"Fetched {len(close_prices)} M1 close prices for {formatted_instrument}")
            logger.info(f"Price range: {min(close_prices):.5f} - {max(close_prices):.5f}")

            return close_prices

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed - check OANDA_API_KEY")
            elif e.response.status_code == 404:
                logger.error(f"Instrument {formatted_instrument} not found")
            else:
                logger.error(f"HTTP error: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to fetch M1 candles: {e}")
            raise

    def get_latest_m1_close(self, instrument: str = 'GBP_JPY') -> List[float]:
        """
        Fetch the latest complete M1 bar close price

        Args:
            instrument: Trading instrument (default: GBP_JPY)

        Returns:
            List with single close price of the most recent complete M1 bar
        """
        # OANDA uses underscore format
        formatted_instrument = instrument.replace('/', '_')
        if 'GBP' in instrument and 'JPY' in instrument and '_' not in instrument:
            formatted_instrument = 'GBP_JPY'

        endpoint = f"{self.api_url}/v3/instruments/{formatted_instrument}/candles"

        params = {
            'granularity': 'M1',
            'count': 2,  # Get last 2 to ensure we have 1 complete
            'price': 'M'  # Mid prices
        }

        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            candles = data.get('candles', [])

            # Find the most recent complete candle
            for candle in reversed(candles):
                if candle.get('complete', False):
                    close_price = float(candle['mid']['c'])
                    timestamp = candle['time']
                    logger.info(f"Latest M1 close for {formatted_instrument}: {close_price:.5f} at {timestamp}")
                    return [close_price]

            logger.warning("No complete M1 candles found")
            return []

        except Exception as e:
            logger.error(f"Failed to fetch latest M1 candle: {e}")
            raise

    def update_256_queue(self, existing_256: List[float], new_m1_close: List[float]) -> List[float]:
        """
        Update the 256-element queue by dropping oldest and adding newest

        Args:
            existing_256: Current list of 256 close prices
            new_m1_close: List containing the new M1 close price

        Returns:
            Updated list of 256 close prices (oldest dropped, newest added)
        """
        if len(existing_256) != 256:
            logger.warning(f"Expected 256 prices, got {len(existing_256)}")

        if not new_m1_close:
            logger.warning("No new M1 close price provided")
            return existing_256

        # Use deque for efficient queue operations
        queue = deque(existing_256, maxlen=256)

        # Add new price (automatically drops oldest due to maxlen)
        queue.append(new_m1_close[0])

        updated_list = list(queue)

        logger.info(f"Updated queue: dropped {existing_256[0]:.5f}, added {new_m1_close[0]:.5f}")
        logger.debug(f"Queue size: {len(updated_list)}, range: {min(updated_list):.5f} - {max(updated_list):.5f}")

        return updated_list


def test_oanda_connection():
    """Test function to verify OANDA connection and data retrieval"""
    puller = OandaM1Puller()

    print("\n" + "="*60)
    print("TESTING OANDA M1 DATA PULLER")
    print("="*60)

    # Test 1: Get last 256 M1 closes
    print("\n1. Fetching last 256 M1 closes...")
    try:
        closes_256 = puller.get_last_256_m1_closes()
        print(f"   ✓ Retrieved {len(closes_256)} close prices")
        print(f"   First 5: {closes_256[:5]}")
        print(f"   Last 5: {closes_256[-5:]}")
        print(f"   Range: {min(closes_256):.5f} - {max(closes_256):.5f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Test 2: Get latest M1 close
    print("\n2. Fetching latest M1 close...")
    try:
        latest = puller.get_latest_m1_close()
        print(f"   ✓ Latest close: {latest[0]:.5f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Test 3: Update queue
    print("\n3. Testing queue update...")
    try:
        updated_queue = puller.update_256_queue(closes_256, latest)
        print(f"   ✓ Queue updated: {len(updated_queue)} prices")
        print(f"   Old first: {closes_256[0]:.5f}")
        print(f"   New first: {updated_queue[0]:.5f}")
        print(f"   Old last: {closes_256[-1]:.5f}")
        print(f"   New last: {updated_queue[-1]:.5f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_oanda_connection()