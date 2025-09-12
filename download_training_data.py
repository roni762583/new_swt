#!/usr/bin/env python3
"""
Download sample training data for SWT system
Uses synthetic data for testing when OANDA credentials are not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_forex_data(
    instrument: str = "GBP_JPY",
    start_date: str = "2024-01-01",
    end_date: str = "2024-02-01",
    output_path: str = "data/GBPJPY_M1_sample.csv"
):
    """
    Generate synthetic forex data for testing
    """
    logger.info(f"Generating synthetic data for {instrument} from {start_date} to {end_date}")
    
    # Create date range (M1 = 1 minute bars)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate timestamps for market hours only (forex is 24/5)
    dates = pd.date_range(start=start, end=end, freq='1min')
    # Filter out weekends
    dates = dates[dates.dayofweek < 5]
    
    num_bars = len(dates)
    logger.info(f"Generating {num_bars} bars")
    
    # Generate realistic GBP/JPY price data around 190-200 range
    base_price = 195.0
    volatility = 0.05  # 50 pips volatility
    
    # Generate price series with realistic patterns
    returns = np.random.normal(0, volatility/100, num_bars)
    # Add some trend
    trend = np.linspace(0, 2, num_bars) / 1000  # Slight upward trend
    returns = returns + trend
    
    # Calculate prices
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high_prices = close_prices + np.abs(np.random.normal(0, 0.01, num_bars))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.01, num_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    # Generate volume (forex doesn't have real volume, but we simulate tick volume)
    volume = np.random.poisson(1000, num_bars)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'complete': True
    })
    
    # Ensure OHLC relationships are correct
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved {num_bars} bars to {output_path}")
    logger.info(f"   Price range: {df['low'].min():.4f} - {df['high'].max():.4f}")
    logger.info(f"   Date range: {df['time'].min()} - {df['time'].max()}")
    
    return df

def download_with_oanda(api_key: str, account_id: str, environment: str = 'practice'):
    """
    Download real data using OANDA API if credentials are available
    """
    try:
        from scripts.download_oanda_data import SWTOandaDownloader
        
        logger.info("Using OANDA API to download real data")
        downloader = SWTOandaDownloader(api_key, account_id, environment)
        
        # Download last 3 months of data for training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        success = downloader.download_historical_csv(
            instrument='GBP_JPY',
            start_date=start_date,
            end_date=end_date,
            output_path='data/GBPJPY_M1_training.csv',
            granularity='M1'
        )
        
        if success:
            logger.info("✅ Successfully downloaded OANDA data")
        return success
        
    except Exception as e:
        logger.error(f"Failed to download OANDA data: {e}")
        return False

def main():
    """Main entry point"""
    import os
    from dotenv import load_dotenv
    
    # Try to load .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    
    # Check for OANDA credentials
    api_key = os.getenv('OANDA_API_KEY')
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    environment = os.getenv('OANDA_ENVIRONMENT', 'practice')
    
    if api_key and account_id and api_key != 'your_oanda_api_key_here':
        # Try to download real data
        if download_with_oanda(api_key, account_id, environment):
            return
        else:
            logger.warning("Failed to download OANDA data, falling back to synthetic data")
    else:
        logger.info("No OANDA credentials found, generating synthetic data")
    
    # Generate synthetic data for testing
    generate_synthetic_forex_data(
        instrument="GBP_JPY",
        start_date="2024-01-01",
        end_date="2024-02-01",
        output_path="data/GBPJPY_M1_training.csv"
    )
    
    # Also generate validation data
    generate_synthetic_forex_data(
        instrument="GBP_JPY",
        start_date="2024-02-01",
        end_date="2024-02-15",
        output_path="data/GBPJPY_M1_validation.csv"
    )
    
    logger.info("✅ Data pipeline ready for training")

if __name__ == "__main__":
    main()