#!/usr/bin/env python3
"""
Create synthetic GBPJPY data for testing validation framework
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_synthetic_gbpjpy_data():
    """
    Create synthetic GBPJPY data matching expected format
    """
    print("📊 Creating synthetic GBPJPY data for testing...")
    
    # Create date range from 2022-01 to 2025-08 (M1 bars)
    start_date = datetime(2025, 7, 1)  # Last 2 months for testing
    end_date = datetime(2025, 8, 31, 23, 59)
    
    # Generate timestamps (M1 = 1 minute bars, forex is 24/5)
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    # Filter out weekends (Saturday and Sunday)
    dates = dates[~dates.to_series().dt.dayofweek.isin([5, 6])]
    
    num_bars = len(dates)
    print(f"   Generating {num_bars:,} bars")
    
    # Generate realistic GBPJPY price data
    base_price = 195.0  # GBPJPY typical range
    volatility = 0.001  # 0.1% volatility per minute
    trend = 0.00001  # Slight upward trend
    
    # Generate returns with realistic patterns
    np.random.seed(13475)  # Use Episode 13475 as seed for reproducibility
    returns = np.random.normal(trend, volatility, num_bars)
    
    # Add some autocorrelation for realism
    for i in range(1, len(returns)):
        returns[i] = 0.8 * returns[i] + 0.2 * returns[i-1]
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from base prices
    open_prices = prices
    high_prices = prices + np.abs(np.random.normal(0, 0.05, num_bars))  # 5 pips variation
    low_prices = prices - np.abs(np.random.normal(0, 0.05, num_bars))
    close_prices = prices + np.random.normal(0, 0.02, num_bars)
    
    # Ensure OHLC relationships are correct
    for i in range(num_bars):
        high_prices[i] = max(open_prices[i], high_prices[i], close_prices[i])
        low_prices[i] = min(open_prices[i], low_prices[i], close_prices[i])
    
    # Generate volume (tick volume for forex)
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
    
    # Save to expected location
    output_path = Path('data/GBPJPY_M1_202201-202508.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Created synthetic data: {output_path}")
    print(f"   Bars: {num_bars:,}")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"   Price range: {df['low'].min():.4f} - {df['high'].max():.4f}")
    
    return df

if __name__ == "__main__":
    create_synthetic_gbpjpy_data()