#!/usr/bin/env python3
"""
Precompute all features and store them in DuckDB for efficient slicing.
This avoids recalculating indicators for every episode and losing rows to initialization.
"""

import duckdb
import pandas as pd
import numpy as np
import logging
from typing import Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def aggregate_bars(data: pd.DataFrame, period: int) -> pd.DataFrame:
    """Aggregate M1 bars to higher timeframe."""
    data['group'] = data['bar_index'] // period

    agg = data.groupby('group').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'bar_index': 'first',
        'timestamp': 'first'
    }).reset_index(drop=True)

    return agg


def compute_m5_features(m5_data: pd.DataFrame) -> pd.DataFrame:
    """Compute all M5 technical features."""
    df = m5_data.copy()

    # Basic SMAs
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma200'] = df['close'].rolling(200).mean()

    # ATR for normalization
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # 1. React Ratio: (close - sma200) / (sma20 - sma200)
    df['reactive'] = df['close'] - df['sma200']
    df['lessreactive'] = df['sma20'] - df['sma200']
    df['react_ratio'] = df['reactive'] / (df['lessreactive'] + 0.0001)
    df['react_ratio'] = df['react_ratio'].clip(-5, 5)

    # 2. Efficiency Ratio (Kaufman's)
    direction = (df['close'] - df['close'].shift(10)).abs()
    volatility = df['close'].diff().abs().rolling(10).sum()
    df['efficiency_ratio'] = direction / (volatility + 0.0001)

    # 3. Bollinger Band Position
    bb_std = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - df['sma20']) / (bb_std * 2 + 0.0001)
    df['bb_position'] = df['bb_position'].clip(-1, 1)

    # 4. RSI Extreme
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_extreme'] = (df['rsi'] - 50) / 50

    # 5. ATR Ratio (volatility regime)
    df['atr_ratio'] = df['atr'] / (df['sma20'] + 0.0001)

    return df


def compute_h1_features(h1_data: pd.DataFrame) -> pd.DataFrame:
    """Compute all H1 features."""
    h1 = h1_data.copy()
    h1['h1_sma20'] = h1['close'].rolling(20).mean()

    # H1 Trend (-1, 0, 1)
    h1['h1_trend'] = 0
    h1.loc[h1['close'] > h1['h1_sma20'] * 1.001, 'h1_trend'] = 1
    h1.loc[h1['close'] < h1['h1_sma20'] * 0.999, 'h1_trend'] = -1

    # H1 Momentum (5-bar ROC)
    h1['h1_momentum'] = h1['close'].pct_change(5).fillna(0)

    return h1


def map_h1_to_m5(m5_features: pd.DataFrame, h1_features: pd.DataFrame) -> pd.DataFrame:
    """Map H1 features to corresponding M5 bars - optimized version."""
    # Use merge_asof for efficient time-series alignment
    m5_features = pd.merge_asof(
        m5_features.sort_values('bar_index'),
        h1_features[['bar_index', 'h1_trend', 'h1_momentum']].sort_values('bar_index'),
        on='bar_index',
        direction='backward',
        suffixes=('', '_h1')
    )

    # Fill any NaN values from the merge
    m5_features['h1_trend'] = m5_features['h1_trend'].fillna(0.0)
    m5_features['h1_momentum'] = m5_features['h1_momentum'].fillna(0.0)

    # Calculate use_mean_reversion flag
    m5_features['use_mean_reversion'] = (
        (m5_features['h1_trend'] == 0) &
        (m5_features['efficiency_ratio'] < 0.3)
    ).astype(float)

    return m5_features


def precompute_all_features(
    source_db_path: str,
    target_db_path: str,
    start_idx: int = 0,
    end_idx: int = 500000
) -> None:
    """Precompute all features and store in DuckDB."""

    logger.info(f"Loading data from {source_db_path}, bars {start_idx} to {end_idx}")

    # Load M1 data
    conn_source = duckdb.connect(source_db_path, read_only=True)
    query = f"""
    SELECT bar_index, timestamp, open, high, low, close, volume
    FROM master
    WHERE bar_index BETWEEN {start_idx} AND {end_idx}
    ORDER BY bar_index
    """
    m1_data = pd.read_sql(query, conn_source)
    conn_source.close()

    logger.info(f"Loaded {len(m1_data)} M1 bars")

    # Aggregate to M5 and H1
    logger.info("Aggregating to M5 and H1...")
    m5_data = aggregate_bars(m1_data, 5)
    h1_data = aggregate_bars(m1_data, 60)

    logger.info(f"Created {len(m5_data)} M5 bars and {len(h1_data)} H1 bars")

    # Compute features
    logger.info("Computing M5 features...")
    m5_features = compute_m5_features(m5_data)

    logger.info("Computing H1 features...")
    h1_features = compute_h1_features(h1_data)

    logger.info("Mapping H1 to M5...")
    m5_features = map_h1_to_m5(m5_features, h1_features)

    # Fill NaN values (from indicator initialization)
    logger.info("Cleaning up NaN values...")
    m5_features = m5_features.ffill().fillna(0)

    # Create or connect to target database
    logger.info(f"Saving to {target_db_path}...")
    conn_target = duckdb.connect(target_db_path)

    # Create table for precomputed features
    conn_target.execute("DROP TABLE IF EXISTS m5_features")

    # Register the DataFrame as a view and create table from it
    conn_target.register('m5_features_df', m5_features)
    conn_target.execute("""
        CREATE TABLE m5_features AS
        SELECT * FROM m5_features_df
    """)

    # Create index for efficient querying
    conn_target.execute("CREATE INDEX idx_bar_index ON m5_features(bar_index)")

    # Verify the data
    count = conn_target.execute("SELECT COUNT(*) FROM m5_features").fetchone()[0]
    columns = conn_target.execute("PRAGMA table_info(m5_features)").fetchall()

    conn_target.close()

    logger.info(f"Successfully saved {count} M5 bars with features")
    logger.info(f"Feature columns: {[col[1] for col in columns]}")

    # Show sample of features
    logger.info("\nSample features (last 5 rows):")
    print(m5_features.tail())


if __name__ == "__main__":
    # Check if master database exists
    if os.path.exists("master.duckdb"):
        # Local development - master.duckdb in current directory
        source_db = "master.duckdb"
    elif not os.path.exists("/app/data/master.duckdb"):
        # Try old location for compatibility
        source_db = "../../../data/master.duckdb"
        if not os.path.exists(source_db):
            # Create sample data for testing
            logger.warning("Master database not found, creating sample data...")
            conn = duckdb.connect("master.duckdb")

            # Create sample M1 data
            dates = pd.date_range('2020-01-01', periods=500000, freq='1min')
            prices = 1.1000 + np.cumsum(np.random.randn(500000) * 0.0001)

            sample_data = pd.DataFrame({
                'bar_index': range(500000),
                'timestamp': dates,
                'open': prices,
                'high': prices + abs(np.random.randn(500000) * 0.0002),
                'low': prices - abs(np.random.randn(500000) * 0.0002),
                'close': prices + np.random.randn(500000) * 0.0001,
                'volume': np.random.randint(100, 1000, 500000)
            })

            conn.execute("CREATE TABLE master AS SELECT * FROM sample_data", {"sample_data": sample_data})
            conn.close()
            source_db = "master.duckdb"
    else:
        source_db = "/app/data/master.duckdb"

    # Precompute features for training and evaluation data
    precompute_all_features(
        source_db_path=source_db,
        target_db_path="precomputed_features.duckdb",
        start_idx=0,
        end_idx=500000  # Enough for both training and evaluation
    )