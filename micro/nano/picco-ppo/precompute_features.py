#!/usr/bin/env python3
"""
Precompute all features for the entire dataset once.
This avoids recalculating indicators for every episode and losing rows to initialization.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Tuple

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
    """Map H1 features to corresponding M5 bars."""
    m5_features['h1_trend'] = 0.0
    m5_features['h1_momentum'] = 0.0

    for i in range(len(m5_features)):
        m5_bar = m5_features.iloc[i]['bar_index']

        # Find corresponding H1 bar
        h1_idx = h1_features[h1_features['bar_index'] <= m5_bar].index

        if len(h1_idx) > 0:
            h1_idx = h1_idx[-1]
            m5_features.loc[i, 'h1_trend'] = h1_features.loc[h1_idx, 'h1_trend']
            m5_features.loc[i, 'h1_momentum'] = h1_features.loc[h1_idx, 'h1_momentum']

    # Calculate use_mean_reversion flag
    m5_features['use_mean_reversion'] = (
        (m5_features['h1_trend'] == 0) &
        (m5_features['efficiency_ratio'] < 0.3)
    ).astype(float)

    return m5_features


def precompute_all_features(
    db_path: str,
    start_idx: int = 0,
    end_idx: int = 500000,
    output_path: str = 'precomputed_features.pkl'
) -> None:
    """Precompute all features for the entire dataset."""

    logger.info(f"Loading data from {db_path}, bars {start_idx} to {end_idx}")

    # Load M1 data
    conn = duckdb.connect(db_path, read_only=True)
    query = f"""
    SELECT bar_index, timestamp, open, high, low, close, volume
    FROM master
    WHERE bar_index BETWEEN {start_idx} AND {end_idx}
    ORDER BY bar_index
    """
    m1_data = pd.read_sql(query, conn)
    conn.close()

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
    m5_features = m5_features.fillna(method='ffill').fillna(0)

    # Save precomputed features
    logger.info(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'm5_features': m5_features,
            'h1_features': h1_features,
            'metadata': {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'total_m5_bars': len(m5_features),
                'total_h1_bars': len(h1_features),
                'feature_columns': list(m5_features.columns)
            }
        }, f)

    logger.info(f"Successfully precomputed {len(m5_features)} M5 bars with features")
    logger.info(f"Feature columns: {list(m5_features.columns)}")

    # Show sample of features
    logger.info("\nSample features (last 5 rows):")
    print(m5_features.tail())


if __name__ == "__main__":
    # Precompute features for training and evaluation data
    precompute_all_features(
        db_path="../../../data/EURUSD_M1_1993-2025.db",
        start_idx=0,
        end_idx=500000,  # Enough for both training and evaluation
        output_path="precomputed_features.pkl"
    )