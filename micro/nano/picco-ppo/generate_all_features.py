#!/usr/bin/env python3
"""
MASTER FEATURE GENERATION SCRIPT
Consolidates all feature generation for master.duckdb table.

Execution order:
1. Swing point detection (M1 and H1)
2. Last swing tracking (indices + prices)
3. H1 swing range position
4. Swing point range (H1 high - H1 low)
5. Z-score features with Window=20

All features generated from OHLCV data in correct dependency order.
"""

import numpy as np
import pandas as pd
import duckdb
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
CONFIG_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/feature_zscore_config.json")


# ============================================================================
# STEP 1: SWING POINT DETECTION
# ============================================================================

def detect_swings_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect M1 swing highs and lows using vectorized operations.
    Pattern: h[i] > h[i-1] and h[i] > h[i+1]
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    swing_high_m1 = np.zeros(n, dtype=bool)
    swing_low_m1 = np.zeros(n, dtype=bool)

    # Vectorized detection (bars 1 to n-2)
    swing_high_m1[1:-1] = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
    swing_low_m1[1:-1] = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])

    df['swing_high_m1'] = swing_high_m1
    df['swing_low_m1'] = swing_low_m1

    return df


def detect_h1_swings_from_m1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect H1 swings: highest M1 swing high and lowest M1 swing low per hour.
    """
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('h')
    df['swing_high_h1'] = False
    df['swing_low_h1'] = False

    for hour, group in df.groupby('hour'):
        m1_highs = group[group['swing_high_m1'] == True]
        m1_lows = group[group['swing_low_m1'] == True]

        if len(m1_highs) > 0:
            max_high_idx = m1_highs['high'].idxmax()
            df.loc[max_high_idx, 'swing_high_h1'] = True

        if len(m1_lows) > 0:
            min_low_idx = m1_lows['low'].idxmin()
            df.loc[min_low_idx, 'swing_low_h1'] = True

    df = df.drop('hour', axis=1)
    return df


def generate_swing_points(conn: duckdb.DuckDBPyConnection):
    """Step 1: Generate swing point detection columns."""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: SWING POINT DETECTION")
    logger.info("="*80)

    # Fetch data
    logger.info("Fetching OHLCV data...")
    df = conn.execute("""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        ORDER BY bar_index
    """).fetch_df()
    logger.info(f"Loaded {len(df):,} rows")

    # Detect M1 swings
    logger.info("Detecting M1 swing points...")
    df = detect_swings_vectorized(df)
    logger.info(f"  Found {df['swing_high_m1'].sum():,} M1 swing highs")
    logger.info(f"  Found {df['swing_low_m1'].sum():,} M1 swing lows")

    # Detect H1 swings
    logger.info("Detecting H1 swing points...")
    df = detect_h1_swings_from_m1(df)
    logger.info(f"  Found {df['swing_high_h1'].sum():,} H1 swing highs")
    logger.info(f"  Found {df['swing_low_h1'].sum():,} H1 swing lows")

    # Update database
    logger.info("Updating database...")
    conn.register('swing_data', df[['bar_index', 'swing_high_m1', 'swing_low_m1',
                                     'swing_high_h1', 'swing_low_h1']])
    conn.execute("""
        UPDATE master
        SET
            swing_high_m1 = swing_data.swing_high_m1,
            swing_low_m1 = swing_data.swing_low_m1,
            swing_high_h1 = swing_data.swing_high_h1,
            swing_low_h1 = swing_data.swing_low_h1
        FROM swing_data
        WHERE master.bar_index = swing_data.bar_index
    """)
    logger.info("✅ Swing points updated")

    return df


# ============================================================================
# STEP 2: LAST SWING TRACKING
# ============================================================================

def generate_last_swing_tracking(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 2: Generate last AND prev swing tracking (indices + values for 16 columns total)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LAST & PREV SWING TRACKING (16 columns)")
    logger.info("="*80)

    n = len(df)

    # Initialize tracking arrays (last = most recent, prev = previous to last)
    last_m1_hsp_idx = np.full(n, -1, dtype=np.int64)
    last_m1_hsp_val = np.full(n, np.nan)
    prev_m1_hsp_idx = np.full(n, -1, dtype=np.int64)
    prev_m1_hsp_val = np.full(n, np.nan)

    last_m1_lsp_idx = np.full(n, -1, dtype=np.int64)
    last_m1_lsp_val = np.full(n, np.nan)
    prev_m1_lsp_idx = np.full(n, -1, dtype=np.int64)
    prev_m1_lsp_val = np.full(n, np.nan)

    last_h1_hsp_idx = np.full(n, -1, dtype=np.int64)
    last_h1_hsp_val = np.full(n, np.nan)
    prev_h1_hsp_idx = np.full(n, -1, dtype=np.int64)
    prev_h1_hsp_val = np.full(n, np.nan)

    last_h1_lsp_idx = np.full(n, -1, dtype=np.int64)
    last_h1_lsp_val = np.full(n, np.nan)
    prev_h1_lsp_idx = np.full(n, -1, dtype=np.int64)
    prev_h1_lsp_val = np.full(n, np.nan)

    # Track M1 swings (last and prev)
    logger.info("Tracking M1 swings (last and prev)...")
    last_high_idx = -1
    last_high_price = np.nan
    prev_high_idx = -1
    prev_high_price = np.nan

    last_low_idx = -1
    last_low_price = np.nan
    prev_low_idx = -1
    prev_low_price = np.nan

    for i in range(n):
        if df.iloc[i]['swing_high_m1']:
            # New high swing: prev becomes last, last becomes new
            prev_high_idx = last_high_idx
            prev_high_price = last_high_price
            last_high_idx = df.iloc[i]['bar_index']
            last_high_price = df.iloc[i]['high']

        if df.iloc[i]['swing_low_m1']:
            # New low swing: prev becomes last, last becomes new
            prev_low_idx = last_low_idx
            prev_low_price = last_low_price
            last_low_idx = df.iloc[i]['bar_index']
            last_low_price = df.iloc[i]['low']

        last_m1_hsp_idx[i] = last_high_idx
        last_m1_hsp_val[i] = last_high_price
        prev_m1_hsp_idx[i] = prev_high_idx
        prev_m1_hsp_val[i] = prev_high_price

        last_m1_lsp_idx[i] = last_low_idx
        last_m1_lsp_val[i] = last_low_price
        prev_m1_lsp_idx[i] = prev_low_idx
        prev_m1_lsp_val[i] = prev_low_price

        if (i + 1) % 100000 == 0:
            logger.info(f"  M1: Processed {i+1:,}/{n:,} rows")

    # Track H1 swings (last and prev)
    logger.info("Tracking H1 swings (last and prev)...")
    last_high_idx = -1
    last_high_price = np.nan
    prev_high_idx = -1
    prev_high_price = np.nan

    last_low_idx = -1
    last_low_price = np.nan
    prev_low_idx = -1
    prev_low_price = np.nan

    for i in range(n):
        if df.iloc[i]['swing_high_h1']:
            # New high swing: prev becomes last, last becomes new
            prev_high_idx = last_high_idx
            prev_high_price = last_high_price
            last_high_idx = df.iloc[i]['bar_index']
            last_high_price = df.iloc[i]['high']

        if df.iloc[i]['swing_low_h1']:
            # New low swing: prev becomes last, last becomes new
            prev_low_idx = last_low_idx
            prev_low_price = last_low_price
            last_low_idx = df.iloc[i]['bar_index']
            last_low_price = df.iloc[i]['low']

        last_h1_hsp_idx[i] = last_high_idx
        last_h1_hsp_val[i] = last_high_price
        prev_h1_hsp_idx[i] = prev_high_idx
        prev_h1_hsp_val[i] = prev_high_price

        last_h1_lsp_idx[i] = last_low_idx
        last_h1_lsp_val[i] = last_low_price
        prev_h1_lsp_idx[i] = prev_low_idx
        prev_h1_lsp_val[i] = prev_low_price

        if (i + 1) % 100000 == 0:
            logger.info(f"  H1: Processed {i+1:,}/{n:,} rows")

    # Add to dataframe (16 columns total)
    df['last_m1_hsp_idx'] = last_m1_hsp_idx
    df['last_m1_hsp_val'] = last_m1_hsp_val
    df['prev_m1_hsp_idx'] = prev_m1_hsp_idx
    df['prev_m1_hsp_val'] = prev_m1_hsp_val

    df['last_m1_lsp_idx'] = last_m1_lsp_idx
    df['last_m1_lsp_val'] = last_m1_lsp_val
    df['prev_m1_lsp_idx'] = prev_m1_lsp_idx
    df['prev_m1_lsp_val'] = prev_m1_lsp_val

    df['last_h1_hsp_idx'] = last_h1_hsp_idx
    df['last_h1_hsp_val'] = last_h1_hsp_val
    df['prev_h1_hsp_idx'] = prev_h1_hsp_idx
    df['prev_h1_hsp_val'] = prev_h1_hsp_val

    df['last_h1_lsp_idx'] = last_h1_lsp_idx
    df['last_h1_lsp_val'] = last_h1_lsp_val
    df['prev_h1_lsp_idx'] = prev_h1_lsp_idx
    df['prev_h1_lsp_val'] = prev_h1_lsp_val

    # Update database (all 16 columns)
    logger.info("Updating database with 16 swing tracking columns...")
    conn.register('tracking_data', df[['bar_index',
                                       'last_m1_hsp_idx', 'last_m1_hsp_val',
                                       'prev_m1_hsp_idx', 'prev_m1_hsp_val',
                                       'last_m1_lsp_idx', 'last_m1_lsp_val',
                                       'prev_m1_lsp_idx', 'prev_m1_lsp_val',
                                       'last_h1_hsp_idx', 'last_h1_hsp_val',
                                       'prev_h1_hsp_idx', 'prev_h1_hsp_val',
                                       'last_h1_lsp_idx', 'last_h1_lsp_val',
                                       'prev_h1_lsp_idx', 'prev_h1_lsp_val']])

    conn.execute("""
        UPDATE master
        SET
            last_m1_hsp_idx = tracking_data.last_m1_hsp_idx,
            last_m1_hsp_val = tracking_data.last_m1_hsp_val,
            prev_m1_hsp_idx = tracking_data.prev_m1_hsp_idx,
            prev_m1_hsp_val = tracking_data.prev_m1_hsp_val,
            last_m1_lsp_idx = tracking_data.last_m1_lsp_idx,
            last_m1_lsp_val = tracking_data.last_m1_lsp_val,
            prev_m1_lsp_idx = tracking_data.prev_m1_lsp_idx,
            prev_m1_lsp_val = tracking_data.prev_m1_lsp_val,
            last_h1_hsp_idx = tracking_data.last_h1_hsp_idx,
            last_h1_hsp_val = tracking_data.last_h1_hsp_val,
            prev_h1_hsp_idx = tracking_data.prev_h1_hsp_idx,
            prev_h1_hsp_val = tracking_data.prev_h1_hsp_val,
            last_h1_lsp_idx = tracking_data.last_h1_lsp_idx,
            last_h1_lsp_val = tracking_data.last_h1_lsp_val,
            prev_h1_lsp_idx = tracking_data.prev_h1_lsp_idx,
            prev_h1_lsp_val = tracking_data.prev_h1_lsp_val
        FROM tracking_data
        WHERE master.bar_index = tracking_data.bar_index
    """)
    logger.info("✅ Last swing tracking updated")

    return df


# ============================================================================
# STEP 3: H1 SWING RANGE POSITION
# ============================================================================

def generate_h1_swing_range_position(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 3: Calculate h1_swing_range_position."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: H1 SWING RANGE POSITION")
    logger.info("="*80)

    logger.info("Calculating position within H1 swing range...")

    # Calculate: (close - last_h1_low) / (last_h1_high - last_h1_low)
    swing_range = df['last_h1_hsp_val'] - df['last_h1_lsp_val']
    valid_range = swing_range > 0

    h1_swing_range_position = np.full(len(df), np.nan)
    h1_swing_range_position[valid_range] = (
        (df['close'].values[valid_range] - df['last_h1_lsp_val'].values[valid_range]) /
        swing_range[valid_range]
    )

    df['h1_swing_range_position'] = h1_swing_range_position

    valid_count = np.sum(~np.isnan(h1_swing_range_position))
    logger.info(f"  Valid positions: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.nanmean(h1_swing_range_position):.3f}")
    logger.info(f"  Std: {np.nanstd(h1_swing_range_position):.3f}")

    # Update database
    logger.info("Updating database...")
    conn.register('position_data', df[['bar_index', 'h1_swing_range_position']])
    conn.execute("""
        UPDATE master
        SET h1_swing_range_position = position_data.h1_swing_range_position
        FROM position_data
        WHERE master.bar_index = position_data.bar_index
    """)
    logger.info("✅ H1 swing range position updated")

    return df


# ============================================================================
# STEP 4: SWING POINT RANGE
# ============================================================================

def generate_swing_point_range(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 4: Calculate swing_point_range (H1 high - H1 low)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SWING POINT RANGE")
    logger.info("="*80)

    logger.info("Calculating swing point range...")

    df['swing_point_range'] = df['last_h1_hsp_val'] - df['last_h1_lsp_val']

    valid_count = np.sum(~np.isnan(df['swing_point_range']))
    logger.info(f"  Valid ranges: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.nanmean(df['swing_point_range']):.6f}")
    logger.info(f"  Std: {np.nanstd(df['swing_point_range']):.6f}")

    # Update database
    logger.info("Updating database...")
    conn.register('range_data', df[['bar_index', 'swing_point_range']])
    conn.execute("""
        UPDATE master
        SET swing_point_range = range_data.swing_point_range
        FROM range_data
        WHERE master.bar_index = range_data.bar_index
    """)
    logger.info("✅ Swing point range updated")

    return df


# ============================================================================
# STEP 5: SWING SLOPES
# ============================================================================

def generate_swing_slopes(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 5: Calculate swing slope features (rate of change between swings) for M1 and H1."""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SWING SLOPES (M1 and H1)")
    logger.info("="*80)

    # Add columns to database if they don't exist
    for col in ['high_swing_slope_m1', 'low_swing_slope_m1', 'high_swing_slope_h1', 'low_swing_slope_h1']:
        try:
            conn.execute(f"ALTER TABLE master ADD COLUMN {col} DOUBLE")
            logger.info(f"✅ Added column: {col}")
        except:
            logger.info(f"⚠️  Column {col} already exists, will update")

    n = len(df)

    # Initialize slope arrays
    high_swing_slope_m1 = np.full(n, np.nan)
    low_swing_slope_m1 = np.full(n, np.nan)
    high_swing_slope_h1 = np.full(n, np.nan)
    low_swing_slope_h1 = np.full(n, np.nan)

    # Calculate M1 swing slopes
    logger.info("Calculating M1 swing slopes...")

    # M1 High Swing Slope
    m1_high_mask = df['swing_high_m1'].fillna(False).astype(bool)
    m1_high_indices = np.where(m1_high_mask)[0]

    for i in range(1, len(m1_high_indices)):
        curr_idx_pos = m1_high_indices[i]
        prev_idx_pos = m1_high_indices[i-1]

        curr_idx = df.iloc[curr_idx_pos]['bar_index']
        prev_idx = df.iloc[prev_idx_pos]['bar_index']
        curr_price = df.iloc[curr_idx_pos]['high']
        prev_price = df.iloc[prev_idx_pos]['high']

        time_diff = curr_idx - prev_idx
        price_diff = (curr_price - prev_price) / 0.01  # Pips

        if time_diff > 0:
            slope = price_diff / time_diff
            high_swing_slope_m1[curr_idx_pos:] = slope

    # M1 Low Swing Slope
    m1_low_mask = df['swing_low_m1'].fillna(False).astype(bool)
    m1_low_indices = np.where(m1_low_mask)[0]

    for i in range(1, len(m1_low_indices)):
        curr_idx_pos = m1_low_indices[i]
        prev_idx_pos = m1_low_indices[i-1]

        curr_idx = df.iloc[curr_idx_pos]['bar_index']
        prev_idx = df.iloc[prev_idx_pos]['bar_index']
        curr_price = df.iloc[curr_idx_pos]['low']
        prev_price = df.iloc[prev_idx_pos]['low']

        time_diff = curr_idx - prev_idx
        price_diff = (curr_price - prev_price) / 0.01

        if time_diff > 0:
            slope = price_diff / time_diff
            low_swing_slope_m1[curr_idx_pos:] = slope

    logger.info("Calculating H1 swing slopes...")

    # H1 High Swing Slope (vectorized)
    h1_high_mask = df['swing_high_h1'].fillna(False).astype(bool)
    h1_high_indices = np.where(h1_high_mask)[0]

    for i in range(1, len(h1_high_indices)):
        curr_idx_pos = h1_high_indices[i]
        prev_idx_pos = h1_high_indices[i-1]

        curr_idx = df.iloc[curr_idx_pos]['bar_index']
        prev_idx = df.iloc[prev_idx_pos]['bar_index']
        curr_price = df.iloc[curr_idx_pos]['high']
        prev_price = df.iloc[prev_idx_pos]['high']

        time_diff = curr_idx - prev_idx
        price_diff = (curr_price - prev_price) / 0.01  # Pips

        if time_diff > 0:
            slope = price_diff / time_diff
            high_swing_slope_h1[curr_idx_pos:] = slope

    # H1 Low Swing Slope (vectorized)
    h1_low_mask = df['swing_low_h1'].fillna(False).astype(bool)
    h1_low_indices = np.where(h1_low_mask)[0]

    for i in range(1, len(h1_low_indices)):
        curr_idx_pos = h1_low_indices[i]
        prev_idx_pos = h1_low_indices[i-1]

        curr_idx = df.iloc[curr_idx_pos]['bar_index']
        prev_idx = df.iloc[prev_idx_pos]['bar_index']
        curr_price = df.iloc[curr_idx_pos]['low']
        prev_price = df.iloc[prev_idx_pos]['low']

        time_diff = curr_idx - prev_idx
        price_diff = (curr_price - prev_price) / 0.01

        if time_diff > 0:
            slope = price_diff / time_diff
            low_swing_slope_h1[curr_idx_pos:] = slope

    # Add to dataframe
    df['high_swing_slope_m1'] = high_swing_slope_m1
    df['low_swing_slope_m1'] = low_swing_slope_m1
    df['high_swing_slope_h1'] = high_swing_slope_h1
    df['low_swing_slope_h1'] = low_swing_slope_h1

    # Statistics
    for col in ['high_swing_slope_m1', 'low_swing_slope_m1', 'high_swing_slope_h1', 'low_swing_slope_h1']:
        valid = df[col].dropna()
        if len(valid) > 0:
            logger.info(f"\n📊 {col}:")
            logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
            logger.info(f"  Mean: {valid.mean():+.6f} pips/bar")
            logger.info(f"  Median: {valid.median():+.6f} pips/bar")
            logger.info(f"  Std: {valid.std():.6f}")
            logger.info(f"  Range: [{valid.min():+.4f}, {valid.max():+.4f}]")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('slope_data', df[['bar_index', 'high_swing_slope_m1', 'low_swing_slope_m1',
                                     'high_swing_slope_h1', 'low_swing_slope_h1']])
    conn.execute("""
        UPDATE master
        SET
            high_swing_slope_m1 = slope_data.high_swing_slope_m1,
            low_swing_slope_m1 = slope_data.low_swing_slope_m1,
            high_swing_slope_h1 = slope_data.high_swing_slope_h1,
            low_swing_slope_h1 = slope_data.low_swing_slope_h1
        FROM slope_data
        WHERE master.bar_index = slope_data.bar_index
    """)
    logger.info("✅ Swing slopes (M1 and H1) updated")

    return df


# ============================================================================
# STEP 6: Z-SCORE FEATURES (WINDOW=20)
# ============================================================================

def calculate_fixed_std_zscore(data: np.ndarray, fixed_std: float, window: int = 20) -> np.ndarray:
    """Calculate z-score using rolling mean with fixed std."""
    series = pd.Series(data)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()

    zscores = np.full(len(data), np.nan)
    valid_mask = ~rolling_mean.isna()
    zscores[valid_mask] = (data[valid_mask] - rolling_mean[valid_mask]) / fixed_std

    return zscores


def generate_zscore_features(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 5: Generate z-score features with Window=20 and interaction feature."""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Z-SCORE FEATURES (WINDOW=20) + INTERACTION")
    logger.info("="*80)

    # Add z-score columns to database if they don't exist
    for col in ['high_swing_slope_h1_zsarctan', 'low_swing_slope_h1_zsarctan']:
        try:
            conn.execute(f"ALTER TABLE master ADD COLUMN {col} DOUBLE")
            logger.info(f"✅ Added column: {col}")
        except:
            logger.info(f"⚠️  Column {col} already exists, will update")

    # Calculate training std for both features
    train_end = int(len(df) * 0.7)

    # h1_swing_range_position
    logger.info("Calculating fixed std for h1_swing_range_position...")
    position_train_data = df['h1_swing_range_position'].iloc[:train_end].dropna().values
    position_std = float(np.std(position_train_data))
    position_mean = float(np.mean(position_train_data))
    logger.info(f"  Training rows: {len(position_train_data):,}")
    logger.info(f"  Mean: {position_mean:.6f}")
    logger.info(f"  Std: {position_std:.6f}")

    # swing_point_range
    logger.info("Calculating fixed std for swing_point_range...")
    range_train_data = df['swing_point_range'].iloc[:train_end].dropna().values
    range_std = float(np.std(range_train_data))
    range_mean = float(np.mean(range_train_data))
    logger.info(f"  Training rows: {len(range_train_data):,}")
    logger.info(f"  Mean: {range_mean:.6f}")
    logger.info(f"  Std: {range_std:.6f}")

    # high_swing_slope_h1
    logger.info("Calculating fixed std for high_swing_slope_h1...")
    high_slope_train_data = df['high_swing_slope_h1'].iloc[:train_end].dropna().values
    high_slope_std = float(np.std(high_slope_train_data))
    high_slope_mean = float(np.mean(high_slope_train_data))
    logger.info(f"  Training rows: {len(high_slope_train_data):,}")
    logger.info(f"  Mean: {high_slope_mean:.6f}")
    logger.info(f"  Std: {high_slope_std:.6f}")

    # low_swing_slope_h1
    logger.info("Calculating fixed std for low_swing_slope_h1...")
    low_slope_train_data = df['low_swing_slope_h1'].iloc[:train_end].dropna().values
    low_slope_std = float(np.std(low_slope_train_data))
    low_slope_mean = float(np.mean(low_slope_train_data))
    logger.info(f"  Training rows: {len(low_slope_train_data):,}")
    logger.info(f"  Mean: {low_slope_mean:.6f}")
    logger.info(f"  Std: {low_slope_std:.6f}")

    # Calculate z-scores
    logger.info("\nCalculating z-scores with Window=20...")

    # h1_swing_range_position_zsarctan_w20
    z_position = calculate_fixed_std_zscore(df['h1_swing_range_position'].values, position_std, window=20)
    arctan_position = np.arctan(z_position) * 2 / np.pi
    df['h1_swing_range_position_zsarctan_w20'] = arctan_position

    # swing_point_range_zsarctan_w20
    z_range = calculate_fixed_std_zscore(df['swing_point_range'].values, range_std, window=20)
    arctan_range = np.arctan(z_range) * 2 / np.pi
    df['swing_point_range_zsarctan_w20'] = arctan_range

    # high_swing_slope_h1_zsarctan
    z_high_slope = calculate_fixed_std_zscore(df['high_swing_slope_h1'].values, high_slope_std, window=20)
    arctan_high_slope = np.arctan(z_high_slope) * 2 / np.pi
    df['high_swing_slope_h1_zsarctan'] = arctan_high_slope

    # low_swing_slope_h1_zsarctan
    z_low_slope = calculate_fixed_std_zscore(df['low_swing_slope_h1'].values, low_slope_std, window=20)
    arctan_low_slope = np.arctan(z_low_slope) * 2 / np.pi
    df['low_swing_slope_h1_zsarctan'] = arctan_low_slope

    # combo_geometric (interaction feature using geometric mean)
    logger.info("\nCalculating combo_geometric interaction feature...")
    prod = df['swing_point_range_zsarctan_w20'] * df['h1_swing_range_position_zsarctan_w20']
    df['combo_geometric'] = np.sign(prod) * np.sqrt(
        np.abs(df['swing_point_range_zsarctan_w20']) * np.abs(df['h1_swing_range_position_zsarctan_w20'])
    )

    # Statistics
    logger.info("\n📊 h1_swing_range_position_zsarctan_w20:")
    valid = arctan_position[~np.isnan(arctan_position)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 swing_point_range_zsarctan_w20:")
    valid = arctan_range[~np.isnan(arctan_range)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 high_swing_slope_h1_zsarctan:")
    valid = arctan_high_slope[~np.isnan(arctan_high_slope)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 low_swing_slope_h1_zsarctan:")
    valid = arctan_low_slope[~np.isnan(arctan_low_slope)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 combo_geometric (interaction via geometric mean):")
    combo_valid = df['combo_geometric'].values[~np.isnan(df['combo_geometric'].values)]
    logger.info(f"  Valid: {len(combo_valid):,} ({len(combo_valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(combo_valid):.6f}")
    logger.info(f"  Std: {np.std(combo_valid):.6f}")
    logger.info(f"  Range: [{np.min(combo_valid):.4f}, {np.max(combo_valid):.4f}]")
    logger.info(f"  Strong signals (|z|>0.5): {np.sum(np.abs(combo_valid) > 0.5):,} ({np.sum(np.abs(combo_valid) > 0.5)/len(combo_valid)*100:.2f}%)")
    logger.info(f"  Very strong (|z|>0.6): {np.sum(np.abs(combo_valid) > 0.6):,} ({np.sum(np.abs(combo_valid) > 0.6)/len(combo_valid)*100:.2f}%)")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('zscore_data', df[['bar_index', 'h1_swing_range_position_zsarctan_w20',
                                     'swing_point_range_zsarctan_w20', 'high_swing_slope_h1_zsarctan',
                                     'low_swing_slope_h1_zsarctan', 'combo_geometric']])
    conn.execute("""
        UPDATE master
        SET
            h1_swing_range_position_zsarctan_w20 = zscore_data.h1_swing_range_position_zsarctan_w20,
            swing_point_range_zsarctan_w20 = zscore_data.swing_point_range_zsarctan_w20,
            high_swing_slope_h1_zsarctan = zscore_data.high_swing_slope_h1_zsarctan,
            low_swing_slope_h1_zsarctan = zscore_data.low_swing_slope_h1_zsarctan,
            combo_geometric = zscore_data.combo_geometric
        FROM zscore_data
        WHERE master.bar_index = zscore_data.bar_index
    """)
    logger.info("✅ Z-score features updated")

    # Update config
    config = {
        'description': 'Fixed standard deviations for z-score feature generation',
        'instrument': 'GBPJPY',
        'training_fraction': 0.7,
        'last_updated': datetime.now().strftime('%Y-%m-%d'),
        'features': {
            'h1_swing_range_position': {
                'fixed_std': position_std,
                'training_rows': len(position_train_data),
                'training_mean': position_mean,
                'description': 'Price position within H1 swing range (0=at low, 1=at high)',
                'zscore_column': 'h1_swing_range_position_zsarctan_w20',
                'window': 20
            },
            'swing_point_range': {
                'fixed_std': range_std,
                'training_rows': len(range_train_data),
                'training_mean': range_mean,
                'description': 'H1 swing range magnitude (swing_high - swing_low)',
                'zscore_column': 'swing_point_range_zsarctan_w20',
                'window': 20
            },
            'high_swing_slope_h1': {
                'fixed_std': high_slope_std,
                'training_rows': len(high_slope_train_data),
                'training_mean': high_slope_mean,
                'description': 'Rate of change between consecutive H1 swing highs (pips/bar)',
                'zscore_column': 'high_swing_slope_h1_zsarctan',
                'window': 20
            },
            'low_swing_slope_h1': {
                'fixed_std': low_slope_std,
                'training_rows': len(low_slope_train_data),
                'training_mean': low_slope_mean,
                'description': 'Rate of change between consecutive H1 swing lows (pips/bar)',
                'zscore_column': 'low_swing_slope_h1_zsarctan',
                'window': 20
            }
        },
        'notes': [
            'Fixed std prevents false extremes during consolidation periods',
            'Window=20 provides short-term regime detection',
            'Use fixed std to detect TRUE regime changes, not adaptive noise'
        ]
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Config saved to {CONFIG_PATH}")

    return df


# ============================================================================
# STEP 7: H1 TREND SLOPE (Market Structure Based)
# ============================================================================

def generate_h1_trend_slope(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 7: Generate h1_trend_slope based on market structure (HHHL, LHLL, divergence)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 7: H1 TREND SLOPE (Market Structure Based)")
    logger.info("="*80)

    # Add column if doesn't exist
    try:
        conn.execute("ALTER TABLE master ADD COLUMN h1_trend_slope DOUBLE")
        logger.info("✅ Added column: h1_trend_slope")
    except:
        logger.info("⚠️  Column h1_trend_slope already exists, will update")

    logger.info("Calculating h1_trend_slope based on market structure...")

    # Get current and previous swing values
    last_hsp = df['last_h1_hsp_val'].values
    prev_hsp = df['prev_h1_hsp_val'].values
    last_lsp = df['last_h1_lsp_val'].values
    prev_lsp = df['prev_h1_lsp_val'].values

    # Get slope z-scores
    high_slope_z = df['high_swing_slope_h1_zsarctan'].values
    low_slope_z = df['low_swing_slope_h1_zsarctan'].values

    # Initialize output
    h1_trend_slope = np.full(len(df), np.nan)

    # Classify market structure
    higher_highs = last_hsp > prev_hsp
    higher_lows = last_lsp > prev_lsp
    lower_highs = last_hsp < prev_hsp
    lower_lows = last_lsp < prev_lsp

    # HHHL: Higher Highs and Higher Lows (uptrend) -> use low_swing_slope
    hhhl_mask = higher_highs & higher_lows
    h1_trend_slope[hhhl_mask] = low_slope_z[hhhl_mask]

    # LHLL: Lower Highs and Lower Lows (downtrend) -> use high_swing_slope
    lhll_mask = lower_highs & lower_lows
    h1_trend_slope[lhll_mask] = high_slope_z[lhll_mask]

    # Divergence (HHLL or LHHL) -> use average
    hhll_mask = higher_highs & lower_lows
    lhhl_mask = lower_highs & higher_lows
    divergence_mask = hhll_mask | lhhl_mask
    h1_trend_slope[divergence_mask] = (high_slope_z[divergence_mask] + low_slope_z[divergence_mask]) / 2.0

    df['h1_trend_slope'] = h1_trend_slope

    # Statistics
    valid = h1_trend_slope[~np.isnan(h1_trend_slope)]
    hhhl_count = np.sum(hhhl_mask)
    lhll_count = np.sum(lhll_mask)
    divergence_count = np.sum(divergence_mask)

    logger.info(f"\n📊 Market Structure Distribution:")
    logger.info(f"  HHHL (uptrend): {hhhl_count:,} ({hhhl_count/len(df)*100:.1f}%)")
    logger.info(f"  LHLL (downtrend): {lhll_count:,} ({lhll_count/len(df)*100:.1f}%)")
    logger.info(f"  Divergence (HHLL/LHHL): {divergence_count:,} ({divergence_count/len(df)*100:.1f}%)")

    logger.info(f"\n📊 h1_trend_slope:")
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.6f}")
    logger.info(f"  Std: {np.std(valid):.6f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.5): {np.sum(np.abs(valid) > 0.5):,} ({np.sum(np.abs(valid) > 0.5)/len(valid)*100:.2f}%)")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('trend_slope_data', df[['bar_index', 'h1_trend_slope']])
    conn.execute("""
        UPDATE master
        SET h1_trend_slope = trend_slope_data.h1_trend_slope
        FROM trend_slope_data
        WHERE master.bar_index = trend_slope_data.bar_index
    """)
    logger.info("✅ h1_trend_slope updated")

    return df


# ============================================================================
# STEP 8: M1 TREND SLOPE (Market Structure Based)
# ============================================================================

def generate_m1_trend_slope(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 8: Generate m1_trend_slope based on market structure (HHHL, LHLL, divergence)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 8: M1 TREND SLOPE (Market Structure Based)")
    logger.info("="*80)

    # Add column if doesn't exist
    try:
        conn.execute("ALTER TABLE master ADD COLUMN m1_trend_slope DOUBLE")
        logger.info("✅ Added column: m1_trend_slope")
    except:
        logger.info("⚠️  Column m1_trend_slope already exists, will update")

    logger.info("Calculating m1_trend_slope based on market structure...")

    # Get current and previous swing values
    last_hsp = df['last_m1_hsp_val'].values
    prev_hsp = df['prev_m1_hsp_val'].values
    last_lsp = df['last_m1_lsp_val'].values
    prev_lsp = df['prev_m1_lsp_val'].values

    # Get slope values (raw, not z-scores since M1 slopes don't have z-scores yet)
    high_slope = df['high_swing_slope_m1'].values
    low_slope = df['low_swing_slope_m1'].values

    # Initialize output
    m1_trend_slope = np.full(len(df), np.nan)

    # Classify market structure
    higher_highs = last_hsp > prev_hsp
    higher_lows = last_lsp > prev_lsp
    lower_highs = last_hsp < prev_hsp
    lower_lows = last_lsp < prev_lsp

    # HHHL: Higher Highs and Higher Lows (uptrend) -> use low_swing_slope
    hhhl_mask = higher_highs & higher_lows
    m1_trend_slope[hhhl_mask] = low_slope[hhhl_mask]

    # LHLL: Lower Highs and Lower Lows (downtrend) -> use high_swing_slope
    lhll_mask = lower_highs & lower_lows
    m1_trend_slope[lhll_mask] = high_slope[lhll_mask]

    # Divergence (HHLL or LHHL) -> use average
    hhll_mask = higher_highs & lower_lows
    lhhl_mask = lower_highs & higher_lows
    divergence_mask = hhll_mask | lhhl_mask
    m1_trend_slope[divergence_mask] = (high_slope[divergence_mask] + low_slope[divergence_mask]) / 2.0

    df['m1_trend_slope'] = m1_trend_slope

    # Statistics
    valid = m1_trend_slope[~np.isnan(m1_trend_slope)]
    hhhl_count = np.sum(hhhl_mask)
    lhll_count = np.sum(lhll_mask)
    divergence_count = np.sum(divergence_mask)

    logger.info(f"\n📊 M1 Market Structure Distribution:")
    logger.info(f"  HHHL (uptrend): {hhhl_count:,} ({hhhl_count/len(df)*100:.1f}%)")
    logger.info(f"  LHLL (downtrend): {lhll_count:,} ({lhll_count/len(df)*100:.1f}%)")
    logger.info(f"  Divergence (HHLL/LHHL): {divergence_count:,} ({divergence_count/len(df)*100:.1f}%)")

    logger.info(f"\n📊 m1_trend_slope:")
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.6f} pips/bar")
    logger.info(f"  Std: {np.std(valid):.6f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|slope|>0.5): {np.sum(np.abs(valid) > 0.5):,} ({np.sum(np.abs(valid) > 0.5)/len(valid)*100:.2f}%)")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('m1_trend_slope_data', df[['bar_index', 'm1_trend_slope']])
    conn.execute("""
        UPDATE master
        SET m1_trend_slope = m1_trend_slope_data.m1_trend_slope
        FROM m1_trend_slope_data
        WHERE master.bar_index = m1_trend_slope_data.bar_index
    """)
    logger.info("✅ m1_trend_slope updated")

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute all feature generation steps in order."""
    logger.info("="*80)
    logger.info("MASTER FEATURE GENERATION")
    logger.info("="*80)
    logger.info("Generates all price-derived features for master.duckdb")
    logger.info("="*80)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    conn = duckdb.connect(str(DB_PATH))

    try:
        # Verify OHLCV data exists
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"\nTotal rows in master table: {total_rows:,}")

        # Execute all steps in order
        df = generate_swing_points(conn)
        df = generate_last_swing_tracking(conn, df)
        df = generate_h1_swing_range_position(conn, df)
        df = generate_swing_point_range(conn, df)
        df = generate_swing_slopes(conn, df)
        df = generate_zscore_features(conn, df)
        df = generate_h1_trend_slope(conn, df)
        df = generate_m1_trend_slope(conn, df)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL FEATURES GENERATED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("\nGenerated columns:")
        logger.info("  1. swing_high_m1, swing_low_m1, swing_high_h1, swing_low_h1")
        logger.info("  2. M1 Swing Points (8 columns):")
        logger.info("     last_m1_hsp_idx, last_m1_hsp_val, prev_m1_hsp_idx, prev_m1_hsp_val")
        logger.info("     last_m1_lsp_idx, last_m1_lsp_val, prev_m1_lsp_idx, prev_m1_lsp_val")
        logger.info("  3. H1 Swing Points (8 columns):")
        logger.info("     last_h1_hsp_idx, last_h1_hsp_val, prev_h1_hsp_idx, prev_h1_hsp_val")
        logger.info("     last_h1_lsp_idx, last_h1_lsp_val, prev_h1_lsp_idx, prev_h1_lsp_val")
        logger.info("  4. h1_swing_range_position")
        logger.info("  5. swing_point_range")
        logger.info("  6. high_swing_slope_m1, low_swing_slope_m1, high_swing_slope_h1, low_swing_slope_h1")
        logger.info("  7. h1_swing_range_position_zsarctan_w20, swing_point_range_zsarctan_w20")
        logger.info("     high_swing_slope_h1_zsarctan, low_swing_slope_h1_zsarctan, combo_geometric")
        logger.info("  8. h1_trend_slope (H1 market structure based: HHHL/LHLL/divergence)")
        logger.info("  9. m1_trend_slope (M1 market structure based: HHHL/LHLL/divergence)")
        logger.info(f"\nConfig saved: {CONFIG_PATH}")

    except Exception as e:
        logger.error(f"Error during feature generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
