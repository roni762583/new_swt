#!/usr/bin/env python3
"""
MASTER FEATURE GENERATION SCRIPT
Consolidates all feature generation for master.duckdb table.

Execution order:
1. ZigZag indicator (30 pips minimum)
2. Swing point detection (M1 and H1)
3. Last swing tracking (indices + prices)
4. H1 swing range position
5. Swing point range (H1 high - H1 low)
6. Swing slopes (M1 and H1)
7. Z-score features with Window=20
8. H1 trend slope
9. Pretrain action labels (from ZigZag)
10. M1 trend slope
11. RSI extreme
12. BB position

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

# Constants
MIN_ZIGZAG_SWING = 0.30  # 30 pips for GBPJPY


# ============================================================================
# STEP 0: ZIGZAG INDICATOR
# ============================================================================

def calculate_zigzag(highs: np.ndarray, lows: np.ndarray, min_swing: float):
    """
    Calculate ZigZag indicator with minimum swing threshold.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        min_swing: Minimum swing size in price units (0.30 = 30 pips for GBPJPY)

    Returns:
        pivots: Array of pivot prices (NaN for non-pivots)
        directions: Array of trend direction (+1 up, -1 down, 0 undefined)
        is_pivot: Boolean array marking pivot points
    """
    n = len(highs)
    pivots = np.full(n, np.nan)
    directions = np.zeros(n, dtype=np.int8)
    is_pivot = np.zeros(n, dtype=bool)

    # Find first pivot
    last_pivot_idx = 0
    last_pivot_price = highs[0]
    last_pivot_type = 1  # 1=high, -1=low

    i = 1
    while i < n:
        current_high = highs[i]
        current_low = lows[i]

        if last_pivot_type == 1:  # Last pivot was a high, looking for low
            potential_swing = last_pivot_price - current_low
            if potential_swing >= min_swing:
                # Found significant low
                pivots[i] = current_low
                is_pivot[i] = True
                directions[last_pivot_idx:i+1] = -1  # Downtrend
                last_pivot_idx = i
                last_pivot_price = current_low
                last_pivot_type = -1

        else:  # Last pivot was a low, looking for high
            potential_swing = current_high - last_pivot_price
            if potential_swing >= min_swing:
                # Found significant high
                pivots[i] = current_high
                is_pivot[i] = True
                directions[last_pivot_idx:i+1] = 1  # Uptrend
                last_pivot_idx = i
                last_pivot_price = current_high
                last_pivot_type = 1

        i += 1
        if i % 100000 == 0:
            logger.info(f"  Processed {i:,}/{n:,} bars")

    return pivots, directions, is_pivot


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI indicator."""
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def generate_zigzag(conn: duckdb.DuckDBPyConnection):
    """Step 0: Generate ZigZag indicator with 30 pip minimum swing."""
    logger.info("\n" + "="*80)
    logger.info(f"STEP 0: ZIGZAG INDICATOR (MIN_SWING = {MIN_ZIGZAG_SWING} = 30 pips)")
    logger.info("="*80)

    # Add columns if needed
    for col in ['zigzag_price', 'zigzag_direction', 'is_zigzag_pivot']:
        try:
            col_type = 'DOUBLE' if col == 'zigzag_price' else ('TINYINT' if col == 'zigzag_direction' else 'BOOLEAN')
            conn.execute(f"ALTER TABLE master ADD COLUMN {col} {col_type}")
            logger.info(f"✅ Added column: {col}")
        except:
            logger.info(f"⚠️  Column {col} exists, will update")

    # Fetch data
    logger.info("Fetching OHLC data...")
    df = conn.execute("""
        SELECT bar_index, high, low
        FROM master
        ORDER BY bar_index
    """).fetch_df()
    logger.info(f"Loaded {len(df):,} rows")

    # Calculate ZigZag
    logger.info(f"Calculating ZigZag...")
    zigzag_price, zigzag_direction, is_zigzag_pivot = calculate_zigzag(
        df['high'].values,
        df['low'].values,
        MIN_ZIGZAG_SWING
    )

    df['zigzag_price'] = zigzag_price
    df['zigzag_direction'] = zigzag_direction
    df['is_zigzag_pivot'] = is_zigzag_pivot

    # Statistics
    pivot_count = np.sum(is_zigzag_pivot)
    pivot_prices = zigzag_price[~np.isnan(zigzag_price)]
    if len(pivot_prices) > 1:
        swings = np.abs(np.diff(pivot_prices)) / 0.01
        avg_swing = np.mean(swings)
        swing_range = [np.min(swings), np.max(swings)]
    else:
        avg_swing = 0
        swing_range = [0, 0]

    logger.info(f"\n📊 ZigZag Statistics:")
    logger.info(f"  Pivots: {pivot_count:,} ({pivot_count/len(df)*100:.2f}%)")
    logger.info(f"  Avg swing: {avg_swing:.1f} pips")
    logger.info(f"  Range: [{swing_range[0]:.1f}, {swing_range[1]:.1f}] pips")

    # Update database
    logger.info("Updating database...")
    conn.register('zigzag_data', df[['bar_index', 'zigzag_price', 'zigzag_direction', 'is_zigzag_pivot']])
    conn.execute("""
        UPDATE master
        SET
            zigzag_price = zigzag_data.zigzag_price,
            zigzag_direction = zigzag_data.zigzag_direction,
            is_zigzag_pivot = zigzag_data.is_zigzag_pivot
        FROM zigzag_data
        WHERE master.bar_index = zigzag_data.bar_index
    """)
    logger.info("✅ ZigZag updated")

    return df


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

    # Track M1 swings (last and prev) - Vectorized
    logger.info("Tracking M1 swings (last and prev)...")

    # Pre-extract values for speed
    swing_high_m1 = df['swing_high_m1'].values
    swing_low_m1 = df['swing_low_m1'].values
    bar_indices = df['bar_index'].values
    highs = df['high'].values
    lows = df['low'].values

    last_high_idx = -1
    last_high_price = np.nan
    prev_high_idx = -1
    prev_high_price = np.nan
    last_low_idx = -1
    last_low_price = np.nan
    prev_low_idx = -1
    prev_low_price = np.nan

    for i in range(n):
        if swing_high_m1[i]:
            prev_high_idx = last_high_idx
            prev_high_price = last_high_price
            last_high_idx = bar_indices[i]
            last_high_price = highs[i]

        if swing_low_m1[i]:
            prev_low_idx = last_low_idx
            prev_low_price = last_low_price
            last_low_idx = bar_indices[i]
            last_low_price = lows[i]

        last_m1_hsp_idx[i] = last_high_idx
        last_m1_hsp_val[i] = last_high_price
        prev_m1_hsp_idx[i] = prev_high_idx
        prev_m1_hsp_val[i] = prev_high_price
        last_m1_lsp_idx[i] = last_low_idx
        last_m1_lsp_val[i] = last_low_price
        prev_m1_lsp_idx[i] = prev_low_idx
        prev_m1_lsp_val[i] = prev_low_price

    # Track H1 swings (last and prev) - Vectorized
    logger.info("Tracking H1 swings (last and prev)...")

    swing_high_h1 = df['swing_high_h1'].values
    swing_low_h1 = df['swing_low_h1'].values

    last_high_idx = -1
    last_high_price = np.nan
    prev_high_idx = -1
    prev_high_price = np.nan
    last_low_idx = -1
    last_low_price = np.nan
    prev_low_idx = -1
    prev_low_price = np.nan

    for i in range(n):
        if swing_high_h1[i]:
            prev_high_idx = last_high_idx
            prev_high_price = last_high_price
            last_high_idx = bar_indices[i]
            last_high_price = highs[i]

        if swing_low_h1[i]:
            prev_low_idx = last_low_idx
            prev_low_price = last_low_price
            last_low_idx = bar_indices[i]
            last_low_price = lows[i]

        last_h1_hsp_idx[i] = last_high_idx
        last_h1_hsp_val[i] = last_high_price
        prev_h1_hsp_idx[i] = prev_high_idx
        prev_h1_hsp_val[i] = prev_high_price
        last_h1_lsp_idx[i] = last_low_idx
        last_h1_lsp_val[i] = last_low_price
        prev_h1_lsp_idx[i] = prev_low_idx
        prev_h1_lsp_val[i] = prev_low_price

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

    # Calculate M1 swing slopes - Optimized
    logger.info("Calculating M1 swing slopes...")

    # Pre-extract arrays
    bar_idx_arr = df['bar_index'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    swing_high_m1_arr = df['swing_high_m1'].fillna(False).astype(bool).values
    swing_low_m1_arr = df['swing_low_m1'].fillna(False).astype(bool).values

    # M1 High Swing Slope
    m1_high_indices = np.where(swing_high_m1_arr)[0]
    for i in range(1, len(m1_high_indices)):
        curr_pos = m1_high_indices[i]
        prev_pos = m1_high_indices[i-1]
        time_diff = bar_idx_arr[curr_pos] - bar_idx_arr[prev_pos]
        if time_diff > 0:
            price_diff = (high_arr[curr_pos] - high_arr[prev_pos]) / 0.01
            high_swing_slope_m1[curr_pos:] = price_diff / time_diff

    # M1 Low Swing Slope
    m1_low_indices = np.where(swing_low_m1_arr)[0]
    for i in range(1, len(m1_low_indices)):
        curr_pos = m1_low_indices[i]
        prev_pos = m1_low_indices[i-1]
        time_diff = bar_idx_arr[curr_pos] - bar_idx_arr[prev_pos]
        if time_diff > 0:
            price_diff = (low_arr[curr_pos] - low_arr[prev_pos]) / 0.01
            low_swing_slope_m1[curr_pos:] = price_diff / time_diff

    logger.info("Calculating H1 swing slopes...")

    swing_high_h1_arr = df['swing_high_h1'].fillna(False).astype(bool).values
    swing_low_h1_arr = df['swing_low_h1'].fillna(False).astype(bool).values

    # H1 High Swing Slope
    h1_high_indices = np.where(swing_high_h1_arr)[0]
    for i in range(1, len(h1_high_indices)):
        curr_pos = h1_high_indices[i]
        prev_pos = h1_high_indices[i-1]
        time_diff = bar_idx_arr[curr_pos] - bar_idx_arr[prev_pos]
        if time_diff > 0:
            price_diff = (high_arr[curr_pos] - high_arr[prev_pos]) / 0.01
            high_swing_slope_h1[curr_pos:] = price_diff / time_diff

    # H1 Low Swing Slope
    h1_low_indices = np.where(swing_low_h1_arr)[0]
    for i in range(1, len(h1_low_indices)):
        curr_pos = h1_low_indices[i]
        prev_pos = h1_low_indices[i-1]
        time_diff = bar_idx_arr[curr_pos] - bar_idx_arr[prev_pos]
        if time_diff > 0:
            price_diff = (low_arr[curr_pos] - low_arr[prev_pos]) / 0.01
            low_swing_slope_h1[curr_pos:] = price_diff / time_diff

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
# STEP 8: PRETRAIN ACTION LABELS (ZigZag Based)
# ============================================================================

def generate_pretrain_action(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 8: Generate pretrain_action labels from ZigZag pivots.

    Logic:
    - Enter 2 bars after pivot confirmed (pivot index + 2)
    - Direction: BUY if zigzag going up, SELL if going down
    - Exit: Bar before next pivot
    - Actions: 0=hold, 1=buy, 2=sell, 3=close
    """
    logger.info("\n" + "="*80)
    logger.info("STEP 8: PRETRAIN ACTION LABELS (ZigZag Based)")
    logger.info("="*80)

    # Add column if doesn't exist
    try:
        conn.execute("ALTER TABLE master ADD COLUMN pretrain_action TINYINT")
        logger.info("✅ Added column: pretrain_action")
    except:
        logger.info("⚠️  Column pretrain_action already exists, will update")

    # Fetch ZigZag data
    logger.info("Fetching ZigZag data...")
    zigzag_df = conn.execute("""
        SELECT bar_index, is_zigzag_pivot, zigzag_direction
        FROM master
        ORDER BY bar_index
    """).fetch_df()

    logger.info("Generating pretrain action labels from ZigZag pivots...")

    # Initialize actions (0 = hold)
    n = len(zigzag_df)
    pretrain_action = np.zeros(n, dtype=np.int8)

    # Find all pivot indices
    pivot_indices = np.where(zigzag_df['is_zigzag_pivot'].fillna(False).astype(bool))[0]
    logger.info(f"  Found {len(pivot_indices):,} pivot points")

    # Generate entry/exit signals
    for i in range(len(pivot_indices)):
        pivot_idx = pivot_indices[i]

        # Entry: 2 bars after pivot confirmation
        entry_idx = pivot_idx + 2
        if entry_idx >= n:
            continue

        # Get direction at entry
        direction = zigzag_df.iloc[entry_idx]['zigzag_direction']

        # Set entry action (1=buy if uptrend, 2=sell if downtrend)
        if direction == 1:
            pretrain_action[entry_idx] = 1  # BUY
        elif direction == -1:
            pretrain_action[entry_idx] = 2  # SELL

        # Exit: bar before next pivot (if there is one)
        if i + 1 < len(pivot_indices):
            next_pivot_idx = pivot_indices[i + 1]
            exit_idx = next_pivot_idx - 1

            if exit_idx > entry_idx and exit_idx < n:
                pretrain_action[exit_idx] = 3  # CLOSE

    zigzag_df['pretrain_action'] = pretrain_action

    # Statistics
    action_counts = {
        0: np.sum(pretrain_action == 0),
        1: np.sum(pretrain_action == 1),
        2: np.sum(pretrain_action == 2),
        3: np.sum(pretrain_action == 3),
    }

    logger.info(f"\n📊 Pretrain Action Distribution:")
    logger.info(f"  0 (HOLD):  {action_counts[0]:,} ({action_counts[0]/n*100:.2f}%)")
    logger.info(f"  1 (BUY):   {action_counts[1]:,} ({action_counts[1]/n*100:.2f}%)")
    logger.info(f"  2 (SELL):  {action_counts[2]:,} ({action_counts[2]/n*100:.2f}%)")
    logger.info(f"  3 (CLOSE): {action_counts[3]:,} ({action_counts[3]/n*100:.2f}%)")

    total_trades = action_counts[1] + action_counts[2]
    logger.info(f"\n  Total trades: {total_trades:,}")
    logger.info(f"  Buy/Sell ratio: {action_counts[1]/action_counts[2]:.2f}" if action_counts[2] > 0 else "  Buy/Sell ratio: N/A")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('pretrain_data', zigzag_df[['bar_index', 'pretrain_action']])
    conn.execute("""
        UPDATE master
        SET pretrain_action = pretrain_data.pretrain_action
        FROM pretrain_data
        WHERE master.bar_index = pretrain_data.bar_index
    """)
    logger.info("✅ pretrain_action updated")

    return df


# ============================================================================
# STEP 9: RSI EXTREME
# ============================================================================

def generate_rsi_extreme(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 9: Generate RSI extreme: (RSI14 - 50) / 50, bounded [-1, 1]."""
    logger.info("\n" + "="*80)
    logger.info("STEP 9: RSI EXTREME")
    logger.info("="*80)

    try:
        conn.execute("ALTER TABLE master ADD COLUMN rsi_extreme DOUBLE")
        logger.info("✅ Added column: rsi_extreme")
    except:
        logger.info("⚠️  Column exists, will update")

    logger.info("Calculating RSI14...")
    rsi = calculate_rsi(df['close'].values, period=14)
    rsi_extreme = np.clip((rsi - 50) / 50, -1, 1)
    df['rsi_extreme'] = rsi_extreme

    valid = rsi_extreme[~np.isnan(rsi_extreme)]
    logger.info(f"\n📊 RSI Extreme:")
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|x|>0.8): {np.sum(np.abs(valid) > 0.8):,}")

    logger.info("Updating database...")
    conn.register('rsi_data', df[['bar_index', 'rsi_extreme']])
    conn.execute("""
        UPDATE master
        SET rsi_extreme = rsi_data.rsi_extreme
        FROM rsi_data
        WHERE master.bar_index = rsi_data.bar_index
    """)
    logger.info("✅ rsi_extreme updated")

    return df


# ============================================================================
# STEP 10: BB POSITION
# ============================================================================

def generate_bb_position(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 10: Generate BB position: (close - sma20) / (2 * std20), clipped [-2, 2]."""
    logger.info("\n" + "="*80)
    logger.info("STEP 10: BB POSITION")
    logger.info("="*80)

    try:
        conn.execute("ALTER TABLE master ADD COLUMN bb_position DOUBLE")
        logger.info("✅ Added column: bb_position")
    except:
        logger.info("⚠️  Column exists, will update")

    logger.info("Calculating BB position...")
    sma20 = df['close'].rolling(window=20, min_periods=20).mean()
    std20 = df['close'].rolling(window=20, min_periods=20).std()
    bb_position = np.clip((df['close'] - sma20) / (2 * std20), -2.0, 2.0)
    df['bb_position'] = bb_position

    valid = bb_position.dropna().values
    above_1 = np.sum(valid > 1.0)
    below_minus1 = np.sum(valid < -1.0)
    in_range = len(valid) - above_1 - below_minus1

    logger.info(f"\n📊 BB Position:")
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  >+1.0: {above_1:,} ({above_1/len(valid)*100:.2f}%)")
    logger.info(f"  -1 to +1: {in_range:,} ({in_range/len(valid)*100:.2f}%)")
    logger.info(f"  <-1.0: {below_minus1:,} ({below_minus1/len(valid)*100:.2f}%)")

    logger.info("Updating database...")
    conn.register('bb_data', df[['bar_index', 'bb_position']])
    conn.execute("""
        UPDATE master
        SET bb_position = bb_data.bb_position
        FROM bb_data
        WHERE master.bar_index = bb_data.bar_index
    """)
    logger.info("✅ bb_position updated")

    return df


# ============================================================================
# STEP 11: M1 TREND SLOPE (Market Structure Based)
# ============================================================================

def generate_m1_trend_slope(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame):
    """Step 11: Generate m1_trend_slope based on market structure (HHHL, LHLL, divergence)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 11: M1 TREND SLOPE (Market Structure Based)")
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
        df = generate_zigzag(conn)  # Step 0: ZigZag (30 pips)
        df = generate_swing_points(conn)  # Step 1
        df = generate_last_swing_tracking(conn, df)  # Step 2
        df = generate_h1_swing_range_position(conn, df)  # Step 3
        df = generate_swing_point_range(conn, df)  # Step 4
        df = generate_swing_slopes(conn, df)  # Step 5
        df = generate_zscore_features(conn, df)  # Step 6
        df = generate_h1_trend_slope(conn, df)  # Step 7
        df = generate_pretrain_action(conn, df)  # Step 8
        df = generate_rsi_extreme(conn, df)  # Step 9
        df = generate_bb_position(conn, df)  # Step 10
        df = generate_m1_trend_slope(conn, df)  # Step 11

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
        logger.info("  9. pretrain_action (ZigZag labels: 0=hold, 1=buy, 2=sell, 3=close)")
        logger.info(" 10. m1_trend_slope (M1 market structure based: HHHL/LHLL/divergence)")
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
