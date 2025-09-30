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
    """Step 2: Generate last swing tracking (indices + prices)."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LAST SWING TRACKING")
    logger.info("="*80)

    n = len(df)

    # Initialize tracking arrays
    last_swing_high_idx_m1 = np.full(n, -1, dtype=np.int64)
    last_swing_high_price_m1 = np.full(n, np.nan)
    last_swing_low_idx_m1 = np.full(n, -1, dtype=np.int64)
    last_swing_low_price_m1 = np.full(n, np.nan)

    last_swing_high_idx_h1 = np.full(n, -1, dtype=np.int64)
    last_swing_high_price_h1 = np.full(n, np.nan)
    last_swing_low_idx_h1 = np.full(n, -1, dtype=np.int64)
    last_swing_low_price_h1 = np.full(n, np.nan)

    # Track M1 swings
    logger.info("Tracking M1 swings...")
    last_high_idx = -1
    last_high_price = np.nan
    last_low_idx = -1
    last_low_price = np.nan

    for i in range(n):
        if df.iloc[i]['swing_high_m1']:
            last_high_idx = df.iloc[i]['bar_index']
            last_high_price = df.iloc[i]['high']

        if df.iloc[i]['swing_low_m1']:
            last_low_idx = df.iloc[i]['bar_index']
            last_low_price = df.iloc[i]['low']

        last_swing_high_idx_m1[i] = last_high_idx
        last_swing_high_price_m1[i] = last_high_price
        last_swing_low_idx_m1[i] = last_low_idx
        last_swing_low_price_m1[i] = last_low_price

        if (i + 1) % 100000 == 0:
            logger.info(f"  Processed {i+1:,}/{n:,} rows")

    # Track H1 swings
    logger.info("Tracking H1 swings...")
    last_high_idx = -1
    last_high_price = np.nan
    last_low_idx = -1
    last_low_price = np.nan

    for i in range(n):
        if df.iloc[i]['swing_high_h1']:
            last_high_idx = df.iloc[i]['bar_index']
            last_high_price = df.iloc[i]['high']

        if df.iloc[i]['swing_low_h1']:
            last_low_idx = df.iloc[i]['bar_index']
            last_low_price = df.iloc[i]['low']

        last_swing_high_idx_h1[i] = last_high_idx
        last_swing_high_price_h1[i] = last_high_price
        last_swing_low_idx_h1[i] = last_low_idx
        last_swing_low_price_h1[i] = last_low_price

        if (i + 1) % 100000 == 0:
            logger.info(f"  Processed {i+1:,}/{n:,} rows")

    # Add to dataframe
    df['last_swing_high_idx_m1'] = last_swing_high_idx_m1
    df['last_swing_high_price_m1'] = last_swing_high_price_m1
    df['last_swing_low_idx_m1'] = last_swing_low_idx_m1
    df['last_swing_low_price_m1'] = last_swing_low_price_m1

    df['last_swing_high_idx_h1'] = last_swing_high_idx_h1
    df['last_swing_high_price_h1'] = last_swing_high_price_h1
    df['last_swing_low_idx_h1'] = last_swing_low_idx_h1
    df['last_swing_low_price_h1'] = last_swing_low_price_h1

    # Update database
    logger.info("Updating database...")
    conn.register('tracking_data', df[['bar_index',
                                       'last_swing_high_idx_m1', 'last_swing_high_price_m1',
                                       'last_swing_low_idx_m1', 'last_swing_low_price_m1',
                                       'last_swing_high_idx_h1', 'last_swing_high_price_h1',
                                       'last_swing_low_idx_h1', 'last_swing_low_price_h1']])

    conn.execute("""
        UPDATE master
        SET
            last_swing_high_idx_m1 = tracking_data.last_swing_high_idx_m1,
            last_swing_high_price_m1 = tracking_data.last_swing_high_price_m1,
            last_swing_low_idx_m1 = tracking_data.last_swing_low_idx_m1,
            last_swing_low_price_m1 = tracking_data.last_swing_low_price_m1,
            last_swing_high_idx_h1 = tracking_data.last_swing_high_idx_h1,
            last_swing_high_price_h1 = tracking_data.last_swing_high_price_h1,
            last_swing_low_idx_h1 = tracking_data.last_swing_low_idx_h1,
            last_swing_low_price_h1 = tracking_data.last_swing_low_price_h1
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
    swing_range = df['last_swing_high_price_h1'] - df['last_swing_low_price_h1']
    valid_range = swing_range > 0

    h1_swing_range_position = np.full(len(df), np.nan)
    h1_swing_range_position[valid_range] = (
        (df['close'].values[valid_range] - df['last_swing_low_price_h1'].values[valid_range]) /
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

    df['swing_point_range'] = df['last_swing_high_price_h1'] - df['last_swing_low_price_h1']

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
# STEP 5: Z-SCORE FEATURES (WINDOW=20)
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

    # Calculate z-scores
    logger.info("\nCalculating z-scores with Window=20...")

    # h1_swing_range_position_zsarctan_w20
    z_position = calculate_fixed_std_zscore(df['h1_swing_range_position'].values, position_std, window=20)
    arctan_position = np.arctan(z_position) * 2 / np.pi
    df['h1_swing_range_position_zsarctan_w20'] = arctan_position

    # swing_point_range_zsarctan
    z_range = calculate_fixed_std_zscore(df['swing_point_range'].values, range_std, window=20)
    arctan_range = np.arctan(z_range) * 2 / np.pi
    df['swing_point_range_zsarctan'] = arctan_range

    # combo_multiply (interaction feature)
    logger.info("\nCalculating combo_multiply interaction feature...")
    df['combo_multiply'] = df['swing_point_range_zsarctan'] * df['h1_swing_range_position_zsarctan_w20']

    # Statistics
    logger.info("\n📊 h1_swing_range_position_zsarctan_w20:")
    valid = arctan_position[~np.isnan(arctan_position)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 swing_point_range_zsarctan:")
    valid = arctan_range[~np.isnan(arctan_range)]
    logger.info(f"  Valid: {len(valid):,} ({len(valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(valid):.4f}")
    logger.info(f"  Std: {np.std(valid):.4f}")
    logger.info(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
    logger.info(f"  Extremes (|z|>0.8): {np.sum(np.abs(valid) > 0.8):,} ({np.sum(np.abs(valid) > 0.8)/len(valid)*100:.2f}%)")

    logger.info("\n📊 combo_multiply (interaction):")
    combo_valid = df['combo_multiply'].values[~np.isnan(df['combo_multiply'].values)]
    logger.info(f"  Valid: {len(combo_valid):,} ({len(combo_valid)/len(df)*100:.1f}%)")
    logger.info(f"  Mean: {np.mean(combo_valid):.6f}")
    logger.info(f"  Std: {np.std(combo_valid):.6f}")
    logger.info(f"  Range: [{np.min(combo_valid):.4f}, {np.max(combo_valid):.4f}]")
    logger.info(f"  Strong signals (|z|>0.4): {np.sum(np.abs(combo_valid) > 0.4):,} ({np.sum(np.abs(combo_valid) > 0.4)/len(combo_valid)*100:.2f}%)")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('zscore_data', df[['bar_index', 'h1_swing_range_position_zsarctan_w20',
                                     'swing_point_range_zsarctan', 'combo_multiply']])
    conn.execute("""
        UPDATE master
        SET
            h1_swing_range_position_zsarctan_w20 = zscore_data.h1_swing_range_position_zsarctan_w20,
            swing_point_range_zsarctan = zscore_data.swing_point_range_zsarctan,
            combo_multiply = zscore_data.combo_multiply
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
                'zscore_column': 'swing_point_range_zsarctan',
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
        df = generate_zscore_features(conn, df)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL FEATURES GENERATED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("\nGenerated columns:")
        logger.info("  1. swing_high_m1, swing_low_m1, swing_high_h1, swing_low_h1")
        logger.info("  2. last_swing_high_idx_m1, last_swing_high_price_m1")
        logger.info("     last_swing_low_idx_m1, last_swing_low_price_m1")
        logger.info("     last_swing_high_idx_h1, last_swing_high_price_h1")
        logger.info("     last_swing_low_idx_h1, last_swing_low_price_h1")
        logger.info("  3. h1_swing_range_position")
        logger.info("  4. swing_point_range")
        logger.info("  5. h1_swing_range_position_zsarctan_w20, swing_point_range_zsarctan, combo_multiply")
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
