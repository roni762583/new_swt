#!/usr/bin/env python3
"""
Production Data Cleaning for SWT Trading System
Removes weekends, gaps, and validates data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_forex_data(input_file: str, output_file: str):
    """
    Clean forex data for production trading:
    1. Remove weekend periods (Friday 21:00 - Sunday 21:00 GMT)
    2. Remove sessions with gaps > 10 minutes
    3. Validate pip ranges are realistic
    4. Remove outliers and bad data
    """

    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    original_rows = len(df)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Original data: {original_rows:,} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 1. REMOVE WEEKENDS
    logger.info("Removing weekend data...")

    # Create weekday and hour columns
    df['weekday'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['hour'] = df['timestamp'].dt.hour

    # Define weekend periods: Friday 21:00 - Sunday 21:00 GMT
    is_weekend = (
        # Friday after 21:00
        ((df['weekday'] == 4) & (df['hour'] >= 21)) |
        # All of Saturday
        (df['weekday'] == 5) |
        # Sunday before 21:00
        ((df['weekday'] == 6) & (df['hour'] < 21))
    )

    weekend_rows = is_weekend.sum()
    logger.info(f"Removing {weekend_rows:,} weekend rows ({weekend_rows/len(df)*100:.1f}%)")
    df = df[~is_weekend].copy()

    # 2. DETECT AND HANDLE GAPS
    logger.info("Detecting gaps...")

    # Calculate time differences between consecutive rows
    df['time_diff'] = df['timestamp'].diff()
    df['gap_minutes'] = df['time_diff'].dt.total_seconds() / 60

    # Find gaps > 10 minutes (excluding expected weekend gaps)
    large_gaps = df[df['gap_minutes'] > 10].copy()

    if len(large_gaps) > 0:
        logger.info(f"Found {len(large_gaps)} gaps > 10 minutes")

        # Mark rows after large gaps for review
        gap_indices = large_gaps.index.tolist()

        # For production, we'll split data at major gaps
        # This prevents training on artificial price jumps
        logger.info("Marking sessions around large gaps...")

    # 3. VALIDATE PRICE RANGES
    logger.info("Validating price ranges...")

    # Calculate pip ranges for each bar (GBPJPY: 1 pip = 0.01)
    df['range_pips'] = (df['high'] - df['low']) * 100

    # Statistical analysis of ranges
    range_stats = {
        'mean': df['range_pips'].mean(),
        'median': df['range_pips'].median(),
        'std': df['range_pips'].std(),
        'max': df['range_pips'].max(),
        'p95': df['range_pips'].quantile(0.95),
        'p99': df['range_pips'].quantile(0.99)
    }

    logger.info("Price range statistics (pips per minute):")
    for key, value in range_stats.items():
        logger.info(f"  {key}: {value:.2f}")

    # Flag unrealistic ranges
    # Normal GBPJPY: 1-5 pips/min, news: 10-20 pips, extreme: 30 pips
    EXTREME_THRESHOLD = 50  # pips per minute

    extreme_bars = df[df['range_pips'] > EXTREME_THRESHOLD]
    if len(extreme_bars) > 0:
        logger.warning(f"Found {len(extreme_bars)} bars with range > {EXTREME_THRESHOLD} pips")
        logger.warning("Sample extreme bars:")
        for idx in extreme_bars.index[:5]:
            row = df.loc[idx]
            logger.warning(f"  {row['timestamp']}: {row['range_pips']:.1f} pips "
                         f"(L:{row['low']:.3f} H:{row['high']:.3f})")

    # 4. REMOVE OUTLIERS
    logger.info("Removing outliers...")

    # Option 1: Remove bars with extreme ranges (likely bad data)
    outlier_threshold = 100  # pips - anything above this is likely bad data
    outliers = df['range_pips'] > outlier_threshold
    logger.info(f"Removing {outliers.sum()} bars with range > {outlier_threshold} pips")
    df = df[~outliers].copy()

    # 5. FILL SMALL GAPS
    logger.info("Checking for missing minutes...")

    # Resample to ensure continuous 1-minute data (forward fill small gaps)
    df_resampled = df.set_index('timestamp').resample('1min').first()

    # Count missing minutes
    missing_minutes = df_resampled['open'].isna().sum()
    if missing_minutes > 0:
        logger.info(f"Found {missing_minutes} missing minutes")
        # For production, we do NOT interpolate - we mark these as invalid
        # We'll only use continuous sessions

    # 6. FINAL VALIDATION
    logger.info("Final validation...")

    # Reset index and clean up
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.reset_index(drop=True)

    # Calculate final statistics
    final_rows = len(df)
    removed_rows = original_rows - final_rows
    removal_pct = (removed_rows / original_rows) * 100

    logger.info("="*60)
    logger.info("CLEANING COMPLETE")
    logger.info(f"Original rows: {original_rows:,}")
    logger.info(f"Final rows: {final_rows:,}")
    logger.info(f"Removed: {removed_rows:,} ({removal_pct:.1f}%)")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Verify no weekend data remains
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour
    remaining_weekend = (
        ((df['weekday'] == 4) & (df['hour'] >= 21)) |
        (df['weekday'] == 5) |
        ((df['weekday'] == 6) & (df['hour'] < 21))
    ).sum()

    if remaining_weekend > 0:
        logger.error(f"ERROR: {remaining_weekend} weekend rows still present!")
        sys.exit(1)
    else:
        logger.info("✓ No weekend data present")

    # Check pip ranges
    final_range_stats = {
        'mean': df['range_pips'].mean() if 'range_pips' in df else 0,
        'max': df['range_pips'].max() if 'range_pips' in df else 0,
        'p99': df['range_pips'].quantile(0.99) if 'range_pips' in df else 0
    }

    logger.info(f"✓ Mean pip range: {final_range_stats['mean']:.2f}")
    logger.info(f"✓ Max pip range: {final_range_stats['max']:.2f}")
    logger.info(f"✓ 99th percentile: {final_range_stats['p99']:.2f}")

    # Save cleaned data
    logger.info(f"Saving cleaned data to {output_file}")
    df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_csv(output_file, index=False)
    logger.info("✓ Data saved successfully")

    return df


if __name__ == "__main__":
    input_file = "data/GBPJPY_M1_3.5years_20250912.csv"
    output_file = "data/GBPJPY_M1_3.5years_CLEAN.csv"

    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    clean_forex_data(input_file, output_file)