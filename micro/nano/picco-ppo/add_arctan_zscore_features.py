#!/usr/bin/env python3
"""
Add arctan-transformed z-score features to master table using FIXED STD approach.

Key Innovation: Uses rolling mean with FIXED std from training data to detect
true regime changes and avoid false extremes during consolidation.

Configuration: Fixed std values stored in feature_zscore_config.json

Functions:
- calculate_training_std_from_range(): Calculate fixed std from specified row range
- save_fixed_std_to_config(): Save calculated std to config file
- load_fixed_std_from_config(): Load fixed std from config file
- calculate_fixed_std_zscore(): Rolling mean with fixed std (regime-aware)
- calculate_arctan_transform(): Arctan transformation bounded to [-1, 1]
- add_arctan_zscore_feature(): Main function to add feature to master table
"""

import numpy as np
import pandas as pd
import duckdb
import logging
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
CONFIG_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/feature_zscore_config.json")


def calculate_training_std_from_range(
    conn: duckdb.DuckDBPyConnection,
    column_name: str,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
    train_fraction: float = 0.7
) -> Dict[str, float]:
    """
    Calculate standard deviation from specified row range or training fraction.

    Args:
        conn: DuckDB connection
        column_name: Name of column to analyze
        start_row: Start bar_index (if None, uses 0)
        end_row: End bar_index (if None, uses train_fraction of data)
        train_fraction: Fraction to use as training if end_row not specified

    Returns:
        Dict with 'std', 'mean', 'start_row', 'end_row', 'n_rows'

    Purpose:
        Calculate fixed std from stable training period for consistent
        baseline across all market regimes.
    """
    # Get total rows if range not specified
    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM master WHERE {column_name} IS NOT NULL"
    ).fetchone()[0]

    if start_row is None:
        start_row = 0

    if end_row is None:
        end_row = int(total_rows * train_fraction)

    # Fetch data from specified range
    query = f"""
        SELECT {column_name}
        FROM master
        WHERE {column_name} IS NOT NULL
        ORDER BY bar_index
        LIMIT {end_row - start_row}
        OFFSET {start_row}
    """
    data = conn.execute(query).fetchdf()[column_name].values

    # Calculate statistics
    train_std = float(np.std(data))
    train_mean = float(np.mean(data))
    n_rows = len(data)

    logger.info(f"Training statistics for {column_name}:")
    logger.info(f"  Row range: {start_row:,} to {end_row:,}")
    logger.info(f"  N rows: {n_rows:,}")
    logger.info(f"  Mean: {train_mean:.6f}")
    logger.info(f"  Std: {train_std:.6f}")
    logger.info(f"  Training fraction: {n_rows/total_rows*100:.1f}%")

    return {
        'std': train_std,
        'mean': train_mean,
        'start_row': start_row,
        'end_row': end_row,
        'n_rows': n_rows
    }


def save_fixed_std_to_config(
    column_name: str,
    stats: Dict[str, float],
    zscore_column_name: str,
    window: int = 500,
    description: str = ""
):
    """
    Save calculated fixed std to configuration file.

    Args:
        column_name: Source column name
        stats: Dictionary with 'std', 'mean', 'n_rows' from calculate_training_std_from_range
        zscore_column_name: Name of resulting z-score column
        window: Rolling window size for mean
        description: Description of the feature
    """
    # Load existing config or create new
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'description': 'Fixed standard deviations for z-score feature generation',
            'instrument': 'GBPJPY',
            'training_fraction': 0.7,
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'features': {},
            'notes': [
                'Fixed std prevents false extremes during consolidation periods',
                'Use fixed std to detect TRUE regime changes, not adaptive noise'
            ]
        }

    # Add/update feature
    config['features'][column_name] = {
        'fixed_std': stats['std'],
        'training_rows': stats['n_rows'],
        'training_mean': stats['mean'],
        'description': description,
        'zscore_column': zscore_column_name,
        'window': window
    }
    config['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    # Save
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Saved fixed std config to {CONFIG_PATH}")


def load_fixed_std_from_config(column_name: str) -> Optional[Dict]:
    """
    Load fixed std configuration for a column.

    Args:
        column_name: Source column name

    Returns:
        Dict with feature config or None if not found
    """
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found: {CONFIG_PATH}")
        return None

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if column_name not in config.get('features', {}):
        logger.warning(f"Column '{column_name}' not found in config")
        return None

    return config['features'][column_name]


def calculate_fixed_std_zscore(
    data: np.ndarray,
    fixed_std: float,
    window: int = 500
) -> np.ndarray:
    """
    Calculate z-score using rolling mean with FIXED std from training.

    Args:
        data: Input array of values
        fixed_std: Fixed standard deviation from training data
        window: Rolling window size for mean (default: 500)

    Returns:
        Array of z-scores (same length as input, first window-1 values are NaN)

    Formula:
        z[i] = (x[i] - rolling_mean(x, window)) / fixed_std_training

    Benefits:
        - Adapts to recent price levels (rolling mean)
        - Consistent volatility baseline (fixed std)
        - Detects TRUE regime changes (breakouts from consolidation)
        - Avoids false extremes during low volatility periods
    """
    n = len(data)
    zscores = np.full(n, np.nan)

    # Calculate rolling mean
    series = pd.Series(data)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()

    # Calculate z-score with fixed std
    valid_mask = ~rolling_mean.isna()
    zscores[valid_mask] = (data[valid_mask] - rolling_mean[valid_mask]) / fixed_std

    return zscores


def calculate_arctan_transform(data: np.ndarray) -> np.ndarray:
    """
    Apply arctan transformation to bound data to [-1, 1].

    Args:
        data: Input array (typically z-scores)

    Returns:
        Transformed array bounded to [-1, 1]

    Formula:
        y = arctan(x) * 2 / π

    Properties:
        - Smooth, differentiable
        - Maps (-∞, +∞) → (-1, +1)
        - Preserves sign and order
        - Zero stays at zero
    """
    return np.arctan(data) * 2 / np.pi


def add_arctan_zscore_feature(
    conn: duckdb.DuckDBPyConnection,
    source_column: str = "h1_swing_range_position",
    zscore_column: str = "h1_swing_range_position_zsarctan",
    window: int = 500,
    recalculate_std: bool = False,
    train_fraction: float = 0.7,
    description: str = "Price position within H1 swing range"
) -> Tuple[int, float]:
    """
    Add arctan(zscore(column)) to master table using fixed std approach.

    Args:
        conn: DuckDB connection
        source_column: Source column name (default: h1_swing_range_position)
        zscore_column: Target column name (default: h1_swing_range_position_zsarctan)
        window: Rolling window for mean (default: 500)
        recalculate_std: Force recalculation of fixed std (default: False)
        train_fraction: Training fraction if recalculating (default: 0.7)
        description: Feature description for config

    Returns:
        Tuple of (number of valid values, fixed std used)

    Column added:
        Arctan-transformed z-score with fixed std from training data,
        bounded to [-1, 1]
    """
    # Get total rows
    total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    logger.info(f"Total rows to process: {total_rows:,}")

    # Load or calculate fixed std
    if recalculate_std or not CONFIG_PATH.exists():
        logger.info(f"\nCalculating fixed std for '{source_column}'...")
        stats = calculate_training_std_from_range(
            conn, source_column, train_fraction=train_fraction
        )
        fixed_std = stats['std']

        # Save to config
        save_fixed_std_to_config(
            source_column, stats, zscore_column, window, description
        )
    else:
        logger.info(f"\nLoading fixed std from config...")
        config = load_fixed_std_from_config(source_column)
        if config is None:
            logger.info(f"Config not found, calculating...")
            stats = calculate_training_std_from_range(
                conn, source_column, train_fraction=train_fraction
            )
            fixed_std = stats['std']
            save_fixed_std_to_config(
                source_column, stats, zscore_column, window, description
            )
        else:
            fixed_std = config['fixed_std']
            logger.info(f"✅ Loaded fixed std: {fixed_std:.6f}")
            logger.info(f"   Training rows: {config['training_rows']:,}")
            logger.info(f"   Training mean: {config['training_mean']:.6f}")

    # Add new column
    logger.info(f"\nAdding '{zscore_column}' column to master table...")
    try:
        conn.execute(f"ALTER TABLE master ADD COLUMN {zscore_column} DOUBLE")
        logger.info(f"  ✅ Added column: {zscore_column} (DOUBLE)")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"  ⚠️  Column {zscore_column} already exists, will update")
            conn.execute(f"UPDATE master SET {zscore_column} = NULL")
        else:
            raise

    # Fetch data ordered by bar_index
    logger.info(f"Fetching {source_column} data...")
    df = conn.execute(f"""
        SELECT
            bar_index,
            {source_column}
        FROM master
        ORDER BY bar_index
    """).fetch_df()

    logger.info(f"Loaded {len(df):,} rows")

    # Calculate z-score with fixed std
    logger.info(f"\nCalculating z-score (rolling mean={window}, fixed std={fixed_std:.6f})...")
    source_data = df[source_column].values
    zscores = calculate_fixed_std_zscore(source_data, fixed_std, window=window)

    # Apply arctan transformation
    logger.info("Applying arctan transformation...")
    arctan_zscores = calculate_arctan_transform(zscores)

    # Add to dataframe
    df[zscore_column] = arctan_zscores

    # Calculate statistics
    valid_values = ~np.isnan(arctan_zscores)
    valid_count = np.sum(valid_values)

    if valid_count > 0:
        valid_data = arctan_zscores[valid_values]
        logger.info("\n📊 Arctan Z-Score Statistics (Fixed Std):")
        logger.info(f"  Valid values: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
        logger.info(f"  Min:          {np.min(valid_data):.4f}")
        logger.info(f"  P01:          {np.percentile(valid_data, 1):.4f}")
        logger.info(f"  Q25:          {np.percentile(valid_data, 25):.4f}")
        logger.info(f"  Median:       {np.median(valid_data):.4f}")
        logger.info(f"  Q75:          {np.percentile(valid_data, 75):.4f}")
        logger.info(f"  P99:          {np.percentile(valid_data, 99):.4f}")
        logger.info(f"  Max:          {np.max(valid_data):.4f}")
        logger.info(f"  Mean:         {np.mean(valid_data):.4f}")
        logger.info(f"  Std:          {np.std(valid_data):.4f}")

        # Distribution analysis
        logger.info(f"\n📈 Distribution Ranges:")
        logger.info(f"  < -0.8 (extreme low):    {np.sum(valid_data < -0.8):,} ({np.sum(valid_data < -0.8)/valid_count*100:.2f}%)")
        logger.info(f"  -0.8 to -0.4:            {np.sum((valid_data >= -0.8) & (valid_data < -0.4)):,} ({np.sum((valid_data >= -0.8) & (valid_data < -0.4))/valid_count*100:.2f}%)")
        logger.info(f"  -0.4 to +0.4 (normal):   {np.sum((valid_data >= -0.4) & (valid_data <= 0.4)):,} ({np.sum((valid_data >= -0.4) & (valid_data <= 0.4))/valid_count*100:.2f}%)")
        logger.info(f"  +0.4 to +0.8:            {np.sum((valid_data > 0.4) & (valid_data <= 0.8)):,} ({np.sum((valid_data > 0.4) & (valid_data <= 0.8))/valid_count*100:.2f}%)")
        logger.info(f"  > +0.8 (extreme high):   {np.sum(valid_data > 0.8):,} ({np.sum(valid_data > 0.8)/valid_count*100:.2f}%)")

        # Raw z-score analysis
        raw_zscores = zscores[~np.isnan(zscores)]
        logger.info(f"\n📉 Raw Z-Score Distribution (before arctan):")
        logger.info(f"  |z| > 3 (extreme):       {np.sum(np.abs(raw_zscores) > 3):,} ({np.sum(np.abs(raw_zscores) > 3)/len(raw_zscores)*100:.2f}%)")
        logger.info(f"  |z| > 2 (very high):     {np.sum(np.abs(raw_zscores) > 2):,} ({np.sum(np.abs(raw_zscores) > 2)/len(raw_zscores)*100:.2f}%)")
        logger.info(f"  |z| < 1 (normal):        {np.sum(np.abs(raw_zscores) < 1):,} ({np.sum(np.abs(raw_zscores) < 1)/len(raw_zscores)*100:.2f}%)")

    # Update database
    logger.info("\nUpdating database...")
    conn.register('zscore_data', df[['bar_index', zscore_column]])

    conn.execute(f"""
        UPDATE master
        SET {zscore_column} = zscore_data.{zscore_column}
        FROM zscore_data
        WHERE master.bar_index = zscore_data.bar_index
    """)

    logger.info("✅ Database updated successfully")

    return valid_count, fixed_std


def main():
    """Main entry point."""
    logger.info("="*70)
    logger.info("Adding Arctan Z-Score Feature with Fixed Std")
    logger.info("="*70)
    logger.info("\n💡 Using FIXED STD from training data (70%)")
    logger.info("   → Detects TRUE regime changes")
    logger.info("   → Avoids false extremes during consolidation")
    logger.info("   → Consistent baseline across all market conditions\n")

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    try:
        # Calculate and add feature
        valid_count, fixed_std = add_arctan_zscore_feature(
            conn,
            source_column="h1_swing_range_position",
            zscore_column="h1_swing_range_position_zsarctan",
            window=500,
            recalculate_std=False,  # Set to True to force recalculation
            description="Price position within H1 swing range (0=at low, 1=at high)"
        )

        logger.info(f"\n✅ Successfully added h1_swing_range_position_zsarctan column")
        logger.info(f"📊 Calculated {valid_count:,} valid values")
        logger.info(f"📏 Fixed std used: {fixed_std:.6f}")
        logger.info(f"⚙️  Config saved to: {CONFIG_PATH}")

    except Exception as e:
        logger.error(f"Error processing feature: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    logger.info("\n✅ Feature calculation complete!")
    logger.info("📊 Column added: h1_swing_range_position_zsarctan")
    logger.info("💡 Interpretation (using fixed std baseline):")
    logger.info("  -1.0 to -0.8: EXTREME regime divergence (far below historical)")
    logger.info("  -0.8 to -0.4: Significant divergence (potential mean reversion)")
    logger.info("  -0.4 to +0.4: Normal variation (within historical range)")
    logger.info("  +0.4 to +0.8: Significant divergence (potential mean reversion)")
    logger.info("  +0.8 to +1.0: EXTREME regime divergence (far above historical)")
    logger.info("\n🎯 Key Benefit: Detects TRUE breakouts from consolidation")
    logger.info("   (not false alarms from adaptive rolling std)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())