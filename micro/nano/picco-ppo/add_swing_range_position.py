#!/usr/bin/env python3
"""
Optimized: Add column to master table for price position within H1 swing range.
Formula: (close - last_h1_swing_low) / (last_h1_swing_high - last_h1_swing_low)
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")


def calculate_h1_swing_range_position_vectorized(conn: duckdb.DuckDBPyConnection):
    """
    Calculate price position within H1 swing range using vectorized operations.
    """

    # Get total rows
    total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    logger.info(f"Total rows to process: {total_rows:,}")

    # Add new column
    logger.info("Adding h1_swing_range_position column to master table...")
    try:
        conn.execute("ALTER TABLE master ADD COLUMN h1_swing_range_position DOUBLE")
        logger.info("  Added column: h1_swing_range_position (DOUBLE)")
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("  Column h1_swing_range_position already exists, will update")
            conn.execute("UPDATE master SET h1_swing_range_position = NULL")
        else:
            raise

    # Fetch all data ordered by bar_index
    logger.info("Fetching data from master table...")
    df = conn.execute("""
        SELECT
            bar_index,
            close,
            high,
            low,
            swing_high_h1,
            swing_low_h1
        FROM master
        ORDER BY bar_index
    """).fetch_df()

    logger.info(f"Loaded {len(df):,} rows")
    logger.info("Calculating H1 swing range positions using vectorized operations...")

    # Find indices where H1 swings occur
    h1_high_indices = df[df['swing_high_h1'] == True].index.tolist()
    h1_low_indices = df[df['swing_low_h1'] == True].index.tolist()

    # Initialize arrays
    n = len(df)
    last_h1_high = np.full(n, np.nan)
    last_h1_low = np.full(n, np.nan)

    # Forward fill H1 swing high values
    logger.info("  Processing H1 swing highs...")
    if h1_high_indices:
        for i in range(len(h1_high_indices)):
            start_idx = h1_high_indices[i]
            end_idx = h1_high_indices[i + 1] if i < len(h1_high_indices) - 1 else n
            last_h1_high[start_idx:end_idx] = df.iloc[start_idx]['high']

    # Forward fill H1 swing low values
    logger.info("  Processing H1 swing lows...")
    if h1_low_indices:
        for i in range(len(h1_low_indices)):
            start_idx = h1_low_indices[i]
            end_idx = h1_low_indices[i + 1] if i < len(h1_low_indices) - 1 else n
            last_h1_low[start_idx:end_idx] = df.iloc[start_idx]['low']

    # Calculate position in range (vectorized)
    logger.info("  Calculating range positions...")
    swing_range = last_h1_high - last_h1_low

    # Avoid division by zero
    valid_range = swing_range > 0
    h1_swing_range_position = np.full(n, np.nan)

    # Calculate position only where we have valid range
    h1_swing_range_position[valid_range] = (
        (df['close'].values[valid_range] - last_h1_low[valid_range]) /
        swing_range[valid_range]
    )

    # Add to dataframe
    df['h1_swing_range_position'] = h1_swing_range_position

    # Calculate statistics
    valid_positions = h1_swing_range_position[~np.isnan(h1_swing_range_position)]
    if len(valid_positions) > 0:
        logger.info("\n📊 H1 Swing Range Position Statistics:")
        logger.info(f"  Valid values: {len(valid_positions):,} ({len(valid_positions)/n*100:.1f}%)")
        logger.info(f"  Mean position: {np.mean(valid_positions):.3f}")
        logger.info(f"  Median position: {np.median(valid_positions):.3f}")
        logger.info(f"  Std deviation: {np.std(valid_positions):.3f}")
        logger.info(f"  Min position: {np.min(valid_positions):.3f}")
        logger.info(f"  Max position: {np.max(valid_positions):.3f}")

        # Distribution analysis
        logger.info(f"\n📍 Position Distribution:")
        logger.info(f"  Below range (<0): {np.sum(valid_positions < 0):,} ({np.sum(valid_positions < 0)/len(valid_positions)*100:.1f}%)")
        logger.info(f"  Lower quarter (0-0.25): {np.sum((valid_positions >= 0) & (valid_positions < 0.25)):,} ({np.sum((valid_positions >= 0) & (valid_positions < 0.25))/len(valid_positions)*100:.1f}%)")
        logger.info(f"  Lower middle (0.25-0.5): {np.sum((valid_positions >= 0.25) & (valid_positions < 0.5)):,} ({np.sum((valid_positions >= 0.25) & (valid_positions < 0.5))/len(valid_positions)*100:.1f}%)")
        logger.info(f"  Upper middle (0.5-0.75): {np.sum((valid_positions >= 0.5) & (valid_positions < 0.75)):,} ({np.sum((valid_positions >= 0.5) & (valid_positions < 0.75))/len(valid_positions)*100:.1f}%)")
        logger.info(f"  Upper quarter (0.75-1.0): {np.sum((valid_positions >= 0.75) & (valid_positions <= 1.0)):,} ({np.sum((valid_positions >= 0.75) & (valid_positions <= 1.0))/len(valid_positions)*100:.1f}%)")
        logger.info(f"  Above range (>1): {np.sum(valid_positions > 1):,} ({np.sum(valid_positions > 1)/len(valid_positions)*100:.1f}%)")

    # Update database
    logger.info("\nUpdating database with H1 swing range positions...")

    # Create temporary table with the calculated values
    conn.register('swing_range_data', df[['bar_index', 'h1_swing_range_position']])

    # Update master table
    conn.execute("""
        UPDATE master
        SET h1_swing_range_position = swing_range_data.h1_swing_range_position
        FROM swing_range_data
        WHERE master.bar_index = swing_range_data.bar_index
    """)

    logger.info("✅ Database updated successfully")

    return len(valid_positions)


def verify_swing_range_position(conn: duckdb.DuckDBPyConnection):
    """
    Verify the swing range position calculations with sample data.
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION OF H1 SWING RANGE POSITIONS")
    logger.info("="*60)

    # Get overall statistics
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(h1_swing_range_position) as valid_positions,
            AVG(h1_swing_range_position) as avg_position,
            MIN(h1_swing_range_position) as min_position,
            MAX(h1_swing_range_position) as max_position,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY h1_swing_range_position) as q25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY h1_swing_range_position) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY h1_swing_range_position) as q75
        FROM master
        WHERE h1_swing_range_position IS NOT NULL
    """).fetchone()

    logger.info(f"Total rows: {stats[0]:,}")
    logger.info(f"Valid positions: {stats[1]:,} ({stats[1]/stats[0]*100:.1f}%)")
    logger.info(f"\n📈 Position Statistics:")
    logger.info(f"  Min: {stats[3]:.3f}")
    logger.info(f"  Q25: {stats[5]:.3f}")
    logger.info(f"  Median: {stats[6]:.3f}")
    logger.info(f"  Mean: {stats[2]:.3f}")
    logger.info(f"  Q75: {stats[7]:.3f}")
    logger.info(f"  Max: {stats[4]:.3f}")

    # Show sample at different position levels
    logger.info("\n" + "-"*60)
    logger.info("SAMPLE: Recent bars at different positions")

    sample = conn.execute("""
        WITH recent_data AS (
            SELECT
                bar_index,
                timestamp,
                close,
                h1_swing_range_position,
                CASE
                    WHEN h1_swing_range_position < 0 THEN 'Below Range'
                    WHEN h1_swing_range_position <= 0.25 THEN 'Lower Quarter'
                    WHEN h1_swing_range_position <= 0.5 THEN 'Lower Middle'
                    WHEN h1_swing_range_position <= 0.75 THEN 'Upper Middle'
                    WHEN h1_swing_range_position <= 1.0 THEN 'Upper Quarter'
                    ELSE 'Above Range'
                END as position_zone
            FROM master
            WHERE h1_swing_range_position IS NOT NULL
            ORDER BY bar_index DESC
            LIMIT 20
        )
        SELECT
            bar_index,
            timestamp,
            ROUND(close, 3) as close,
            ROUND(h1_swing_range_position, 3) as position,
            position_zone
        FROM recent_data
        ORDER BY bar_index DESC
    """).fetch_df()

    if len(sample) > 0:
        print(sample.to_string(index=False))


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Adding H1 Swing Range Position to Master Table (Optimized)")
    logger.info("="*60)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    try:
        # Calculate and add swing range positions
        valid_count = calculate_h1_swing_range_position_vectorized(conn)

        logger.info(f"\n✅ Successfully added h1_swing_range_position column")
        logger.info(f"📊 Calculated {valid_count:,} valid positions")

        # Verify results
        verify_swing_range_position(conn)

    except Exception as e:
        logger.error(f"Error processing swing range positions: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    logger.info("\n✅ H1 Swing Range Position calculation complete!")
    logger.info("📊 Column added: h1_swing_range_position")
    logger.info("💡 Values interpretation:")
    logger.info("  - 0.0 = Price at H1 swing low")
    logger.info("  - 0.5 = Price at middle of H1 range")
    logger.info("  - 1.0 = Price at H1 swing high")
    logger.info("  - <0 = Price below H1 swing low (bearish breakout)")
    logger.info("  - >1 = Price above H1 swing high (bullish breakout)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())