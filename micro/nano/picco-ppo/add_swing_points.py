#!/usr/bin/env python3
"""
Optimized version: Add swing high/low point columns to master table.
This version only adds the boolean swing detection columns for faster execution.
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


def detect_swings_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect swing highs and lows using vectorized operations.
    Simple 3-bar pattern:
    - Swing high: h[i] > h[i-1] and h[i] > h[i+1]
    - Swing low: l[i] < l[i-1] and l[i] < l[i+1]
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    # Initialize arrays
    swing_high_m1 = np.zeros(n, dtype=bool)
    swing_low_m1 = np.zeros(n, dtype=bool)

    # Vectorized swing detection (for bars 1 to n-2)
    # Swing high: current bar is higher than both neighbors
    swing_high_m1[1:-1] = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])

    # Swing low: current bar is lower than both neighbors
    swing_low_m1[1:-1] = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])

    df['swing_high_m1'] = swing_high_m1
    df['swing_low_m1'] = swing_low_m1

    return df


def detect_h1_swings_from_m1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect H1 swings from M1 swing points.
    Groups M1 swings by hour and picks highest/lowest per hour.
    """
    # Create hour column
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('h')

    # Initialize H1 swing columns
    df['swing_high_h1'] = False
    df['swing_low_h1'] = False

    # Process each hour
    logger.info("  Processing hourly groups...")
    total_hours = df['hour'].nunique()
    processed = 0

    for hour, group in df.groupby('hour'):
        # Find M1 swings within this hour
        m1_highs = group[group['swing_high_m1'] == True]
        m1_lows = group[group['swing_low_m1'] == True]

        # Mark highest M1 swing high as H1 swing high
        if len(m1_highs) > 0:
            max_high_idx = m1_highs['high'].idxmax()
            df.loc[max_high_idx, 'swing_high_h1'] = True

        # Mark lowest M1 swing low as H1 swing low
        if len(m1_lows) > 0:
            min_low_idx = m1_lows['low'].idxmin()
            df.loc[min_low_idx, 'swing_low_h1'] = True

        processed += 1
        if processed % 1000 == 0:
            logger.info(f"    Processed {processed}/{total_hours} hours")

    # Drop temporary hour column
    df = df.drop('hour', axis=1)

    return df


def add_swing_columns(conn: duckdb.DuckDBPyConnection):
    """
    Add swing detection columns to master table.
    """
    # Get total row count
    total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    logger.info(f"Total rows to process: {total_rows:,}")

    # Add columns if they don't exist
    logger.info("Adding swing point columns to master table...")

    new_columns = [
        ('swing_high_m1', 'BOOLEAN'),
        ('swing_low_m1', 'BOOLEAN'),
        ('swing_high_h1', 'BOOLEAN'),
        ('swing_low_h1', 'BOOLEAN')
    ]

    for col_name, col_type in new_columns:
        try:
            conn.execute(f"ALTER TABLE master ADD COLUMN {col_name} {col_type}")
            logger.info(f"  Added column: {col_name} ({col_type})")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"  Column {col_name} already exists, will update")
                # Set to NULL first
                conn.execute(f"UPDATE master SET {col_name} = NULL")
            else:
                raise

    # Fetch data
    logger.info("Fetching data for swing detection...")
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
    logger.info("Detecting H1 swing points from M1 swings...")
    df = detect_h1_swings_from_m1(df)
    logger.info(f"  Found {df['swing_high_h1'].sum():,} H1 swing highs")
    logger.info(f"  Found {df['swing_low_h1'].sum():,} H1 swing lows")

    # Update database
    logger.info("Updating database with swing points...")

    # Create temporary table
    conn.register('swing_data', df[['bar_index', 'swing_high_m1', 'swing_low_m1',
                                     'swing_high_h1', 'swing_low_h1']])

    # Update all columns at once
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

    logger.info("✅ Database updated successfully")

    return len(df)


def get_last_two_swings(conn: duckdb.DuckDBPyConnection,
                         as_of_bar: int = None) -> dict:
    """
    Get the last two swing highs and lows for both M1 and H1 timeframes.

    Args:
        conn: DuckDB connection
        as_of_bar: Bar index to look back from (None = latest bar)

    Returns:
        Dictionary with swing information:
        {
            'M1': {
                'highs': [(bar_index, price), (bar_index, price)],
                'lows': [(bar_index, price), (bar_index, price)]
            },
            'H1': {
                'highs': [(bar_index, price), (bar_index, price)],
                'lows': [(bar_index, price), (bar_index, price)]
            }
        }
    """

    # If no specific bar given, use the latest
    if as_of_bar is None:
        as_of_bar = conn.execute("SELECT MAX(bar_index) FROM master").fetchone()[0]

    result = {
        'M1': {'highs': [], 'lows': []},
        'H1': {'highs': [], 'lows': []}
    }

    # Get last 2 M1 swing highs
    m1_highs = conn.execute("""
        SELECT bar_index, high
        FROM master
        WHERE swing_high_m1 = true AND bar_index < ?
        ORDER BY bar_index DESC
        LIMIT 2
    """, [as_of_bar]).fetchall()

    result['M1']['highs'] = [(idx, round(price, 3)) for idx, price in reversed(m1_highs)]

    # Get last 2 M1 swing lows
    m1_lows = conn.execute("""
        SELECT bar_index, low
        FROM master
        WHERE swing_low_m1 = true AND bar_index < ?
        ORDER BY bar_index DESC
        LIMIT 2
    """, [as_of_bar]).fetchall()

    result['M1']['lows'] = [(idx, round(price, 3)) for idx, price in reversed(m1_lows)]

    # Get last 2 H1 swing highs
    h1_highs = conn.execute("""
        SELECT bar_index, high
        FROM master
        WHERE swing_high_h1 = true AND bar_index < ?
        ORDER BY bar_index DESC
        LIMIT 2
    """, [as_of_bar]).fetchall()

    result['H1']['highs'] = [(idx, round(price, 3)) for idx, price in reversed(h1_highs)]

    # Get last 2 H1 swing lows
    h1_lows = conn.execute("""
        SELECT bar_index, low
        FROM master
        WHERE swing_low_h1 = true AND bar_index < ?
        ORDER BY bar_index DESC
        LIMIT 2
    """, [as_of_bar]).fetchall()

    result['H1']['lows'] = [(idx, round(price, 3)) for idx, price in reversed(h1_lows)]

    return result


def get_swing_levels_for_trading(conn: duckdb.DuckDBPyConnection,
                                  current_bar: int = None) -> dict:
    """
    Get swing levels formatted for trading decisions (SL/TP).

    Args:
        conn: DuckDB connection
        current_bar: Current bar index (None = latest)

    Returns:
        Dictionary with trading levels:
        {
            'current_price': float,
            'current_bar': int,
            'M1_support': float,    # Most recent M1 swing low
            'M1_resistance': float,  # Most recent M1 swing high
            'H1_support': float,     # Most recent H1 swing low
            'H1_resistance': float,  # Most recent H1 swing high
            'M1_prev_support': float,    # Previous M1 swing low
            'M1_prev_resistance': float, # Previous M1 swing high
        }
    """

    # Get current bar info
    if current_bar is None:
        current_info = conn.execute("""
            SELECT bar_index, close
            FROM master
            ORDER BY bar_index DESC
            LIMIT 1
        """).fetchone()
        current_bar = current_info[0]
        current_price = current_info[1]
    else:
        current_price = conn.execute("""
            SELECT close FROM master WHERE bar_index = ?
        """, [current_bar]).fetchone()[0]

    # Get swing levels
    swings = get_last_two_swings(conn, current_bar)

    result = {
        'current_price': round(current_price, 3),
        'current_bar': current_bar,
    }

    # M1 levels
    if len(swings['M1']['highs']) > 0:
        result['M1_resistance'] = swings['M1']['highs'][-1][1]
        if len(swings['M1']['highs']) > 1:
            result['M1_prev_resistance'] = swings['M1']['highs'][-2][1]

    if len(swings['M1']['lows']) > 0:
        result['M1_support'] = swings['M1']['lows'][-1][1]
        if len(swings['M1']['lows']) > 1:
            result['M1_prev_support'] = swings['M1']['lows'][-2][1]

    # H1 levels
    if len(swings['H1']['highs']) > 0:
        result['H1_resistance'] = swings['H1']['highs'][-1][1]

    if len(swings['H1']['lows']) > 0:
        result['H1_support'] = swings['H1']['lows'][-1][1]

    return result


def verify_swing_points(conn: duckdb.DuckDBPyConnection):
    """
    Verify swing point detection results.
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION OF SWING POINTS")
    logger.info("="*60)

    # Get statistics
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            SUM(CASE WHEN swing_high_m1 THEN 1 ELSE 0 END) as m1_swing_highs,
            SUM(CASE WHEN swing_low_m1 THEN 1 ELSE 0 END) as m1_swing_lows,
            SUM(CASE WHEN swing_high_h1 THEN 1 ELSE 0 END) as h1_swing_highs,
            SUM(CASE WHEN swing_low_h1 THEN 1 ELSE 0 END) as h1_swing_lows
        FROM master
    """).fetchone()

    logger.info(f"Total rows: {stats[0]:,}")
    logger.info(f"\nM1 Swings:")
    logger.info(f"  Swing highs: {stats[1]:,} ({stats[1]/stats[0]*100:.2f}%)")
    logger.info(f"  Swing lows: {stats[2]:,} ({stats[2]/stats[0]*100:.2f}%)")

    logger.info(f"\nH1 Swings:")
    logger.info(f"  Swing highs: {stats[3]:,} ({stats[3]/stats[0]*100:.2f}%)")
    logger.info(f"  Swing lows: {stats[4]:,} ({stats[4]/stats[0]*100:.2f}%)")

    # Show sample of recent swing points
    logger.info("\n" + "-"*60)
    logger.info("SAMPLE M1 SWING POINTS (last 20):")

    m1_swings = conn.execute("""
        SELECT
            bar_index,
            timestamp,
            ROUND(high, 3) as high,
            ROUND(low, 3) as low,
            CASE
                WHEN swing_high_m1 THEN 'HIGH'
                WHEN swing_low_m1 THEN 'LOW'
            END as swing_type
        FROM master
        WHERE swing_high_m1 OR swing_low_m1
        ORDER BY bar_index DESC
        LIMIT 20
    """).fetch_df()

    if len(m1_swings) > 0:
        print(m1_swings.to_string(index=False))

    logger.info("\n" + "-"*60)
    logger.info("SAMPLE H1 SWING POINTS (last 10):")

    h1_swings = conn.execute("""
        SELECT
            bar_index,
            timestamp,
            ROUND(high, 3) as high,
            ROUND(low, 3) as low,
            CASE
                WHEN swing_high_h1 THEN 'HIGH'
                WHEN swing_low_h1 THEN 'LOW'
            END as swing_type,
            DATE_TRUNC('hour', timestamp) as hour
        FROM master
        WHERE swing_high_h1 OR swing_low_h1
        ORDER BY bar_index DESC
        LIMIT 10
    """).fetch_df()

    if len(h1_swings) > 0:
        print(h1_swings.to_string(index=False))


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Adding Swing Point Detection to Master Table (Optimized)")
    logger.info("="*60)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    # Connect to database
    conn = duckdb.connect(str(DB_PATH))

    try:
        # Add swing columns
        rows_processed = add_swing_columns(conn)
        logger.info(f"\n✅ Successfully processed {rows_processed:,} rows")

        # Verify results
        verify_swing_points(conn)

        logger.info("\n✅ Swing point detection complete!")
        logger.info("📊 Added 4 swing detection columns to master table:")
        logger.info("  - swing_high_m1: M1 timeframe swing highs")
        logger.info("  - swing_low_m1: M1 timeframe swing lows")
        logger.info("  - swing_high_h1: H1 timeframe swing highs")
        logger.info("  - swing_low_h1: H1 timeframe swing lows")
        logger.info("💡 Use these for stop loss and take profit levels")

        # Demonstrate the new functions
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRATING NEW SWING QUERY FUNCTIONS")
        logger.info("="*60)

        # Get last two swings
        swings = get_last_two_swings(conn)
        logger.info("\n📊 Last Two Swings for Each Timeframe:")
        logger.info("-"*40)

        for timeframe in ['M1', 'H1']:
            logger.info(f"\n{timeframe} Timeframe:")

            if swings[timeframe]['highs']:
                logger.info(f"  Swing Highs:")
                for bar_idx, price in swings[timeframe]['highs']:
                    logger.info(f"    Bar {bar_idx}: {price}")
            else:
                logger.info(f"  Swing Highs: None found")

            if swings[timeframe]['lows']:
                logger.info(f"  Swing Lows:")
                for bar_idx, price in swings[timeframe]['lows']:
                    logger.info(f"    Bar {bar_idx}: {price}")
            else:
                logger.info(f"  Swing Lows: None found")

        # Get trading levels
        levels = get_swing_levels_for_trading(conn)
        logger.info("\n💹 Trading Levels (for SL/TP):")
        logger.info("-"*40)
        for key, value in levels.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error processing swing points: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())