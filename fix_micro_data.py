#!/usr/bin/env python3
"""
Fix missing price_change_pips data in micro_features database.
Calculate price_change_pips from close prices.
"""

import duckdb
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_price_change_pips():
    """Calculate and populate price_change_pips columns from close prices."""

    db_path = "/home/aharon/projects/new_swt/data/micro_features.duckdb"
    logger.info(f"Opening database: {db_path}")

    conn = duckdb.connect(db_path)

    # First, let's check if we have close price data
    result = conn.execute("""
        SELECT COUNT(*), MIN(close), MAX(close), AVG(close)
        FROM micro_features
    """).fetchone()

    total_rows, min_close, max_close, avg_close = result
    logger.info(f"Database has {total_rows:,} rows")
    logger.info(f"Close prices: min={min_close:.5f}, max={max_close:.5f}, avg={avg_close:.5f}")

    # Calculate price changes and update columns
    logger.info("Calculating price_change_pips from close prices...")

    # For GBPJPY, 1 pip = 0.01 (2 decimal places)
    pip_size = 0.01

    # First, get all close prices ordered by bar_index
    logger.info("Fetching close prices...")
    close_data = conn.execute("""
        SELECT bar_index, close
        FROM micro_features
        ORDER BY bar_index
    """).fetchdf()

    logger.info(f"Calculating price changes for {len(close_data):,} rows...")

    # Calculate price changes
    close_prices = close_data['close'].values
    price_changes = np.zeros((len(close_prices), 32))

    for i in range(1, len(close_prices)):
        for lag in range(min(32, i)):
            if lag == 0:
                # Current bar price change
                price_changes[i, lag] = (close_prices[i] - close_prices[i-1]) / pip_size
            elif i > lag:
                # Lagged price change
                price_changes[i, lag] = (close_prices[i-lag] - close_prices[i-lag-1]) / pip_size

    # Update database in batches
    batch_size = 10000
    for start_idx in range(0, len(close_data), batch_size):
        end_idx = min(start_idx + batch_size, len(close_data))
        logger.info(f"Updating rows {start_idx:,} to {end_idx:,}...")

        for lag in range(32):
            col_name = f"price_change_pips_{lag}"

            # Create temporary table with updates
            temp_data = close_data.iloc[start_idx:end_idx].copy()
            temp_data[col_name] = price_changes[start_idx:end_idx, lag]

            # Update using JOIN
            update_query = f"""
            UPDATE micro_features
            SET price_change_pips_{lag} = updates.new_value
            FROM (
                SELECT bar_index, {col_name} as new_value
                FROM (
                    SELECT bar_index,
                           CAST({' '.join(str(v) for v in temp_data[col_name].values)} AS DOUBLE[])[$$ + 1] as new_value
                    FROM (SELECT row_number() OVER () - 1 as $$, bar_index
                          FROM micro_features
                          WHERE bar_index >= {temp_data['bar_index'].min()}
                            AND bar_index <= {temp_data['bar_index'].max()})
                )
            ) as updates
            WHERE micro_features.bar_index = updates.bar_index
            """

            # Simpler approach - create temp table
            conn.execute(f"CREATE TEMP TABLE IF NOT EXISTS price_updates_{lag} (bar_index BIGINT, value DOUBLE)")
            conn.execute(f"DELETE FROM price_updates_{lag}")

            # Insert batch values
            for idx, row in temp_data.iterrows():
                conn.execute(f"""
                    INSERT INTO price_updates_{lag} VALUES ({row['bar_index']}, {price_changes[idx, lag]})
                """)

            # Update from temp table
            conn.execute(f"""
                UPDATE micro_features
                SET price_change_pips_{lag} = price_updates_{lag}.value
                FROM price_updates_{lag}
                WHERE micro_features.bar_index = price_updates_{lag}.bar_index
            """)

        logger.info(f"  Batch updated successfully")

    # Check the results
    logger.info("\nVerifying price_change_pips columns:")
    for lag in [0, 1, 2, 3, 4]:  # Check first 5 lags
        stats_query = f"""
        SELECT
            COUNT(*) as count,
            COUNT(CASE WHEN price_change_pips_{lag} IS NULL THEN 1 END) as nulls,
            MIN(price_change_pips_{lag}) as min_val,
            MAX(price_change_pips_{lag}) as max_val,
            AVG(price_change_pips_{lag}) as avg_val,
            STDDEV(price_change_pips_{lag}) as std_val
        FROM micro_features
        """
        stats = conn.execute(stats_query).fetchone()
        logger.info(f"  price_change_pips_{lag}:")
        logger.info(f"    NULLs: {stats[1]}/{stats[0]}")
        logger.info(f"    Range: [{stats[2]:.2f}, {stats[3]:.2f}] pips")
        logger.info(f"    Mean: {stats[4]:.4f}, Std: {stats[5]:.4f}")

    conn.close()
    logger.info("\n✅ Successfully fixed price_change_pips data!")
    logger.info("The micro training should now work properly.")

if __name__ == "__main__":
    fix_price_change_pips()