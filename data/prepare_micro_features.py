#!/usr/bin/env python3
"""
Prepare micro features table in DuckDB for Micro Stochastic MuZero training.

Creates a table with:
- 3 metadata columns (timestamp, bar_index, close)
- 160 technical indicator columns (5 indicators × 32 lags)
- 128 cyclical time columns (4 features × 32 lags)
- 6 position features (current only, no lags)
Total: 297 columns
"""

import duckdb
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MicroFeatureBuilder:
    """Build micro features table from master.duckdb"""

    def __init__(
        self,
        master_db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
        micro_db_path: str = "/home/aharon/projects/new_swt/data/micro_features.duckdb",
        lag_window: int = 32
    ):
        """
        Initialize micro feature builder.

        Args:
            master_db_path: Path to master database with clean data
            micro_db_path: Path to output micro features database
            lag_window: Number of lagged values per feature (32)
        """
        self.master_path = master_db_path
        self.micro_path = micro_db_path
        self.lag_window = lag_window

        # Feature groups
        self.technical_features = [
            'position_in_range_60',
            'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range',
            'min_max_scaled_momentum_5'
        ]

        self.cyclical_features = [
            'dow_cos_final',
            'dow_sin_final',
            'hour_cos_final',
            'hour_sin_final'
        ]

        # Position features (no lags, current only)
        self.position_features = [
            'position_side',
            'position_pips',
            'bars_since_entry',
            'pips_from_peak',
            'max_drawdown_pips',
            'accumulated_dd'
        ]

        logger.info(f"Initialized MicroFeatureBuilder")
        logger.info(f"  Master DB: {master_db_path}")
        logger.info(f"  Output DB: {micro_db_path}")
        logger.info(f"  Lag window: {lag_window}")
        logger.info(f"  Total features: 297")

    def add_price_change_feature(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Add price_change_pips feature to master table if not exists.
        Formula: tanh((close - prev_close) * 100 / 10)
        """
        logger.info("Adding price_change_pips feature...")

        # Check if column exists
        existing_cols = [row[0] for row in conn.execute("DESCRIBE master").fetchall()]

        if 'price_change_pips' not in existing_cols:
            # Add column
            conn.execute("ALTER TABLE master ADD COLUMN price_change_pips DOUBLE")

            # Calculate price change using a subquery
            conn.execute("""
                UPDATE master
                SET price_change_pips = (
                    SELECT TANH((m1.close - m2.close) * 100 / 10)
                    FROM master m1, master m2
                    WHERE m1.bar_index = master.bar_index
                    AND m2.bar_index = master.bar_index - 1
                )
                WHERE bar_index > 0
            """)

            # Handle first row
            conn.execute("""
                UPDATE master
                SET price_change_pips = 0.0
                WHERE bar_index = 0
            """)

            logger.info("  Added and populated price_change_pips")
        else:
            logger.info("  price_change_pips already exists")

        # Update technical features list
        if 'price_change_pips' not in self.technical_features:
            self.technical_features.append('price_change_pips')

    def simulate_position_features(self, conn: duckdb.DuckDBPyConnection) -> None:
        """
        Simulate position features for training data.
        In production, these come from actual trading environment.
        """
        logger.info("Simulating position features...")

        # For now, create random but realistic position features
        # In production, these would come from backtesting with actual position tracking

        position_cols = []
        for feat in self.position_features:
            if feat not in [row[0] for row in conn.execute("DESCRIBE master").fetchall()]:
                conn.execute(f"ALTER TABLE master ADD COLUMN {feat} DOUBLE")
                position_cols.append(feat)

        if position_cols:
            # Simulate realistic trading patterns
            total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]

            # Create position sequences (alternating flat/long/short periods)
            conn.execute("""
                UPDATE master
                SET
                    -- Random position side with 60% flat, 20% long, 20% short
                    position_side = CASE
                        WHEN RANDOM() < 0.6 THEN 0.0
                        WHEN RANDOM() < 0.5 THEN 1.0
                        ELSE -1.0
                    END,

                    -- Simulated P&L (small random walk when in position)
                    position_pips = TANH((RANDOM() * 200 - 100) / 100),

                    -- Random bars in position
                    bars_since_entry = TANH((RANDOM() * 120) / 100),

                    -- Distance from peak (usually negative)
                    pips_from_peak = TANH((RANDOM() * -50) / 100),

                    -- Max drawdown (always negative or zero)
                    max_drawdown_pips = TANH((RANDOM() * -30) / 100),

                    -- Accumulated drawdown (positive value)
                    accumulated_dd = TANH((RANDOM() * 100) / 100)
            """)

            # Set all position features to 0 when flat
            conn.execute("""
                UPDATE master
                SET
                    position_pips = 0.0,
                    bars_since_entry = 0.0,
                    pips_from_peak = 0.0,
                    max_drawdown_pips = 0.0,
                    accumulated_dd = 0.0
                WHERE position_side = 0.0
            """)

            logger.info(f"  Simulated {len(position_cols)} position features")

    def create_micro_table_schema(self, conn: duckdb.DuckDBPyConnection) -> str:
        """
        Generate CREATE TABLE statement for micro_features.

        Returns:
            SQL CREATE TABLE statement
        """
        columns = []

        # Metadata columns
        columns.append("timestamp TIMESTAMP")
        columns.append("bar_index BIGINT")
        columns.append("close DOUBLE")

        # Technical indicators with lags (5 × 32 = 160 columns)
        for feature in self.technical_features:
            for lag in range(self.lag_window):
                columns.append(f"{feature}_{lag} DOUBLE")

        # Cyclical features with lags (4 × 32 = 128 columns)
        for feature in self.cyclical_features:
            for lag in range(self.lag_window):
                columns.append(f"{feature}_{lag} DOUBLE")

        # Position features (current only, no lags)
        for feature in self.position_features:
            columns.append(f"{feature} DOUBLE")

        create_sql = f"""
        CREATE TABLE micro_features (
            {', '.join(columns)}
        )
        """

        logger.info(f"Schema has {len(columns)} columns")
        return create_sql

    def populate_micro_features(
        self,
        master_conn: duckdb.DuckDBPyConnection,
        micro_conn: duckdb.DuckDBPyConnection,
        batch_size: int = 10000
    ) -> None:
        """
        Populate micro_features table with lagged values.
        """
        logger.info("Populating micro_features table...")

        # Get total rows
        total_rows = master_conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]

        # We need at least lag_window rows of history
        start_idx = self.lag_window - 1

        # Process in batches
        rows_processed = 0

        for batch_start in range(start_idx, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_time = time.time()

            logger.info(f"Processing rows {batch_start:,} to {batch_end:,}...")

            # Build SELECT statement with LAG functions
            select_parts = [
                "timestamp",
                "bar_index",
                "close"
            ]

            # Add lagged technical features
            for feature in self.technical_features:
                for lag in range(self.lag_window):
                    if lag == 0:
                        select_parts.append(f"{feature} AS {feature}_{lag}")
                    else:
                        select_parts.append(
                            f"LAG({feature}, {lag}) OVER (ORDER BY timestamp) AS {feature}_{lag}"
                        )

            # Add lagged cyclical features
            for feature in self.cyclical_features:
                for lag in range(self.lag_window):
                    if lag == 0:
                        select_parts.append(f"{feature} AS {feature}_{lag}")
                    else:
                        select_parts.append(
                            f"LAG({feature}, {lag}) OVER (ORDER BY timestamp) AS {feature}_{lag}"
                        )

            # Add current position features (no lags)
            for feature in self.position_features:
                select_parts.append(feature)

            # Build and execute query
            query = f"""
            INSERT INTO micro_features
            SELECT {', '.join(select_parts)}
            FROM master
            WHERE bar_index >= {batch_start} AND bar_index < {batch_end}
            ORDER BY timestamp
            """

            micro_conn.execute(query)

            rows_processed += (batch_end - batch_start)
            elapsed = time.time() - batch_time
            logger.info(f"  Inserted {batch_end - batch_start} rows in {elapsed:.1f}s")

        logger.info(f"Populated {rows_processed:,} rows total")

    def build(self) -> None:
        """
        Main build process.
        """
        start_time = time.time()

        try:
            # Connect to master database
            logger.info("Connecting to master database...")
            master_conn = duckdb.connect(self.master_path)

            # Verify master_clean exists (no NULLs)
            tables = master_conn.execute("SHOW TABLES").fetchall()
            if ('master_clean',) in tables:
                logger.info("Using master_clean table (no NULLs)")
                # Rename for consistency
                master_conn.execute("DROP TABLE IF EXISTS master_backup")
                master_conn.execute("ALTER TABLE master RENAME TO master_backup")
                master_conn.execute("ALTER TABLE master_clean RENAME TO master")

            # Add price_change_pips feature
            self.add_price_change_feature(master_conn)

            # Simulate position features (in production, these come from environment)
            self.simulate_position_features(master_conn)

            # Create micro features database
            logger.info(f"Creating micro features database...")
            micro_conn = duckdb.connect(self.micro_path)

            # Drop existing table
            micro_conn.execute("DROP TABLE IF EXISTS micro_features")

            # Create new table
            schema_sql = self.create_micro_table_schema(micro_conn)
            micro_conn.execute(schema_sql)

            # Close master connection before attaching to micro
            master_conn.close()

            # Attach master database to micro connection
            logger.info("Attaching master database...")
            micro_conn.execute(f"ATTACH '{self.master_path}' AS master_db")

            # Build INSERT query with all features
            logger.info("Building feature extraction query...")

            # Start with metadata
            select_cols = ["m.timestamp", "m.bar_index", "m.close"]

            # Add lagged features
            for feature in self.technical_features:
                for lag in range(self.lag_window):
                    if lag == 0:
                        select_cols.append(f"m.{feature} AS {feature}_{lag}")
                    else:
                        select_cols.append(
                            f"LAG(m.{feature}, {lag}) OVER (ORDER BY m.timestamp) AS {feature}_{lag}"
                        )

            for feature in self.cyclical_features:
                for lag in range(self.lag_window):
                    if lag == 0:
                        select_cols.append(f"m.{feature} AS {feature}_{lag}")
                    else:
                        select_cols.append(
                            f"LAG(m.{feature}, {lag}) OVER (ORDER BY m.timestamp) AS {feature}_{lag}"
                        )

            # Add position features (no lags)
            for feature in self.position_features:
                select_cols.append(f"m.{feature}")

            # Execute INSERT
            logger.info("Populating micro_features table (this may take a while)...")
            insert_query = f"""
            INSERT INTO micro_features
            SELECT {', '.join(select_cols)}
            FROM master_db.master m
            WHERE m.bar_index >= {self.lag_window - 1}
            ORDER BY m.timestamp
            """

            micro_conn.execute(insert_query)

            # Get statistics
            total_rows = micro_conn.execute("SELECT COUNT(*) FROM micro_features").fetchone()[0]
            total_cols = len(micro_conn.execute("DESCRIBE micro_features").fetchall())

            # Verify no NULLs in critical columns
            logger.info("Verifying data quality...")
            null_check_features = self.technical_features[:2] + self.cyclical_features[:2]
            for feature in null_check_features:
                null_count = micro_conn.execute(
                    f"SELECT COUNT(*) FROM micro_features WHERE {feature}_0 IS NULL"
                ).fetchone()[0]
                if null_count > 0:
                    logger.warning(f"  {feature}_0 has {null_count} NULLs")

            # Summary statistics
            logger.info("\n" + "="*60)
            logger.info("✅ MICRO FEATURES TABLE CREATED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"  Database: {self.micro_path}")
            logger.info(f"  Total rows: {total_rows:,}")
            logger.info(f"  Total columns: {total_cols}")
            logger.info(f"  Features breakdown:")
            logger.info(f"    - Metadata: 3 columns")
            logger.info(f"    - Technical (with lags): {len(self.technical_features)} × {self.lag_window} = {len(self.technical_features) * self.lag_window} columns")
            logger.info(f"    - Cyclical (with lags): {len(self.cyclical_features)} × {self.lag_window} = {len(self.cyclical_features) * self.lag_window} columns")
            logger.info(f"    - Position (current): {len(self.position_features)} columns")
            logger.info(f"  Time taken: {time.time() - start_time:.1f} seconds")

            # Close connections
            micro_conn.close()

        except Exception as e:
            logger.error(f"Failed to build micro features: {e}")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build micro features table for training")
    parser.add_argument(
        "--master-db",
        default="/home/aharon/projects/new_swt/data/master.duckdb",
        help="Path to master database"
    )
    parser.add_argument(
        "--output-db",
        default="/home/aharon/projects/new_swt/data/micro_features.duckdb",
        help="Path to output micro features database"
    )
    parser.add_argument(
        "--lag-window",
        type=int,
        default=32,
        help="Number of lag timesteps per feature"
    )

    args = parser.parse_args()

    builder = MicroFeatureBuilder(
        master_db_path=args.master_db,
        micro_db_path=args.output_db,
        lag_window=args.lag_window
    )

    builder.build()