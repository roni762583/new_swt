#!/usr/bin/env python3
"""
Handmade feature builder for populating c_1 through c_255 columns
with shifted close price values in DuckDB.
"""

import duckdb
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def discover_scattering_shape():
    """
    Discover the output shape of Scattering Transform features.
    Requires torch and kymatio to be installed.
    """
    try:
        import torch
        from kymatio.torch import Scattering1D
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install torch and kymatio or run in appropriate environment")
        return None

    # Parameters
    T = 256   # input length
    J = 6
    Q = 4

    scat = Scattering1D(J=J, Q=Q, shape=T)
    x = torch.randn(1, T)   # dummy input

    Sx = scat(x)
    print("Raw scattering output shape:", Sx.shape)

    # If time-averaged (common for feature vectors)
    features = Sx.mean(-1)
    print("Time-averaged feature shape:", features.shape)
    print(f"Number of WST features: {features.shape[1]}")

    # Additional info
    print("\nWST Configuration:")
    print(f"  Input length (T): {T}")
    print(f"  Scales (J): {J}")
    print(f"  Wavelets per octave (Q): {Q}")
    print(f"  Output features: {features.shape[1]}")

    return features.shape


def populate_close_lag_features(
    db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
    batch_size: int = 10
) -> None:
    """
    Populate c_1 through c_255 columns with lagged close price values.

    Each c_N column contains the close price from N periods ago.
    c_1 = close price from 1 period ago (previous close)
    c_2 = close price from 2 periods ago
    ...
    c_255 = close price from 255 periods ago

    Args:
        db_path: Path to the DuckDB database
        batch_size: Number of columns to update in each batch

    Raises:
        RuntimeError: If database operations fail
    """
    try:
        conn = duckdb.connect(db_path)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if ('master',) not in tables:
            raise RuntimeError(f"Table 'master' not found in {db_path}")

        logger.info(f"Connected to {db_path}")

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Total rows in master table: {total_rows:,}")

        # Get all data sorted by timestamp for proper lag calculation
        logger.info("Loading close prices for lag calculation...")
        close_data = conn.execute("""
            SELECT timestamp, close
            FROM master
            ORDER BY timestamp
        """).fetchdf()

        logger.info(f"Loaded {len(close_data):,} rows")

        # Process columns in batches
        total_columns = 255
        start_time = time.time()

        for batch_start in range(1, total_columns + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_columns)
            batch_time = time.time()

            logger.info(f"Processing columns c_{batch_start} to c_{batch_end}...")

            # Build the lag columns for this batch
            lag_select = ["timestamp"]
            for lag in range(batch_start, batch_end + 1):
                lag_select.append(f"LAG(close, {lag}) OVER (ORDER BY timestamp) as c_{lag}")

            lag_sql = f"""
            SELECT {', '.join(lag_select)}
            FROM master
            ORDER BY timestamp
            """

            # Get lag values
            lag_df = conn.execute(lag_sql).fetchdf()

            # Update each column in the batch
            for col_num in range(batch_start, batch_end + 1):
                col_name = f"c_{col_num}"

                # Create temporary table with the lag values
                conn.execute(f"""
                    CREATE OR REPLACE TEMP TABLE lag_update AS
                    SELECT timestamp, c_{col_num} as lag_value
                    FROM lag_df
                    WHERE c_{col_num} IS NOT NULL
                """)

                # Update master table
                conn.execute(f"""
                    UPDATE master
                    SET {col_name} = lag_update.lag_value
                    FROM lag_update
                    WHERE master.timestamp = lag_update.timestamp
                """)

                conn.execute("DROP TABLE IF EXISTS lag_update")

            # Verify the batch
            non_nulls = []
            for col_num in range(batch_start, min(batch_start + 5, batch_end + 1)):
                count = conn.execute(f"SELECT COUNT(*) FROM master WHERE c_{col_num} IS NOT NULL").fetchone()[0]
                non_nulls.append(f"c_{col_num}:{count:,}")

            elapsed = time.time() - batch_time
            logger.info(f"  Batch complete in {elapsed:.1f}s. Sample counts: {', '.join(non_nulls)}")

        total_elapsed = time.time() - start_time
        logger.info(f"\nAll columns processed in {total_elapsed:.1f} seconds")

        # Final verification
        logger.info("\nVerifying all columns...")
        for check_col in [1, 10, 50, 100, 200, 255]:
            non_null_count = conn.execute(f"""
                SELECT COUNT(*)
                FROM master
                WHERE c_{check_col} IS NOT NULL
            """).fetchone()[0]
            null_count = total_rows - non_null_count
            expected_nulls = min(check_col, total_rows)
            logger.info(f"  c_{check_col}: {non_null_count:,} values, {null_count:,} nulls "
                       f"(expected {expected_nulls:,} nulls)")

        # Show sample data
        logger.info("\nSample data verification (rows 255-260):")
        sample = conn.execute("""
            SELECT bar_index, close, c_1, c_2, c_5, c_10
            FROM master
            WHERE bar_index BETWEEN 255 AND 260
            ORDER BY bar_index
        """).fetchall()

        for row in sample:
            logger.info(f"  Row {row[0]}: close={row[1]:.5f}, "
                       f"c_1={row[2]:.5f if row[2] else 'NULL'}, "
                       f"c_2={row[3]:.5f if row[3] else 'NULL'}, "
                       f"c_5={row[4]:.5f if row[4] else 'NULL'}, "
                       f"c_10={row[5]:.5f if row[5] else 'NULL'}")

        # Verify lag logic with specific example
        logger.info("\nLag verification (checking c_1 = previous close):")
        verify = conn.execute("""
            SELECT
                a.bar_index,
                a.close as current_close,
                a.c_1 as lag_1_value,
                b.close as actual_prev_close,
                CASE WHEN a.c_1 = b.close THEN 'OK' ELSE 'MISMATCH' END as status
            FROM master a
            JOIN master b ON a.bar_index = b.bar_index + 1
            WHERE a.bar_index BETWEEN 100 AND 105
            ORDER BY a.bar_index
        """).fetchall()

        for row in verify:
            logger.info(f"  Row {row[0]}: c_1={row[2]:.5f if row[2] else 'NULL'}, "
                       f"prev_close={row[3]:.5f if row[3] else 'NULL'} [{row[4]}]")

        conn.close()
        logger.info("\n✅ Successfully populated all c_1 through c_255 columns with lagged close values")

    except Exception as e:
        logger.error(f"Failed to populate lag features: {e}")
        raise RuntimeError(f"Failed to populate lag features: {e}")


def populate_wst_features(
    db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
    batch_size: int = 10000,
    j_scale: int = 6,
    q_wavelets: int = 4
) -> None:
    """
    Populate wst_0 through wst_66 columns with Wavelet Scattering Transform features.

    Uses a sliding window of 256 close prices (current close + 255 lags) as input
    to generate 67 WST features for each row.

    Args:
        db_path: Path to the DuckDB database
        batch_size: Number of rows to process at a time
        j_scale: Number of scales (J parameter for kymatio)
        q_wavelets: Number of wavelets per octave (Q parameter for kymatio)

    Raises:
        RuntimeError: If database operations fail or dependencies missing
    """
    try:
        import torch
        from kymatio.torch import Scattering1D
        import numpy as np
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(f"Required modules not available: {e}. Run in Docker container.")

    try:
        conn = duckdb.connect(db_path)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if ('master',) not in tables:
            raise RuntimeError(f"Table 'master' not found in {db_path}")

        logger.info(f"Connected to {db_path}")

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Total rows in master table: {total_rows:,}")

        # Initialize Scattering Transform
        T = 256  # Input length (close + 255 lags)
        device = torch.device('cpu')
        scattering = Scattering1D(J=j_scale, Q=q_wavelets, shape=T).to(device)
        logger.info(f"Initialized WST with J={j_scale}, Q={q_wavelets}, T={T}")

        # Test to confirm feature count
        test_input = torch.randn(1, T)
        test_output = scattering(test_input).mean(-1)
        n_features = test_output.shape[1]
        logger.info(f"WST produces {n_features} features")

        if n_features != 67:
            logger.warning(f"Expected 67 features but got {n_features}")

        # Process in batches
        start_time = time.time()
        rows_processed = 0

        # We need at least 256 previous closes to compute WST
        min_required_row = 255

        for batch_start in range(min_required_row, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_time = time.time()

            logger.info(f"Processing rows {batch_start:,} to {batch_end:,}...")

            # Get data for this batch - need close and 255 lag columns
            lag_columns = ["close"] + [f"c_{i}" for i in range(1, 256)]
            query = f"""
            SELECT
                bar_index,
                timestamp,
                {', '.join(lag_columns)}
            FROM master
            WHERE bar_index >= {batch_start} AND bar_index < {batch_end}
            ORDER BY bar_index
            """

            batch_df = conn.execute(query).fetchdf()

            if len(batch_df) == 0:
                continue

            # Prepare input tensor (batch_size, 256)
            # Each row contains [close, c_1, c_2, ..., c_255]
            input_data = batch_df[lag_columns].values

            # Check for NaN values
            valid_mask = ~np.isnan(input_data).any(axis=1)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                logger.warning(f"  No valid rows in batch (all contain NaN)")
                continue

            # Process only valid rows
            valid_data = input_data[valid_mask]
            input_tensor = torch.FloatTensor(valid_data).to(device)

            # Apply Scattering Transform
            with torch.no_grad():
                scattering_output = scattering(input_tensor)
                # Average over time dimension to get feature vector
                wst_features = scattering_output.mean(dim=-1).cpu().numpy()

            # Create temporary table with WST features
            wst_df = pd.DataFrame(wst_features, columns=[f"wst_{i}" for i in range(n_features)])
            wst_df['bar_index'] = batch_df.iloc[valid_indices]['bar_index'].values

            # Register as temporary table
            conn.register('wst_temp', wst_df)

            # Update master table with WST features
            set_clauses = [f"wst_{i} = wst_temp.wst_{i}" for i in range(min(n_features, 67))]
            update_sql = f"""
            UPDATE master
            SET {', '.join(set_clauses)}
            FROM wst_temp
            WHERE master.bar_index = wst_temp.bar_index
            """

            conn.execute(update_sql)
            conn.unregister('wst_temp')

            rows_processed += len(valid_indices)
            elapsed = time.time() - batch_time
            logger.info(f"  Processed {len(valid_indices)} valid rows in {elapsed:.1f}s")

        total_elapsed = time.time() - start_time
        logger.info(f"\nProcessed {rows_processed:,} rows in {total_elapsed:.1f} seconds")

        # Verification
        logger.info("\nVerifying WST features...")
        for col in [0, 10, 30, 50, 66]:
            non_null = conn.execute(f"SELECT COUNT(*) FROM master WHERE wst_{col} IS NOT NULL").fetchone()[0]
            logger.info(f"  wst_{col}: {non_null:,} non-null values")

        # Show sample
        logger.info("\nSample WST values (row 1000):")
        sample = conn.execute("""
            SELECT wst_0, wst_1, wst_2, wst_10, wst_30, wst_50, wst_66
            FROM master
            WHERE bar_index = 1000
        """).fetchone()

        if sample and sample[0] is not None:
            for i, val in enumerate([0, 1, 2, 10, 30, 50, 66]):
                logger.info(f"  wst_{val}: {sample[i]:.6f}")

        # Check if any WST columns beyond 66 exist and warn
        if n_features < 70:
            logger.info(f"\nNote: wst_{n_features} through wst_69 remain NULL (WST only produces {n_features} features)")

        conn.close()
        logger.info(f"\n✅ Successfully populated wst_0 through wst_{n_features-1} with WST features")

    except Exception as e:
        logger.error(f"Failed to populate WST features: {e}")
        raise RuntimeError(f"Failed to populate WST features: {e}")


def populate_time_cyclical_features(
    db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
    batch_size: int = 100000
) -> None:
    """
    Populate dow_cos_final, dow_sin_final, hour_cos_final, hour_sin_final columns.

    Uses cyclical encoding for time features following the MuZero pipeline:
    - 120-hour trading week (Sunday 22:00 UTC to Friday 22:00 UTC)
    - 24-hour daily cycle
    - Aligned to broker timezone (Eastern Time)

    Args:
        db_path: Path to the DuckDB database
        batch_size: Number of rows to process at a time

    Raises:
        RuntimeError: If database operations fail
    """
    import numpy as np
    from datetime import datetime

    try:
        conn = duckdb.connect(db_path)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if ('master',) not in tables:
            raise RuntimeError(f"Table 'master' not found in {db_path}")

        logger.info(f"Connected to {db_path}")

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Total rows in master table: {total_rows:,}")

        # Check if columns exist, if not create them
        columns_to_check = ['dow_cos_final', 'dow_sin_final', 'hour_cos_final', 'hour_sin_final']
        existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(master)").fetchall()]

        for col in columns_to_check:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE master ADD COLUMN {col} DOUBLE")
                logger.info(f"Added column {col}")

        # Process in batches
        start_time = time.time()
        rows_processed = 0

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_time = time.time()

            logger.info(f"Processing rows {batch_start:,} to {batch_end:,}...")

            # Get timestamps for this batch
            query = f"""
            SELECT bar_index, timestamp
            FROM master
            WHERE bar_index >= {batch_start} AND bar_index < {batch_end}
            ORDER BY bar_index
            """

            batch_df = conn.execute(query).fetchdf()

            if len(batch_df) == 0:
                continue

            # Calculate cyclical time features
            time_features = []
            for _, row in batch_df.iterrows():
                timestamp = row['timestamp']

                # Calculate hours since trading week start (Sunday 17:00 ET = 22:00 UTC)
                # Trading week runs Sunday 22:00 UTC to Friday 22:00 UTC (120 hours)
                dow = timestamp.weekday()  # 0=Monday, 6=Sunday
                hour_utc = timestamp.hour + timestamp.minute / 60.0

                if dow == 6:  # Sunday
                    if hour_utc >= 22:  # After market open (22:00 UTC = 17:00 ET)
                        hours_since_trading_start = hour_utc - 22
                    else:  # Before market open - map to previous Friday
                        hours_since_trading_start = (hour_utc + 24 - 22) + 96  # Previous week + current hours
                elif dow <= 4:  # Monday(0) through Friday(4)
                    base_hours = 2 + (dow * 24)  # 2 hours Sunday + full days
                    hours_since_trading_start = base_hours + hour_utc
                else:  # Saturday (5) - market closed, map to Friday 22:00 UTC
                    hours_since_trading_start = 120  # End of trading week

                # Ensure we stay within 120-hour cycle
                hours_since_trading_start = hours_since_trading_start % 120

                # 120-hour trading week cycle encoding
                dow_cos = np.cos(2 * np.pi * hours_since_trading_start / 120)
                dow_sin = np.sin(2 * np.pi * hours_since_trading_start / 120)

                # 24-hour daily cycle encoding
                hour_cos = np.cos(2 * np.pi * hour_utc / 24)
                hour_sin = np.sin(2 * np.pi * hour_utc / 24)

                time_features.append({
                    'bar_index': row['bar_index'],
                    'dow_cos_final': dow_cos,
                    'dow_sin_final': dow_sin,
                    'hour_cos_final': hour_cos,
                    'hour_sin_final': hour_sin
                })

            # Convert to DataFrame
            import pandas as pd
            time_df = pd.DataFrame(time_features)

            # Register as temporary table
            conn.register('time_temp', time_df)

            # Update master table
            update_sql = """
            UPDATE master
            SET
                dow_cos_final = time_temp.dow_cos_final,
                dow_sin_final = time_temp.dow_sin_final,
                hour_cos_final = time_temp.hour_cos_final,
                hour_sin_final = time_temp.hour_sin_final
            FROM time_temp
            WHERE master.bar_index = time_temp.bar_index
            """

            conn.execute(update_sql)
            conn.unregister('time_temp')

            rows_processed += len(batch_df)
            elapsed = time.time() - batch_time
            logger.info(f"  Processed {len(batch_df)} rows in {elapsed:.1f}s")

        total_elapsed = time.time() - start_time
        logger.info(f"\nProcessed {rows_processed:,} rows in {total_elapsed:.1f} seconds")

        # Verification
        logger.info("\nVerifying time features...")

        # Check for nulls
        for col in columns_to_check:
            non_null = conn.execute(f"SELECT COUNT(*) FROM master WHERE {col} IS NOT NULL").fetchone()[0]
            logger.info(f"  {col}: {non_null:,} non-null values")

        # Show sample values
        logger.info("\nSample time features (rows 1000-1005):")
        sample = conn.execute("""
            SELECT
                bar_index,
                timestamp,
                dow_cos_final,
                dow_sin_final,
                hour_cos_final,
                hour_sin_final
            FROM master
            WHERE bar_index BETWEEN 1000 AND 1005
            ORDER BY bar_index
        """).fetchall()

        for row in sample:
            ts = row[1]
            logger.info(f"  Row {row[0]} ({ts}):")
            logger.info(f"    dow_cos={row[2]:.4f}, dow_sin={row[3]:.4f}")
            logger.info(f"    hour_cos={row[4]:.4f}, hour_sin={row[5]:.4f}")

        # Check value ranges (should be between -1 and 1)
        logger.info("\nValue ranges:")
        for col in columns_to_check:
            min_val = conn.execute(f"SELECT MIN({col}) FROM master").fetchone()[0]
            max_val = conn.execute(f"SELECT MAX({col}) FROM master").fetchone()[0]
            logger.info(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")

        conn.close()
        logger.info("\n✅ Successfully populated cyclical time features")

    except Exception as e:
        logger.error(f"Failed to populate time features: {e}")
        raise RuntimeError(f"Failed to populate time features: {e}")


def populate_technical_indicators(
    db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
    batch_size: int = 50000,
    L: int = 60,  # long-term window
    S: int = 5    # short-term lag
) -> None:
    """
    Populate four technical indicator columns based on momentum and range analysis.

    Indicators:
    1. position_in_range_60: Where price sits in 60-bar high/low range (0-1)
    2. min_max_scaled_momentum_60: 60-bar momentum scaled to 0-1
    3. min_max_scaled_rolling_range: 60-bar volatility scaled to 0-1
    4. min_max_scaled_momentum_5: 5-bar momentum in 60-bar context

    Args:
        db_path: Path to the DuckDB database
        batch_size: Number of rows to process at a time
        L: Long-term window (default 60)
        S: Short-term lag (default 5)

    Raises:
        RuntimeError: If database operations fail
    """
    import numpy as np
    import pandas as pd

    eps = 1e-9  # Small epsilon to avoid division by zero

    try:
        conn = duckdb.connect(db_path)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if ('master',) not in tables:
            raise RuntimeError(f"Table 'master' not found in {db_path}")

        logger.info(f"Connected to {db_path}")

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Total rows in master table: {total_rows:,}")

        # Check if columns exist, if not create them
        indicator_columns = [
            'position_in_range_60',
            'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range',
            'min_max_scaled_momentum_5'
        ]

        existing_cols = [row[0] for row in conn.execute("DESCRIBE master").fetchall()]

        for col in indicator_columns:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE master ADD COLUMN {col} DOUBLE")
                logger.info(f"Added column {col}")

        # Load all data for proper rolling calculations
        logger.info("Loading price data for indicator calculation...")

        price_data = conn.execute("""
            SELECT bar_index, timestamp, close
            FROM master
            ORDER BY timestamp
        """).fetchdf()

        logger.info(f"Loaded {len(price_data):,} rows")

        # Calculate rolling statistics
        logger.info(f"Calculating rolling statistics with L={L}, S={S}...")

        # Rolling high/low/range over L periods
        price_data['rolling_high'] = price_data['close'].rolling(window=L, min_periods=1).max()
        price_data['rolling_low'] = price_data['close'].rolling(window=L, min_periods=1).min()
        price_data['rolling_range'] = price_data['rolling_high'] - price_data['rolling_low']

        # Momentum calculations
        price_data['momentum60'] = price_data['close'] - price_data['close'].shift(L)
        price_data['momentum5'] = price_data['close'] - price_data['close'].shift(S)

        # Rolling min/max for momentum60
        mom60_max = price_data['momentum60'].rolling(window=L, min_periods=1).max()
        mom60_min = price_data['momentum60'].rolling(window=L, min_periods=1).min()
        mom60_range = (mom60_max - mom60_min).clip(lower=eps)

        # Rolling min/max for rolling_range
        roll_range_max = price_data['rolling_range'].rolling(window=L, min_periods=1).max()
        roll_range_min = price_data['rolling_range'].rolling(window=L, min_periods=1).min()
        roll_range_rng = (roll_range_max - roll_range_min).clip(lower=eps)

        # Calculate indicators
        logger.info("Calculating technical indicators...")

        # 1. Position in Range (60-bar)
        price_data['position_in_range_60'] = (
            (price_data['close'] - price_data['rolling_low']) /
            price_data['rolling_range'].clip(lower=eps)
        )

        # 2. Min-Max Scaled Momentum (60-bar)
        price_data['min_max_scaled_momentum_60'] = (
            (price_data['momentum60'] - mom60_min) / mom60_range
        )

        # 3. Min-Max Scaled Rolling Range
        price_data['min_max_scaled_rolling_range'] = (
            (price_data['rolling_range'] - roll_range_min) / roll_range_rng
        )

        # 4. Min-Max Scaled Momentum (5-bar in 60-bar context)
        # Use 60-bar momentum range for scaling to see short-term in long-term context
        price_data['min_max_scaled_momentum_5'] = (
            (price_data['momentum5'] - price_data['momentum60'].rolling(S, min_periods=1).min()) /
            mom60_range
        )

        # Clip to [0, 1] range for safety (NaN values remain as NaN)
        for col in indicator_columns:
            price_data[col] = price_data[col].clip(0, 1)

        # Process in batches for database update
        start_time = time.time()
        rows_processed = 0

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_time = time.time()

            logger.info(f"Updating rows {batch_start:,} to {batch_end:,}...")

            # Get batch data
            batch_df = price_data.iloc[batch_start:batch_end][['bar_index'] + indicator_columns].copy()

            # Register as temporary table
            conn.register('indicators_temp', batch_df)

            # Update master table
            set_clauses = [f"{col} = indicators_temp.{col}" for col in indicator_columns]
            update_sql = f"""
            UPDATE master
            SET {', '.join(set_clauses)}
            FROM indicators_temp
            WHERE master.bar_index = indicators_temp.bar_index
            """

            conn.execute(update_sql)
            conn.unregister('indicators_temp')

            rows_processed += len(batch_df)
            elapsed = time.time() - batch_time
            logger.info(f"  Updated {len(batch_df)} rows in {elapsed:.1f}s")

        total_elapsed = time.time() - start_time
        logger.info(f"\nProcessed {rows_processed:,} rows in {total_elapsed:.1f} seconds")

        # Verification
        logger.info("\nVerifying technical indicators...")

        # Check for nulls and value ranges
        for col in indicator_columns:
            stats = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT({col}) as non_null,
                    MIN({col}) as min_val,
                    MAX({col}) as max_val,
                    AVG({col}) as mean_val
                FROM master
            """).fetchone()

            logger.info(f"  {col}:")
            logger.info(f"    Non-null: {stats[1]:,}/{stats[0]:,}")
            logger.info(f"    Range: [{stats[2]:.4f}, {stats[3]:.4f}]")
            logger.info(f"    Mean: {stats[4]:.4f}")

        # Show sample values
        logger.info("\nSample indicator values (rows 1000-1005):")
        sample = conn.execute(f"""
            SELECT
                bar_index,
                close,
                {', '.join(indicator_columns)}
            FROM master
            WHERE bar_index BETWEEN 1000 AND 1005
            ORDER BY bar_index
        """).fetchall()

        for row in sample:
            logger.info(f"  Row {row[0]} (close={row[1]:.5f}):")
            for i, col in enumerate(indicator_columns):
                logger.info(f"    {col}: {row[i+2]:.4f}")

        conn.close()
        logger.info("\n✅ Successfully populated technical indicator columns")

    except Exception as e:
        logger.error(f"Failed to populate technical indicators: {e}")
        raise RuntimeError(f"Failed to populate technical indicators: {e}")


def drop_null_rows(
    db_path: str = "/home/aharon/projects/new_swt/data/master.duckdb",
    create_clean_table: bool = False
) -> None:
    """
    Drop rows with NULL values from the master table and report what was dropped.

    Analyzes which rows contain NULLs, particularly checking if they're at the
    beginning of the dataframe (which is expected due to lag features and WST).

    Args:
        db_path: Path to the DuckDB database
        create_clean_table: If True, creates a new 'master_clean' table instead of modifying master

    Raises:
        RuntimeError: If database operations fail
    """
    import numpy as np
    import pandas as pd

    try:
        conn = duckdb.connect(db_path)

        # Verify table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        if ('master',) not in tables:
            raise RuntimeError(f"Table 'master' not found in {db_path}")

        logger.info(f"Connected to {db_path}")

        # Get total row count
        total_rows = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Total rows in master table: {total_rows:,}")

        # Get all columns except those we know can have NULLs
        cols = conn.execute("DESCRIBE master").fetchall()
        all_columns = [col[0] for col in cols]

        # Identify columns with NULLs
        logger.info("\nAnalyzing NULL values by column...")
        columns_with_nulls = []

        for col_name in all_columns:
            null_count = conn.execute(f"SELECT COUNT(*) FROM master WHERE {col_name} IS NULL").fetchone()[0]
            if null_count > 0:
                columns_with_nulls.append((col_name, null_count))

        # Report columns with NULLs
        if columns_with_nulls:
            logger.info(f"\nColumns with NULL values ({len(columns_with_nulls)} total):")
            for col_name, null_count in sorted(columns_with_nulls, key=lambda x: -x[1])[:10]:
                logger.info(f"  {col_name}: {null_count:,} NULLs ({null_count/total_rows*100:.2f}%)")
            if len(columns_with_nulls) > 10:
                logger.info(f"  ... and {len(columns_with_nulls)-10} more columns")

        # Find rows with ANY NULL values
        logger.info("\nIdentifying rows with NULL values...")

        # Get the bar_index of rows with NULLs
        null_rows_query = """
        SELECT bar_index, timestamp
        FROM master
        WHERE """ + " OR ".join([f"{col} IS NULL" for col, _ in columns_with_nulls]) + """
        ORDER BY bar_index
        """

        null_rows_df = conn.execute(null_rows_query).fetchdf()
        rows_with_nulls = len(null_rows_df)

        logger.info(f"\nRows with NULL values: {rows_with_nulls:,} ({rows_with_nulls/total_rows*100:.2f}%)")

        if rows_with_nulls > 0:
            # Analyze location of NULL rows
            first_null_idx = null_rows_df['bar_index'].iloc[0]
            last_null_idx = null_rows_df['bar_index'].iloc[-1]

            # Check if NULLs are at the beginning (expected pattern)
            expected_nulls_at_start = 255  # Maximum lag is c_255, WST needs 255 rows
            consecutive_nulls_at_start = 0

            for i in range(min(expected_nulls_at_start + 100, len(null_rows_df))):
                if null_rows_df['bar_index'].iloc[i] == i:
                    consecutive_nulls_at_start += 1
                else:
                    break

            logger.info(f"\nNULL rows distribution:")
            logger.info(f"  First NULL at bar_index: {first_null_idx}")
            logger.info(f"  Last NULL at bar_index: {last_null_idx}")
            logger.info(f"  Consecutive NULLs from start: {consecutive_nulls_at_start}")

            # Show sample of NULL row indices
            if rows_with_nulls <= 20:
                logger.info(f"  All NULL row indices: {null_rows_df['bar_index'].tolist()}")
            else:
                logger.info(f"  First 10 NULL indices: {null_rows_df['bar_index'].head(10).tolist()}")
                logger.info(f"  Last 10 NULL indices: {null_rows_df['bar_index'].tail(10).tolist()}")

            # Check if all NULLs are at the beginning (expected pattern)
            if consecutive_nulls_at_start == rows_with_nulls:
                logger.info(f"\n✅ All NULL rows are at the beginning (rows 0-{rows_with_nulls-1})")
                logger.info("This is EXPECTED due to lag features and WST requiring historical data")
            else:
                logger.warning(f"\n⚠️ NULL rows found beyond the beginning of the dataset")
                logger.warning("This may indicate data quality issues")

        # Create clean table or modify master
        if rows_with_nulls > 0:
            if create_clean_table:
                logger.info(f"\nCreating 'master_clean' table without NULL rows...")

                # Drop existing clean table if it exists
                conn.execute("DROP TABLE IF EXISTS master_clean")

                # Create new clean table
                conn.execute("""
                    CREATE TABLE master_clean AS
                    SELECT * FROM master
                    WHERE bar_index >= ?
                """, [consecutive_nulls_at_start])

                clean_rows = conn.execute("SELECT COUNT(*) FROM master_clean").fetchone()[0]
                logger.info(f"Created 'master_clean' with {clean_rows:,} rows")
                logger.info(f"Removed {total_rows - clean_rows:,} rows with NULLs")

                # Verify no NULLs remain
                verify_query = """
                SELECT COUNT(*)
                FROM master_clean
                WHERE """ + " OR ".join([f"{col} IS NULL" for col, _ in columns_with_nulls])

                remaining_nulls = conn.execute(verify_query).fetchone()[0]
                if remaining_nulls == 0:
                    logger.info("✅ Verified: master_clean contains no NULL values")
                else:
                    logger.warning(f"⚠️ Warning: {remaining_nulls} rows with NULLs remain in master_clean")
            else:
                logger.info(f"\nDeleting {rows_with_nulls:,} rows with NULL values from master table...")

                # Delete rows with NULLs
                delete_query = """
                DELETE FROM master
                WHERE """ + " OR ".join([f"{col} IS NULL" for col, _ in columns_with_nulls])

                conn.execute(delete_query)

                # Verify deletion
                new_total = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
                logger.info(f"Rows after deletion: {new_total:,}")
                logger.info(f"Removed: {total_rows - new_total:,} rows")

                # Update bar_index to be sequential
                logger.info("Updating bar_index to be sequential...")
                conn.execute("""
                    UPDATE master
                    SET bar_index = bar_index - ?
                """, [consecutive_nulls_at_start])

                logger.info("✅ Successfully removed NULL rows and updated indices")
        else:
            logger.info("\n✅ No NULL values found - table is already clean!")

        # Final statistics
        if create_clean_table and rows_with_nulls > 0:
            table_name = "master_clean"
        else:
            table_name = "master"

        logger.info(f"\nFinal {table_name} table statistics:")
        final_stats = conn.execute(f"""
            SELECT
                COUNT(*) as total_rows,
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                MIN(bar_index) as min_idx,
                MAX(bar_index) as max_idx
            FROM {table_name}
        """).fetchone()

        logger.info(f"  Total rows: {final_stats[0]:,}")
        logger.info(f"  Date range: {final_stats[1]} to {final_stats[2]}")
        logger.info(f"  Bar index range: {final_stats[3]} to {final_stats[4]}")

        conn.close()

    except Exception as e:
        logger.error(f"Failed to drop NULL rows: {e}")
        raise RuntimeError(f"Failed to drop NULL rows: {e}")


if __name__ == "__main__":
    import sys
    import numpy as np
    import pandas as pd

    if len(sys.argv) > 1:
        if sys.argv[1] == "discover":
            print("\nDiscovering Scattering Transform shape...")
            shape = discover_scattering_shape()
            print(f"Final feature shape: {shape}")
        elif sys.argv[1] == "wst":
            print("\nPopulating WST features...")
            populate_wst_features()
        elif sys.argv[1] == "time":
            print("\nPopulating cyclical time features...")
            populate_time_cyclical_features()
        elif sys.argv[1] == "indicators":
            print("\nPopulating technical indicators...")
            populate_technical_indicators()
        elif sys.argv[1] == "drop_nulls":
            print("\nDropping NULL rows from master table...")
            drop_null_rows(create_clean_table=False)
        elif sys.argv[1] == "create_clean":
            print("\nCreating master_clean table without NULL rows...")
            drop_null_rows(create_clean_table=True)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python handmade_feature_builder.py [discover|wst|time|indicators|drop_nulls|create_clean]")
    else:
        populate_close_lag_features()