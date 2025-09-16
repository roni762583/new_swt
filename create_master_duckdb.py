#!/usr/bin/env python3
"""
Create master DuckDB database for feature storage
Database location: /home/aharon/projects/new_swt/data/master.duckdb
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def create_master_database():
    """Create and initialize the master DuckDB database"""

    # Set up paths
    data_dir = Path("/home/aharon/projects/new_swt/data")
    db_path = data_dir / "master.duckdb"
    csv_path = data_dir / "GBPJPY_M1_REAL_2022-2025.csv"

    print(f"🚀 Creating master DuckDB database at: {db_path}")

    # Connect to DuckDB (creates file if doesn't exist)
    conn = duckdb.connect(str(db_path))

    try:
        # Create the master table with flexible schema
        print("📊 Creating master table schema...")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS master (
                -- Primary keys
                timestamp TIMESTAMP PRIMARY KEY,
                bar_index BIGINT,

                -- OHLCV data
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,

                -- Session info
                session_id INTEGER,
                session_bar_index INTEGER,

                -- WST features (70D) - will be populated later
                -- Placeholder for 70 WST features

                -- Technical indicators (20D) - will be populated later
                -- Placeholder for technical indicators

                -- Position features (9D) - will be populated later
                -- Placeholder for position features

                -- Metadata
                data_quality_flag INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("✅ Master table created successfully")

        # Import OHLCV data from CSV
        print(f"📥 Loading OHLCV data from {csv_path}...")

        # Read CSV and prepare data
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['bar_index'] = np.arange(len(df))

        # Insert into master table
        conn.execute("""
            INSERT INTO master (timestamp, bar_index, open, high, low, close, volume)
            SELECT timestamp, bar_index, open, high, low, close, volume
            FROM df
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        print(f"✅ Loaded {row_count:,} rows of OHLCV data")

        # Create indexes for efficient querying
        print("🔍 Creating indexes...")
        conn.execute("CREATE INDEX idx_master_timestamp ON master(timestamp)")
        conn.execute("CREATE INDEX idx_master_bar_index ON master(bar_index)")

        # Add WST feature columns (70D)
        print("🌊 Adding WST feature columns (70D)...")
        for i in range(70):
            conn.execute(f"ALTER TABLE master ADD COLUMN IF NOT EXISTS wst_{i} DOUBLE")

        # Add technical indicator columns (placeholder for now)
        print("📈 Adding technical indicator columns...")
        tech_indicators = [
            'rsi_14', 'macd_signal', 'stochastic_k', 'williams_r_14', 'roc_20',  # Momentum
            'atr_14', 'bb_width', 'keltner_position', 'realized_vol_20',  # Volatility
            'vwap_deviation', 'spread_ratio', 'time_sin', 'time_cos',  # Market Structure
            'pivot_s1_dist', 'pivot_r1_dist', 'pivot_dist', 'candle_pattern', 'sr_proximity'  # Price Action
        ]

        for indicator in tech_indicators:
            conn.execute(f"ALTER TABLE master ADD COLUMN IF NOT EXISTS {indicator} DOUBLE")

        # Add position feature columns (9D)
        print("💼 Adding position feature columns (9D)...")
        position_features = [
            'pos_equity_pips', 'pos_bars_since', 'pos_is_long',
            'pos_pips_from_peak', 'pos_max_dd', 'pos_amddp_reward',
            'pos_relative', 'pos_avg_swing', 'pos_dir_strength'
        ]

        for feature in position_features:
            conn.execute(f"ALTER TABLE master ADD COLUMN IF NOT EXISTS {feature} DOUBLE")

        # Create summary statistics view
        print("📊 Creating summary views...")
        conn.execute("""
            CREATE VIEW IF NOT EXISTS data_summary AS
            SELECT
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as total_bars,
                COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as trading_days,
                AVG(close) as avg_close,
                STDDEV(close) as std_close,
                MIN(low) as min_price,
                MAX(high) as max_price
            FROM master
        """)

        # Show summary
        summary = conn.execute("SELECT * FROM data_summary").fetchone()
        print("\n📈 DATABASE SUMMARY:")
        print(f"   Start Date: {summary[0]}")
        print(f"   End Date: {summary[1]}")
        print(f"   Total Bars: {summary[2]:,}")
        print(f"   Trading Days: {summary[3]:,}")
        print(f"   Avg Close: {summary[4]:.3f}")
        print(f"   Std Close: {summary[5]:.3f}")
        print(f"   Price Range: {summary[6]:.3f} - {summary[7]:.3f}")

        # Get schema info
        schema = conn.execute("DESCRIBE master").fetchall()
        total_columns = len(schema)
        print(f"\n📋 SCHEMA INFO:")
        print(f"   Total Columns: {total_columns}")
        print(f"   - OHLCV: 7 columns")
        print(f"   - WST Features: 70 columns")
        print(f"   - Technical Indicators: {len(tech_indicators)} columns")
        print(f"   - Position Features: {len(position_features)} columns")
        print(f"   - Metadata: 4 columns")

        # Create a sample view instead of function (DuckDB doesn't support CREATE FUNCTION this way)
        conn.execute("""
            CREATE OR REPLACE VIEW feature_window_example AS
            SELECT * FROM master
            ORDER BY timestamp
            LIMIT 256
        """)

        print("\n✨ Master database created successfully!")
        print(f"📁 Location: {db_path}")
        print(f"💾 Size: {db_path.stat().st_size / (1024*1024):.2f} MB")

        return conn

    except Exception as e:
        print(f"❌ Error creating database: {e}")
        conn.close()
        raise

    finally:
        # Don't close connection - return it for further use
        pass

    return conn


def test_database_operations(conn):
    """Test basic database operations"""
    print("\n🧪 Testing database operations...")

    # Test 1: Quick count
    count = conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
    print(f"✓ Row count: {count:,}")

    # Test 2: Sample data
    sample = conn.execute("SELECT timestamp, close FROM master LIMIT 5").fetchdf()
    print(f"✓ Sample data retrieved: {len(sample)} rows")

    # Test 3: Date range query
    date_range = conn.execute("""
        SELECT COUNT(*) as cnt
        FROM master
        WHERE timestamp BETWEEN '2022-01-01' AND '2022-02-01'
    """).fetchone()[0]
    print(f"✓ Date range query: {date_range:,} bars in Jan 2022")

    # Test 4: Check feature columns exist
    wst_cols = conn.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'master' AND column_name LIKE 'wst_%'
    """).fetchall()
    print(f"✓ WST columns verified: {len(wst_cols)} columns")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    print("="*60)
    print("MASTER DUCKDB DATABASE CREATOR")
    print("="*60)

    # Create the database
    conn = create_master_database()

    # Run tests
    test_database_operations(conn)

    # Close connection
    conn.close()

    print("\n🎉 Complete! Database ready for feature population.")
    print("Use: conn = duckdb.connect('/home/aharon/projects/new_swt/data/master.duckdb')")