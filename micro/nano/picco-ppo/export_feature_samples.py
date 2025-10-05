#!/usr/bin/env python3
"""
Export first and last 20 rows of selected features and labels as CSV samples.
These samples serve as data snapshots for documentation and version control.
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
OUTPUT_DIR = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo")

# Selected features (26 ML features + 4 OHLCV for reference)
FEATURES = [
    # Core (4) - Reference only
    'close', 'high', 'low', 'volume',

    # Momentum (4)
    'log_return_1m', 'log_return_5m', 'log_return_60m', 'efficiency_ratio_h1',

    # Momentum Extended (1)
    'momentum_strength_10_zsarctan_w20',

    # Volatility (4)
    'atr_14', 'atr_14_zsarctan_w20', 'vol_ratio_deviation', 'realized_vol_60_zsarctan_w20',

    # Swing Structure (5)
    'h1_swing_range_position', 'swing_point_range',
    'high_swing_slope_h1', 'low_swing_slope_h1', 'h1_trend_slope_zsarctan',

    # Z-Score Extremes (6)
    'h1_swing_range_position_zsarctan_w20', 'swing_point_range_zsarctan_w20',
    'high_swing_slope_h1_zsarctan', 'low_swing_slope_h1_zsarctan',
    'high_swing_slope_m1_zsarctan_w20', 'low_swing_slope_m1_zsarctan_w20',
    'combo_geometric',

    # Indicators (1)
    'bb_position',

    # Time (4)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

# Target label
LABEL = 'pretrain_action'

# Metadata for context
METADATA = ['bar_index', 'timestamp']


def export_feature_samples():
    """Export sample with one complete trade cycle."""
    print("=" * 80)
    print("EXPORT FEATURE SAMPLES FOR VERSION CONTROL")
    print("=" * 80)

    conn = duckdb.connect(str(DB_PATH))

    # Build query
    all_columns = METADATA + FEATURES + [LABEL]
    column_list = ', '.join(all_columns)

    print(f"\nColumns to export: {len(all_columns)}")
    print(f"  Metadata: {len(METADATA)}")
    print(f"  Features: {len(FEATURES)}")
    print(f"  Label: 1")

    # Fetch data with labels
    query = f"""
        SELECT {column_list}
        FROM master
        WHERE pretrain_action IS NOT NULL
        ORDER BY bar_index
    """

    print("\nFetching data...")
    df = conn.execute(query).fetch_df()
    print(f"Total rows with labels: {len(df):,}")

    # Use complete trade: SELL at bar 9428, CLOSE at bar 9449 (21 bar duration)
    # Include 5 bars before SELL and 5 bars after CLOSE for context
    # Total: 32 bars (Jan 11, 2022, 09:23-09:54)

    start_idx = 9423  # 5 bars before SELL
    end_idx = 9454    # 5 bars after CLOSE

    combined = df[(df['bar_index'] >= start_idx) & (df['bar_index'] <= end_idx)].copy()

    print(f"\n📊 Complete trade cycle ({len(combined)} rows):")
    print(f"  Bar range: {start_idx}-{end_idx}")
    print(f"  Date: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    # Action distribution for combined
    action_counts = combined['pretrain_action'].value_counts().sort_index()
    print(f"\n  Action distribution ({len(combined)} rows total):")
    for action, count in action_counts.items():
        action_name = ['HOLD', 'BUY', 'SELL', 'CLOSE'][int(action)]
        print(f"    {action_name} ({action}): {count}")

    # Separate features and labels
    feature_columns = METADATA + FEATURES
    label_columns = METADATA + [LABEL]

    features_df = combined[feature_columns].copy()
    labels_df = combined[label_columns].copy()

    # Save to CSV
    features_path = OUTPUT_DIR / "sample_features.csv"
    labels_path = OUTPUT_DIR / "sample_labels.csv"

    features_df.to_csv(features_path, index=False, float_format='%.6f')
    print(f"\n✅ Saved: {features_path}")
    print(f"   Rows: {len(features_df)}, Columns: {len(features_df.columns)}")
    print(f"   Size: {features_path.stat().st_size / 1024:.1f} KB")

    labels_df.to_csv(labels_path, index=False, float_format='%.6f')
    print(f"✅ Saved: {labels_path}")
    print(f"   Rows: {len(labels_df)}, Columns: {len(labels_df.columns)}")
    print(f"   Size: {labels_path.stat().st_size / 1024:.1f} KB")

    # Summary statistics
    print("\n" + "=" * 80)
    print("FEATURE RANGE SUMMARY (from samples)")
    print("=" * 80)

    print("\nML-READY FEATURES (excluding OHLCV):")
    ml_features = [f for f in FEATURES if f not in ['close', 'high', 'low', 'volume']]

    for feat in ml_features[:10]:  # Show first 10
        min_val = combined[feat].min()
        max_val = combined[feat].max()
        mean_val = combined[feat].mean()
        in_range = "✅" if (min_val >= -1.1 and max_val <= 1.1) else "⚠️"
        print(f"{in_range} {feat:40s} [{min_val:+8.4f}, {max_val:+8.4f}]  mean: {mean_val:+8.4f}")

    print(f"\n... and {len(ml_features) - 10} more features")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ EXPORT COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  - sample_features.csv  ({len(combined)} rows)")
    print(f"  - sample_labels.csv    ({len(combined)} rows)")
    print("\nComplete trade captured (32 bars):")
    print("  - Bars 9423-9427: 5 bars before trade (context)")
    print("  - Bar 9428: SELL action (trade entry)")
    print("  - Bars 9429-9448: 20 bars holding position")
    print("  - Bar 9449: CLOSE action (trade exit)")
    print("  - Bars 9450-9454: 5 bars after trade (context)")
    print("\nTrade profit: SELL @ 157.0240, CLOSE @ 156.8160 = +20.8 pips")
    print("\nThese CSV files should be committed to Git for documentation.")


if __name__ == "__main__":
    export_feature_samples()
