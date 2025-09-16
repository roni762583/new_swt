#!/usr/bin/env python3
"""
Analyze bars exceeding 12 pips in GBPJPY data
Identifies patterns and potential data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys

def analyze_high_pip_bars(csv_file: str, pip_threshold: float = 12.0):
    """
    Analyze bars that exceed pip threshold to identify patterns

    Args:
        csv_file: Path to CSV file
        pip_threshold: Pip threshold (default 12 for suspicious activity)
    """

    print(f"\n{'='*70}")
    print(f"ANALYZING BARS > {pip_threshold} PIPS IN GBPJPY DATA")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    total_bars = len(df)
    print(f"Total bars: {total_bars:,}")

    # Calculate pip ranges (GBPJPY: 1 pip = 0.01)
    df['range_pips'] = (df['high'] - df['low']) * 100

    # Find bars exceeding threshold
    high_pip_mask = df['range_pips'] > pip_threshold
    high_pip_bars = df[high_pip_mask].copy()

    num_high_pip = len(high_pip_bars)
    pct_high_pip = (num_high_pip / total_bars) * 100

    print(f"\nBars > {pip_threshold} pips: {num_high_pip:,} ({pct_high_pip:.2f}%)")

    if num_high_pip == 0:
        print("No bars exceed threshold - data appears clean")
        return

    # Add time features for pattern analysis
    high_pip_bars['hour'] = high_pip_bars['timestamp'].dt.hour
    high_pip_bars['minute'] = high_pip_bars['timestamp'].dt.minute
    high_pip_bars['dayofweek'] = high_pip_bars['timestamp'].dt.dayofweek
    high_pip_bars['day_name'] = high_pip_bars['timestamp'].dt.day_name()
    high_pip_bars['date'] = high_pip_bars['timestamp'].dt.date

    # Pattern Analysis
    print(f"\n{'='*40}")
    print("PATTERN ANALYSIS")
    print(f"{'='*40}")

    # 1. Day of week pattern
    print("\n1. BY DAY OF WEEK:")
    day_counts = high_pip_bars['day_name'].value_counts()
    for day, count in day_counts.items():
        pct = (count / num_high_pip) * 100
        bar = '█' * int(pct/2)
        print(f"   {day:9s}: {count:6,} ({pct:5.1f}%) {bar}")

    # 2. Hour pattern
    print("\n2. BY HOUR (GMT):")
    hour_counts = high_pip_bars['hour'].value_counts().sort_index()
    for hour, count in hour_counts.items():
        pct = (count / num_high_pip) * 100
        bar = '█' * int(pct/2)
        print(f"   {hour:02d}:00: {count:6,} ({pct:5.1f}%) {bar}")

    # 3. Specific time patterns (e.g., market open/close)
    print("\n3. SPECIFIC TIME PATTERNS:")

    # Weekend periods (Friday 21:00 - Sunday 21:00)
    weekend_mask = (
        ((high_pip_bars['dayofweek'] == 4) & (high_pip_bars['hour'] >= 21)) |  # Friday evening
        (high_pip_bars['dayofweek'] == 5) |  # Saturday
        ((high_pip_bars['dayofweek'] == 6) & (high_pip_bars['hour'] < 21))  # Sunday until 21:00
    )
    weekend_count = weekend_mask.sum()
    weekend_pct = (weekend_count / num_high_pip) * 100
    print(f"   Weekend periods: {weekend_count:,} ({weekend_pct:.1f}%)")

    # Market open times (Sunday 21:00, Monday 00:00)
    market_open_mask = (
        ((high_pip_bars['dayofweek'] == 6) & (high_pip_bars['hour'] == 21)) |  # Sunday 21:00
        ((high_pip_bars['dayofweek'] == 0) & (high_pip_bars['hour'] == 0))   # Monday 00:00
    )
    open_count = market_open_mask.sum()
    open_pct = (open_count / num_high_pip) * 100
    print(f"   Market open (Sun 21:00/Mon 00:00): {open_count:,} ({open_pct:.1f}%)")

    # 4. Range distribution
    print("\n4. PIP RANGE DISTRIBUTION:")
    ranges = [
        (12, 20, "12-20 pips"),
        (20, 30, "20-30 pips"),
        (30, 50, "30-50 pips"),
        (50, 100, "50-100 pips"),
        (100, float('inf'), ">100 pips")
    ]

    for min_r, max_r, label in ranges:
        mask = (high_pip_bars['range_pips'] >= min_r) & (high_pip_bars['range_pips'] < max_r)
        count = mask.sum()
        pct = (count / num_high_pip) * 100
        bar = '█' * int(pct/2)
        print(f"   {label:12s}: {count:6,} ({pct:5.1f}%) {bar}")

    # 5. Consecutive high pip bars
    print("\n5. CONSECUTIVE PATTERNS:")

    # Find consecutive high pip bars
    high_pip_indices = high_pip_bars.index.tolist()
    consecutive_groups = []
    current_group = [high_pip_indices[0]]

    for i in range(1, len(high_pip_indices)):
        if high_pip_indices[i] == high_pip_indices[i-1] + 1:
            current_group.append(high_pip_indices[i])
        else:
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            current_group = [high_pip_indices[i]]

    if len(current_group) > 1:
        consecutive_groups.append(current_group)

    if consecutive_groups:
        print(f"   Found {len(consecutive_groups)} groups of consecutive high-pip bars")
        group_sizes = [len(g) for g in consecutive_groups]
        print(f"   Largest consecutive group: {max(group_sizes)} bars")
        print(f"   Average group size: {np.mean(group_sizes):.1f} bars")
    else:
        print("   No consecutive patterns found")

    # 6. Sample high pip bars
    print(f"\n6. SAMPLE HIGH PIP BARS (Top 10):")
    top_bars = high_pip_bars.nlargest(10, 'range_pips')

    print(f"   {'Timestamp':<30} {'Range(pips)':>12} {'High':>10} {'Low':>10} {'Day'}")
    print(f"   {'-'*75}")
    for idx, row in top_bars.iterrows():
        print(f"   {str(row['timestamp']):<30} {row['range_pips']:>12.1f} "
              f"{row['high']:>10.3f} {row['low']:>10.3f} {row['day_name']}")

    # 7. Time gaps analysis
    print("\n7. TIME GAP ANALYSIS:")

    # Calculate time differences
    time_diffs = df['timestamp'].diff()
    df['gap_minutes'] = time_diffs.dt.total_seconds() / 60

    # Check gaps before high pip bars
    gaps_before = df.loc[high_pip_mask, 'gap_minutes'].dropna()

    if len(gaps_before) > 0:
        large_gaps = gaps_before[gaps_before > 10]
        if len(large_gaps) > 0:
            print(f"   High pip bars after gaps > 10min: {len(large_gaps):,}")
            print(f"   Average gap before high pip bar: {gaps_before.mean():.1f} minutes")
            print(f"   Max gap before high pip bar: {gaps_before.max():.1f} minutes")
        else:
            print("   No significant gaps before high pip bars")

    # Summary and recommendations
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")

    if weekend_pct > 50:
        print("⚠️  MAJORITY of high pip bars occur during WEEKEND periods")
        print("   → Data likely contains weekend gaps or synthetic weekend data")

    if open_pct > 10:
        print("⚠️  SIGNIFICANT spike at market open (Sunday 21:00 GMT)")
        print("   → Weekend gap effect clearly visible")

    if any(r > 100 for r in high_pip_bars['range_pips']):
        print("⚠️  EXTREME pip ranges detected (>100 pips per minute)")
        print("   → Data quality issue - possibly wrong timeframe or corrupted")

    if num_high_pip / total_bars > 0.5:
        print("⚠️  OVER 50% of bars exceed normal range")
        print("   → Fundamental data problem - not suitable for production")

    print("\n📊 CONCLUSION:")
    if weekend_pct > 30 or num_high_pip / total_bars > 0.3:
        print("   ❌ Data is NOT suitable for production trading")
        print("   ❌ Contains synthetic/interpolated data or wrong timeframe")
        print("   ❌ Requires complete replacement with clean tick data")
    else:
        print("   ⚠️  Data has quality issues that need addressing")
        print("   → Remove weekend periods")
        print("   → Filter out bars with gaps > 10 minutes")
        print("   → Validate against known market hours")

    # Export high pip bars for inspection
    output_file = "high_pip_bars_analysis.csv"
    high_pip_bars[['timestamp', 'open', 'high', 'low', 'close', 'range_pips', 'day_name', 'hour']].to_csv(
        output_file, index=False
    )
    print(f"\n📁 Detailed analysis exported to: {output_file}")

    return high_pip_bars


if __name__ == "__main__":
    csv_file = "data/GBPJPY_M1_REAL_2022-2025.csv"  # Analyze the fresh OANDA data

    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Analyze with 12 pip threshold (normal max is ~10 during news)
    analyze_high_pip_bars(csv_file, pip_threshold=12.0)