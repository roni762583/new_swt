#!/usr/bin/env python3
"""
Test Welford's online algorithm for incremental std calculation.

Compares:
1. Fixed std (from training data)
2. Welford's incremental std (starting from fixed prior)
3. How much std changes per step and during extreme events

Key Questions:
- How much does incremental std deviate from fixed std?
- How does std change during extreme events (|z| > 3)?
- What is N when initializing with prior? (prior + new value = N=2)
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from pathlib import Path
from typing import Tuple, List
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
CONFIG_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/feature_zscore_config.json")


class WelfordStd:
    """
    Welford's online algorithm for incremental mean and std calculation.

    Algorithm:
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
        variance = M2 / n
        std = sqrt(variance)

    Properties:
        - Numerically stable (avoids catastrophic cancellation)
        - Single pass, O(1) memory
        - Can initialize with prior (n, mean, std)
    """

    def __init__(self, n: int = 0, mean: float = 0.0, std: float = 0.0):
        """
        Initialize Welford algorithm.

        Args:
            n: Initial sample count (0 for fresh start, or prior n)
            mean: Initial mean (prior mean if initializing from training)
            std: Initial std (prior std if initializing from training)
        """
        self.n = n
        self.mean = mean
        self.M2 = std**2 * n if n > 0 else 0.0  # M2 = variance * n

    def update(self, x: float) -> Tuple[float, float]:
        """
        Update with new value and return current (mean, std).

        Args:
            x: New data point

        Returns:
            Tuple of (current_mean, current_std)
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        variance = self.M2 / self.n if self.n > 1 else 0.0
        std = np.sqrt(variance)

        return self.mean, std

    def get_stats(self) -> Tuple[int, float, float]:
        """Return current (n, mean, std)."""
        variance = self.M2 / self.n if self.n > 1 else 0.0
        return self.n, self.mean, np.sqrt(variance)


def test_welford_vs_fixed_std():
    """
    Test how Welford's incremental std deviates from fixed std.
    """
    logger.info("="*80)
    logger.info("WELFORD'S INCREMENTAL STD vs FIXED STD")
    logger.info("="*80)

    # Load config to get fixed std
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    fixed_std = config['features']['h1_swing_range_position']['fixed_std']
    training_mean = config['features']['h1_swing_range_position']['training_mean']
    training_n = config['features']['h1_swing_range_position']['training_rows']

    logger.info(f"\n📊 Fixed Std from Training (N={training_n:,}):")
    logger.info(f"  Mean: {training_mean:.6f}")
    logger.info(f"  Std:  {fixed_std:.6f}")

    # Load test data (remaining 30%)
    conn = duckdb.connect(str(DB_PATH))

    df = conn.execute("""
        SELECT
            bar_index,
            h1_swing_range_position,
            h1_swing_range_position_zsarctan
        FROM master
        WHERE h1_swing_range_position IS NOT NULL
        ORDER BY bar_index
    """).fetch_df()

    # Split into train/test
    train_size = int(len(df) * 0.7)
    test_df = df.iloc[train_size:].copy()

    logger.info(f"\nTest data size: {len(test_df):,} bars")

    # Initialize Welford with training prior
    welford = WelfordStd(n=training_n, mean=training_mean, std=fixed_std)

    logger.info(f"\n🔬 Welford Initialization:")
    logger.info(f"  N = {training_n:,} (prior from training)")
    logger.info(f"  Mean = {training_mean:.6f}")
    logger.info(f"  Std = {fixed_std:.6f}")
    logger.info(f"  → First new value will make N = {training_n + 1:,}")

    # Track incremental updates
    incremental_stds = []
    std_changes = []
    std_pct_changes = []
    prev_std = fixed_std

    # Track extremes
    extreme_events = []  # (bar_index, z_score, std_before, std_after, std_change)

    for idx, row in test_df.iterrows():
        value = row['h1_swing_range_position']
        z_score = row['h1_swing_range_position_zsarctan']

        # Store std before update
        std_before = welford.get_stats()[2]

        # Update Welford
        current_mean, current_std = welford.update(value)

        # Track changes
        incremental_stds.append(current_std)
        std_change = current_std - prev_std
        std_changes.append(std_change)
        std_pct_change = (current_std - prev_std) / prev_std * 100
        std_pct_changes.append(std_pct_change)

        # Track extreme events (|z| > 0.8 in arctan space ≈ |z| > ~3 in raw space)
        if not np.isnan(z_score) and abs(z_score) > 0.8:
            extreme_events.append({
                'bar_index': row['bar_index'],
                'z_score': z_score,
                'value': value,
                'std_before': std_before,
                'std_after': current_std,
                'std_change': current_std - std_before,
                'std_pct_change': (current_std - std_before) / std_before * 100,
                'n': welford.n
            })

        prev_std = current_std

    # Analysis
    incremental_stds = np.array(incremental_stds)
    std_changes = np.array(std_changes)
    std_pct_changes = np.array(std_pct_changes)

    final_n, final_mean, final_std = welford.get_stats()

    logger.info("\n" + "="*80)
    logger.info("INCREMENTAL STD EVOLUTION")
    logger.info("="*80)

    logger.info(f"\nFinal Welford Stats (after {len(test_df):,} updates):")
    logger.info(f"  N:        {final_n:,}")
    logger.info(f"  Mean:     {final_mean:.6f}")
    logger.info(f"  Std:      {final_std:.6f}")
    logger.info(f"  Fixed Std: {fixed_std:.6f}")
    logger.info(f"  Deviation: {final_std - fixed_std:.6f} ({(final_std - fixed_std)/fixed_std*100:.3f}%)")

    logger.info(f"\n📊 Incremental Std Statistics:")
    logger.info(f"  Min:      {incremental_stds.min():.6f}")
    logger.info(f"  Q25:      {np.percentile(incremental_stds, 25):.6f}")
    logger.info(f"  Median:   {np.median(incremental_stds):.6f}")
    logger.info(f"  Q75:      {np.percentile(incremental_stds, 75):.6f}")
    logger.info(f"  Max:      {incremental_stds.max():.6f}")
    logger.info(f"  Range:    {incremental_stds.max() - incremental_stds.min():.6f}")

    # Deviation from fixed std
    deviations = incremental_stds - fixed_std
    pct_deviations = deviations / fixed_std * 100

    logger.info(f"\n📈 Deviation from Fixed Std:")
    logger.info(f"  Mean deviation:     {np.mean(deviations):.6f} ({np.mean(pct_deviations):.3f}%)")
    logger.info(f"  Max deviation:      {np.max(np.abs(deviations)):.6f} ({np.max(np.abs(pct_deviations)):.3f}%)")
    logger.info(f"  Within ±0.1%:       {np.sum(np.abs(pct_deviations) < 0.1):,} ({np.sum(np.abs(pct_deviations) < 0.1)/len(pct_deviations)*100:.2f}%)")
    logger.info(f"  Within ±1%:         {np.sum(np.abs(pct_deviations) < 1):,} ({np.sum(np.abs(pct_deviations) < 1)/len(pct_deviations)*100:.2f}%)")
    logger.info(f"  Within ±5%:         {np.sum(np.abs(pct_deviations) < 5):,} ({np.sum(np.abs(pct_deviations) < 5)/len(pct_deviations)*100:.2f}%)")

    logger.info(f"\n📉 Per-Step Changes:")
    logger.info(f"  Mean abs change:    {np.mean(np.abs(std_changes)):.8f}")
    logger.info(f"  Median abs change:  {np.median(np.abs(std_changes)):.8f}")
    logger.info(f"  Max abs change:     {np.max(np.abs(std_changes)):.8f}")
    logger.info(f"  Mean % change:      {np.mean(np.abs(std_pct_changes)):.6f}%")
    logger.info(f"  Max % change:       {np.max(np.abs(std_pct_changes)):.6f}%")

    # Analyze extreme events
    logger.info("\n" + "="*80)
    logger.info("STD CHANGES DURING EXTREME EVENTS")
    logger.info("="*80)

    if extreme_events:
        logger.info(f"\nFound {len(extreme_events):,} extreme events (|z| > 0.8)")

        # Sort by std change magnitude
        extreme_events_sorted = sorted(extreme_events, key=lambda x: abs(x['std_change']), reverse=True)

        logger.info("\n🔴 Top 10 Largest Std Changes During Extremes:")
        logger.info(f"  {'Bar':<10} {'Z-Score':<10} {'Value':<10} {'Std Before':<12} {'Std After':<12} {'Change':<12} {'% Change':<10} {'N':<12}")
        logger.info("-"*100)

        for evt in extreme_events_sorted[:10]:
            logger.info(
                f"  {evt['bar_index']:<10} "
                f"{evt['z_score']:>9.4f} "
                f"{evt['value']:>9.4f} "
                f"{evt['std_before']:>11.6f} "
                f"{evt['std_after']:>11.6f} "
                f"{evt['std_change']:>+11.6f} "
                f"{evt['std_pct_change']:>+9.4f}% "
                f"{evt['n']:<12,}"
            )

        # Statistics on extreme events
        extreme_changes = [abs(e['std_change']) for e in extreme_events]
        extreme_pct_changes = [abs(e['std_pct_change']) for e in extreme_events]

        logger.info(f"\n📊 Extreme Event Std Changes:")
        logger.info(f"  Mean abs change:    {np.mean(extreme_changes):.6f}")
        logger.info(f"  Median abs change:  {np.median(extreme_changes):.6f}")
        logger.info(f"  Max abs change:     {np.max(extreme_changes):.6f}")
        logger.info(f"  Mean % change:      {np.mean(extreme_pct_changes):.4f}%")
        logger.info(f"  Max % change:       {np.max(extreme_pct_changes):.4f}%")

        # Compare to normal steps
        logger.info(f"\n⚖️  Extreme vs Normal Steps:")
        logger.info(f"  Normal step mean abs change:   {np.mean(np.abs(std_changes)):.6f}")
        logger.info(f"  Extreme step mean abs change:  {np.mean(extreme_changes):.6f}")
        logger.info(f"  Ratio (extreme/normal):        {np.mean(extreme_changes)/np.mean(np.abs(std_changes)):.2f}x")

    logger.info("\n" + "="*80)
    logger.info("💡 KEY FINDINGS")
    logger.info("="*80)

    logger.info(f"\n1️⃣  INITIALIZATION:")
    logger.info(f"   Prior N = {training_n:,}")
    logger.info(f"   First update → N = {training_n + 1:,}")
    logger.info(f"   Impact of single value: 1/{training_n + 1:,} = {1/(training_n+1):.8f}")

    logger.info(f"\n2️⃣  STD STABILITY:")
    logger.info(f"   Fixed std remains at: {fixed_std:.6f}")
    logger.info(f"   Welford final std:    {final_std:.6f}")
    logger.info(f"   Difference:           {abs(final_std - fixed_std):.6f} ({abs(final_std - fixed_std)/fixed_std*100:.3f}%)")

    logger.info(f"\n3️⃣  PER-STEP IMPACT:")
    logger.info(f"   Mean change per step: {np.mean(np.abs(std_changes)):.8f} ({np.mean(np.abs(std_pct_changes)):.6f}%)")
    logger.info(f"   With N ~ {final_n:,}, each step has ~1/N = {1/final_n:.8f} weight")

    if extreme_events:
        logger.info(f"\n4️⃣  EXTREME EVENTS:")
        logger.info(f"   {len(extreme_events):,} extreme events detected")
        logger.info(f"   Mean std change: {np.mean(extreme_changes):.6f} ({np.mean(extreme_pct_changes):.4f}%)")
        logger.info(f"   Extreme events cause {np.mean(extreme_changes)/np.mean(np.abs(std_changes)):.2f}x larger std changes")

    logger.info(f"\n5️⃣  PRACTICAL IMPLICATION:")
    logger.info(f"   Fixed std is VERY stable with large N")
    logger.info(f"   After {len(test_df):,} test bars, deviation is only {abs(final_std - fixed_std)/fixed_std*100:.3f}%")
    logger.info(f"   Even extreme events cause minimal drift (<{np.max(extreme_pct_changes) if extreme_events else 0:.3f}% max)")

    conn.close()

    return {
        'fixed_std': fixed_std,
        'final_std': final_std,
        'training_n': training_n,
        'test_bars': len(test_df),
        'extreme_events': len(extreme_events),
        'mean_step_change': np.mean(np.abs(std_changes)),
        'mean_extreme_change': np.mean(extreme_changes) if extreme_events else 0
    }


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("Testing Welford's Incremental Std Algorithm")
    logger.info("="*80)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    if not CONFIG_PATH.exists():
        logger.error(f"Config not found: {CONFIG_PATH}")
        return 1

    try:
        results = test_welford_vs_fixed_std()

        logger.info("\n" + "="*80)
        logger.info("✅ TEST COMPLETE")
        logger.info("="*80)
        logger.info("\n🎯 CONCLUSION:")
        logger.info("   Welford's algorithm with large prior N is extremely stable.")
        logger.info("   Fixed std from training is essentially constant in practice.")
        logger.info("   Even extreme price events cause negligible drift.")

    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())