#!/usr/bin/env python3
"""
Comprehensive extreme event analysis: Price behavior after z-score extremes.

Analyzes:
1. Window=500 extremes (|z| > 3 in raw space, arctan > 0.8)
2. Window=20 extremes (same thresholds)
3. Forward price behavior at 1min, 5min, 30min horizons
4. Pullback frequency and magnitude
5. Continuation vs reversal statistics
6. Comparison between window sizes

Questions answered:
- After extreme up: pullback frequency? continuation? distance traveled?
- After extreme down: same analysis
- Does window size matter for predictive value?
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
CONFIG_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/feature_zscore_config.json")


def calculate_zscore_features(conn: duckdb.DuckDBPyConnection):
    """Calculate both Window=20 and Window=500 zscore features."""
    logger.info("Calculating z-score features for both windows...")

    # Load fixed std
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    fixed_std = config['features']['h1_swing_range_position']['fixed_std']

    # Fetch data
    df = conn.execute("""
        SELECT
            bar_index,
            timestamp,
            close,
            h1_swing_range_position
        FROM master
        WHERE h1_swing_range_position IS NOT NULL
        ORDER BY bar_index
    """).fetch_df()

    logger.info(f"Loaded {len(df):,} rows")

    # Calculate Window=500 z-score
    logger.info("Calculating Window=500 z-score...")
    data = df['h1_swing_range_position'].values
    series = pd.Series(data)

    rolling_mean_500 = series.rolling(window=500, min_periods=500).mean()
    z_500 = (data - rolling_mean_500) / fixed_std
    arctan_500 = np.arctan(z_500) * 2 / np.pi

    # Calculate Window=20 z-score
    logger.info("Calculating Window=20 z-score...")
    rolling_mean_20 = series.rolling(window=20, min_periods=20).mean()
    z_20 = (data - rolling_mean_20) / fixed_std
    arctan_20 = np.arctan(z_20) * 2 / np.pi

    # Add to dataframe
    df['z_500'] = z_500
    df['arctan_500'] = arctan_500
    df['z_20'] = z_20
    df['arctan_20'] = arctan_20

    # Update database with Window=20
    logger.info("Updating database with Window=20 feature...")
    conn.register('w20_data', df[['bar_index', 'arctan_20']])
    conn.execute("""
        UPDATE master
        SET h1_swing_range_position_zsarctan_w20 = w20_data.arctan_20
        FROM w20_data
        WHERE master.bar_index = w20_data.bar_index
    """)

    logger.info("✅ Z-score features calculated and saved")

    return df


def analyze_forward_returns(
    df: pd.DataFrame,
    extreme_mask: np.ndarray,
    direction: str,
    window: int,
    horizons: List[int] = [1, 5, 30]
) -> Dict:
    """
    Analyze forward price behavior after extreme events.

    Args:
        df: DataFrame with price data
        extreme_mask: Boolean mask for extreme events
        direction: 'up' or 'down'
        window: Window size (20 or 500)
        horizons: Forward horizons in minutes (bars)

    Returns:
        Dict with analysis results
    """
    extreme_indices = np.where(extreme_mask)[0]

    if len(extreme_indices) == 0:
        return None

    results = {
        'n_events': len(extreme_indices),
        'direction': direction,
        'window': window,
        'horizons': {}
    }

    # Analyze each horizon
    for horizon in horizons:
        forward_returns = []
        pullback_count = 0
        continuation_count = 0
        max_favorable_moves = []
        max_adverse_moves = []

        for idx in extreme_indices:
            # Skip if not enough forward data
            if idx + horizon >= len(df):
                continue

            entry_price = df.iloc[idx]['close']

            # Get forward prices
            forward_prices = df.iloc[idx+1:idx+horizon+1]['close'].values

            if len(forward_prices) == 0:
                continue

            # Calculate returns (in pips for GBPJPY: 0.01)
            forward_ret = (forward_prices - entry_price) / 0.01

            # Final return at horizon
            final_return = forward_ret[-1]
            forward_returns.append(final_return)

            # Max favorable and adverse moves
            if direction == 'up':
                # Extreme up: favorable = further up, adverse = down
                max_favorable = np.max(forward_ret)
                max_adverse = np.min(forward_ret)

                # Pullback: price goes down before potentially recovering
                if np.any(forward_ret < -2):  # At least 2 pips pullback
                    pullback_count += 1

                # Continuation: final return is positive
                if final_return > 0:
                    continuation_count += 1

            else:  # direction == 'down'
                # Extreme down: favorable = further down, adverse = up
                max_favorable = np.min(forward_ret)
                max_adverse = np.max(forward_ret)

                # Pullback: price goes up before potentially recovering
                if np.any(forward_ret > 2):  # At least 2 pips pullback
                    pullback_count += 1

                # Continuation: final return is negative
                if final_return < 0:
                    continuation_count += 1

            max_favorable_moves.append(max_favorable)
            max_adverse_moves.append(max_adverse)

        # Calculate statistics
        if forward_returns:
            forward_returns = np.array(forward_returns)
            max_favorable_moves = np.array(max_favorable_moves)
            max_adverse_moves = np.array(max_adverse_moves)

            results['horizons'][horizon] = {
                'n_valid': len(forward_returns),
                'mean_return': float(np.mean(forward_returns)),
                'median_return': float(np.median(forward_returns)),
                'std_return': float(np.std(forward_returns)),
                'min_return': float(np.min(forward_returns)),
                'max_return': float(np.max(forward_returns)),
                'pullback_rate': pullback_count / len(forward_returns),
                'continuation_rate': continuation_count / len(forward_returns),
                'mean_max_favorable': float(np.mean(max_favorable_moves)),
                'mean_max_adverse': float(np.mean(max_adverse_moves)),
                'median_max_favorable': float(np.median(max_favorable_moves)),
                'median_max_adverse': float(np.median(max_adverse_moves))
            }

    return results


def comprehensive_extreme_analysis(df: pd.DataFrame):
    """Run comprehensive analysis on both window sizes."""

    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE EXTREME EVENT ANALYSIS")
    logger.info("="*80)

    # Define extreme thresholds (arctan > 0.8 ≈ z > 3)
    extreme_threshold = 0.8

    # Identify extremes for both windows
    extreme_up_500 = (df['arctan_500'] > extreme_threshold) & (~df['arctan_500'].isna())
    extreme_down_500 = (df['arctan_500'] < -extreme_threshold) & (~df['arctan_500'].isna())

    extreme_up_20 = (df['arctan_20'] > extreme_threshold) & (~df['arctan_20'].isna())
    extreme_down_20 = (df['arctan_20'] < -extreme_threshold) & (~df['arctan_20'].isna())

    logger.info(f"\nExtreme Events Detected:")
    logger.info(f"  Window=500: {extreme_up_500.sum():,} up, {extreme_down_500.sum():,} down")
    logger.info(f"  Window=20:  {extreme_up_20.sum():,} up, {extreme_down_20.sum():,} down")

    # Analyze forward behavior
    horizons = [1, 5, 30]  # 1min, 5min, 30min

    analyses = {}

    # Window=500 analysis
    logger.info("\nAnalyzing Window=500 extremes...")
    analyses['w500_up'] = analyze_forward_returns(df, extreme_up_500, 'up', 500, horizons)
    analyses['w500_down'] = analyze_forward_returns(df, extreme_down_500, 'down', 500, horizons)

    # Window=20 analysis
    logger.info("Analyzing Window=20 extremes...")
    analyses['w20_up'] = analyze_forward_returns(df, extreme_up_20, 'up', 20, horizons)
    analyses['w20_down'] = analyze_forward_returns(df, extreme_down_20, 'down', 20, horizons)

    return analyses


def print_analysis_results(analyses: Dict):
    """Print comprehensive analysis results."""

    logger.info("\n" + "="*80)
    logger.info("EXTREME EVENT FORWARD BEHAVIOR ANALYSIS")
    logger.info("="*80)

    for key, analysis in analyses.items():
        if analysis is None:
            continue

        window = analysis['window']
        direction = analysis['direction']
        n_events = analysis['n_events']

        logger.info("\n" + "-"*80)
        logger.info(f"📊 WINDOW={window} | DIRECTION={direction.upper()} | EVENTS={n_events:,}")
        logger.info("-"*80)

        for horizon, stats in analysis['horizons'].items():
            logger.info(f"\n⏱️  {horizon}-minute forward behavior:")
            logger.info(f"  Valid samples:        {stats['n_valid']:,}")
            logger.info(f"  Mean return:          {stats['mean_return']:+8.2f} pips")
            logger.info(f"  Median return:        {stats['median_return']:+8.2f} pips")
            logger.info(f"  Std return:           {stats['std_return']:8.2f} pips")
            logger.info(f"  Range:                [{stats['min_return']:+.1f}, {stats['max_return']:+.1f}] pips")

            logger.info(f"\n  📈 Movement Statistics:")
            logger.info(f"    Mean max favorable:   {stats['mean_max_favorable']:+8.2f} pips")
            logger.info(f"    Mean max adverse:     {stats['mean_max_adverse']:+8.2f} pips")
            logger.info(f"    Median max favorable: {stats['median_max_favorable']:+8.2f} pips")
            logger.info(f"    Median max adverse:   {stats['median_max_adverse']:+8.2f} pips")

            logger.info(f"\n  🔄 Behavior Patterns:")
            logger.info(f"    Pullback rate:        {stats['pullback_rate']*100:5.1f}%")
            logger.info(f"    Continuation rate:    {stats['continuation_rate']*100:5.1f}%")

            # Interpretation
            if direction == 'up':
                if stats['continuation_rate'] > 0.5:
                    logger.info(f"    → 🟢 Extreme UP tends to CONTINUE up")
                else:
                    logger.info(f"    → 🔴 Extreme UP tends to REVERSE")
            else:
                if stats['continuation_rate'] > 0.5:
                    logger.info(f"    → 🟢 Extreme DOWN tends to CONTINUE down")
                else:
                    logger.info(f"    → 🔴 Extreme DOWN tends to REVERSE")

    # Comparison between windows
    logger.info("\n" + "="*80)
    logger.info("📊 WINDOW COMPARISON")
    logger.info("="*80)

    for direction in ['up', 'down']:
        logger.info(f"\n🔍 {direction.upper()} Extremes Comparison:")

        w500_key = f'w500_{direction}'
        w20_key = f'w20_{direction}'

        if w500_key not in analyses or w20_key not in analyses:
            continue

        if analyses[w500_key] is None or analyses[w20_key] is None:
            continue

        logger.info(f"  {'Horizon':<10} {'W500 Mean':>12} {'W20 Mean':>12} {'Diff':>10} {'W500 Cont%':>12} {'W20 Cont%':>12}")
        logger.info("  " + "-"*75)

        for horizon in [1, 5, 30]:
            if horizon in analyses[w500_key]['horizons'] and horizon in analyses[w20_key]['horizons']:
                w500_stats = analyses[w500_key]['horizons'][horizon]
                w20_stats = analyses[w20_key]['horizons'][horizon]

                mean_diff = w500_stats['mean_return'] - w20_stats['mean_return']

                logger.info(
                    f"  {horizon}-min{' '*5}"
                    f"{w500_stats['mean_return']:>+11.2f} "
                    f"{w20_stats['mean_return']:>+11.2f} "
                    f"{mean_diff:>+9.2f} "
                    f"{w500_stats['continuation_rate']*100:>11.1f}% "
                    f"{w20_stats['continuation_rate']*100:>11.1f}%"
                )

    logger.info("\n" + "="*80)
    logger.info("💡 KEY INSIGHTS")
    logger.info("="*80)

    # Determine best predictor
    if 'w500_up' in analyses and analyses['w500_up']:
        w500_30min = analyses['w500_up']['horizons'].get(30, {})
        if w500_30min:
            logger.info(f"\n🎯 Window=500 Extreme UP (30-min forward):")
            logger.info(f"  Mean return: {w500_30min.get('mean_return', 0):+.2f} pips")
            logger.info(f"  Continuation: {w500_30min.get('continuation_rate', 0)*100:.1f}%")

            if w500_30min.get('continuation_rate', 0) > 0.5:
                logger.info(f"  → 🟢 Mean reversion signal is WEAK (trend continues)")
            else:
                logger.info(f"  → 🔴 Mean reversion signal is STRONG (tends to reverse)")

    if 'w20_up' in analyses and analyses['w20_up']:
        w20_30min = analyses['w20_up']['horizons'].get(30, {})
        if w20_30min:
            logger.info(f"\n🎯 Window=20 Extreme UP (30-min forward):")
            logger.info(f"  Mean return: {w20_30min.get('mean_return', 0):+.2f} pips")
            logger.info(f"  Continuation: {w20_30min.get('continuation_rate', 0)*100:.1f}%")

            if w20_30min.get('continuation_rate', 0) > 0.5:
                logger.info(f"  → 🟢 Short-term extreme less predictive of reversal")
            else:
                logger.info(f"  → 🔴 Short-term extreme signals reversal opportunity")


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("Extreme Event Price Behavior Analysis")
    logger.info("="*80)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    conn = duckdb.connect(str(DB_PATH))

    try:
        # Calculate z-score features
        df = calculate_zscore_features(conn)

        # Run comprehensive analysis
        analyses = comprehensive_extreme_analysis(df)

        # Print results
        print_analysis_results(analyses)

        # Save results to JSON
        output_path = Path("extreme_event_analysis_results.json")
        with open(output_path, 'w') as f:
            json.dump(analyses, f, indent=2, default=str)

        logger.info(f"\n✅ Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    logger.info("\n✅ Analysis complete!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())