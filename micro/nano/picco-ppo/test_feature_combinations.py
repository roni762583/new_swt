#!/usr/bin/env python3
"""
Test different methods of combining swing z-score features.

Tests:
1. Multiplication: range_z × position_z
2. Addition: range_z + position_z
3. Max absolute: max(|range_z|, |position_z|) × sign
4. Logical AND: both extreme in same direction
5. Dominant signal: use range_z when |range_z| > |position_z|, else position_z

Evaluates predictive power for 1, 5, 30-min forward returns.
"""

import numpy as np
import pandas as pd
import duckdb
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")


def load_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load z-score features and price data."""
    logger.info("Loading data from database...")

    df = conn.execute("""
        SELECT
            bar_index,
            close,
            swing_point_range_zsarctan as range_z,
            h1_swing_range_position_zsarctan_w20 as position_z
        FROM master
        WHERE swing_point_range_zsarctan IS NOT NULL
          AND h1_swing_range_position_zsarctan_w20 IS NOT NULL
        ORDER BY bar_index
    """).fetch_df()

    logger.info(f"Loaded {len(df):,} rows")
    return df


def calculate_forward_returns(df: pd.DataFrame, horizons: List[int] = [1, 5, 30]) -> pd.DataFrame:
    """Calculate forward returns at multiple horizons."""
    logger.info("Calculating forward returns...")

    for h in horizons:
        df[f'fwd_return_{h}'] = (df['close'].shift(-h) - df['close']) / 0.01  # pips

    return df


def create_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create different combination methods."""
    logger.info("Creating combination features...")

    # Method 1: Multiplication (interaction)
    df['combo_multiply'] = df['range_z'] * df['position_z']

    # Method 2: Addition (average signal)
    df['combo_add'] = df['range_z'] + df['position_z']

    # Method 3: Max absolute (dominant signal)
    df['combo_max_abs'] = np.where(
        np.abs(df['range_z']) > np.abs(df['position_z']),
        df['range_z'],
        df['position_z']
    )

    # Method 4: Logical AND (both extreme same direction)
    threshold = 0.4  # moderate threshold
    df['combo_and'] = np.where(
        (df['range_z'] > threshold) & (df['position_z'] > threshold),
        (df['range_z'] + df['position_z']) / 2,  # average when both positive
        np.where(
            (df['range_z'] < -threshold) & (df['position_z'] < -threshold),
            (df['range_z'] + df['position_z']) / 2,  # average when both negative
            0  # zero when not aligned
        )
    )

    # Method 5: Weighted by conviction (absolute value)
    total_conviction = np.abs(df['range_z']) + np.abs(df['position_z'])
    df['combo_weighted'] = np.where(
        total_conviction > 0,
        (df['range_z'] * np.abs(df['range_z']) + df['position_z'] * np.abs(df['position_z'])) / total_conviction,
        0
    )

    return df


def analyze_predictive_power(df: pd.DataFrame, feature: str, horizons: List[int] = [1, 5, 30]) -> Dict:
    """Analyze predictive power of a feature."""

    results = {
        'feature': feature,
        'horizons': {}
    }

    for h in horizons:
        fwd_col = f'fwd_return_{h}'

        # Remove NaN values
        valid_mask = ~(df[feature].isna() | df[fwd_col].isna())
        feature_vals = df.loc[valid_mask, feature].values
        returns = df.loc[valid_mask, fwd_col].values

        if len(feature_vals) < 100:
            continue

        # Correlation (manual calculation)
        corr = np.corrcoef(feature_vals, returns)[0, 1]

        # P-value approximation using t-statistic
        n = len(feature_vals)
        if abs(corr) < 1.0:
            t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
            # Two-tailed p-value approximation
            p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / (n - 2)))))
        else:
            p_value = 0.0

        # Directional accuracy (when feature is strong)
        strong_threshold = 0.5
        strong_signals = np.abs(feature_vals) > strong_threshold

        if np.sum(strong_signals) > 10:
            strong_returns = returns[strong_signals]
            strong_features = feature_vals[strong_signals]

            # Check if signal direction matches return direction
            correct_direction = np.sign(strong_features) == np.sign(strong_returns)
            directional_accuracy = np.mean(correct_direction)

            # Mean return when signal is strong
            mean_return_strong = np.mean(strong_returns)
            median_return_strong = np.median(strong_returns)

            # Separate by direction
            long_signals = strong_features > strong_threshold
            short_signals = strong_features < -strong_threshold

            mean_return_long = np.mean(returns[strong_signals][long_signals]) if np.sum(long_signals) > 0 else np.nan
            mean_return_short = np.mean(returns[strong_signals][short_signals]) if np.sum(short_signals) > 0 else np.nan
        else:
            directional_accuracy = np.nan
            mean_return_strong = np.nan
            median_return_strong = np.nan
            mean_return_long = np.nan
            mean_return_short = np.nan

        # Quantile analysis (top/bottom 10%)
        q10 = np.percentile(feature_vals, 10)
        q90 = np.percentile(feature_vals, 90)

        bottom_10_returns = returns[feature_vals <= q10]
        top_10_returns = returns[feature_vals >= q90]

        results['horizons'][h] = {
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(feature_vals),
            'n_strong_signals': int(np.sum(strong_signals)),
            'directional_accuracy': directional_accuracy,
            'mean_return_strong': mean_return_strong,
            'median_return_strong': median_return_strong,
            'mean_return_long': mean_return_long,
            'mean_return_short': mean_return_short,
            'bottom_10_mean_return': np.mean(bottom_10_returns),
            'top_10_mean_return': np.mean(top_10_returns),
            'top_bottom_spread': np.mean(top_10_returns) - np.mean(bottom_10_returns)
        }

    return results


def print_comparison_table(all_results: List[Dict]):
    """Print comparison table across all features and horizons."""

    logger.info("\n" + "="*120)
    logger.info("FEATURE COMBINATION COMPARISON")
    logger.info("="*120)

    horizons = [1, 5, 30]

    for h in horizons:
        logger.info(f"\n{'='*120}")
        logger.info(f"⏱️  {h}-MINUTE FORWARD HORIZON")
        logger.info("="*120)

        # Table header
        logger.info(f"\n{'Feature':<25} {'Corr':>8} {'P-val':>8} {'Strong#':>8} {'Dir%':>7} "
                   f"{'MeanStr':>9} {'Long':>9} {'Short':>9} {'Top-Bot':>10}")
        logger.info("-"*120)

        # Sort by absolute correlation
        horizon_results = []
        for res in all_results:
            if h in res['horizons']:
                h_data = res['horizons'][h]
                horizon_results.append((res['feature'], h_data))

        horizon_results.sort(key=lambda x: abs(x[1]['correlation']), reverse=True)

        # Print rows
        for feature, h_data in horizon_results:
            logger.info(
                f"{feature:<25} "
                f"{h_data['correlation']:>+8.4f} "
                f"{h_data['p_value']:>8.2e} "
                f"{h_data['n_strong_signals']:>8} "
                f"{h_data['directional_accuracy']*100 if not np.isnan(h_data['directional_accuracy']) else 0:>6.1f}% "
                f"{h_data['mean_return_strong']:>+9.2f} "
                f"{h_data['mean_return_long']:>+9.2f} "
                f"{h_data['mean_return_short']:>+9.2f} "
                f"{h_data['top_bottom_spread']:>+10.2f}"
            )

    # Summary: Best feature per horizon
    logger.info("\n" + "="*120)
    logger.info("🏆 BEST FEATURES BY HORIZON")
    logger.info("="*120)

    for h in horizons:
        best_corr = max(all_results, key=lambda x: abs(x['horizons'][h]['correlation']) if h in x['horizons'] else 0)
        best_dir = max(all_results, key=lambda x: x['horizons'][h]['directional_accuracy'] if h in x['horizons'] and not np.isnan(x['horizons'][h]['directional_accuracy']) else 0)
        best_spread = max(all_results, key=lambda x: x['horizons'][h]['top_bottom_spread'] if h in x['horizons'] else 0)

        logger.info(f"\n{h}-min horizon:")
        logger.info(f"  Best correlation:         {best_corr['feature']} ({best_corr['horizons'][h]['correlation']:+.4f})")
        if not np.isnan(best_dir['horizons'][h]['directional_accuracy']):
            logger.info(f"  Best directional accuracy: {best_dir['feature']} ({best_dir['horizons'][h]['directional_accuracy']*100:.1f}%)")
        logger.info(f"  Best top-bottom spread:   {best_spread['feature']} ({best_spread['horizons'][h]['top_bottom_spread']:+.2f} pips)")


def main():
    """Main execution."""
    logger.info("="*120)
    logger.info("FEATURE COMBINATION TESTING")
    logger.info("="*120)

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        return 1

    conn = duckdb.connect(str(DB_PATH))

    try:
        # Load data
        df = load_data(conn)

        # Calculate forward returns
        df = calculate_forward_returns(df)

        # Create combination features
        df = create_combination_features(df)

        # Test all features
        features_to_test = [
            'range_z',                  # Baseline: range alone
            'position_z',               # Baseline: position alone
            'combo_multiply',           # Method 1: Multiplication
            'combo_add',                # Method 2: Addition
            'combo_max_abs',            # Method 3: Max absolute
            'combo_and',                # Method 4: Logical AND
            'combo_weighted'            # Method 5: Weighted by conviction
        ]

        logger.info("\nTesting features:")
        for feat in features_to_test:
            logger.info(f"  - {feat}")

        # Analyze each feature
        all_results = []
        for feature in features_to_test:
            logger.info(f"\nAnalyzing {feature}...")
            results = analyze_predictive_power(df, feature)
            all_results.append(results)

        # Print comparison
        print_comparison_table(all_results)

        # Determine best combination
        logger.info("\n" + "="*120)
        logger.info("💡 RECOMMENDATION")
        logger.info("="*120)

        # Calculate average correlation across all horizons
        avg_corrs = {}
        for res in all_results:
            corrs = [abs(res['horizons'][h]['correlation']) for h in [1, 5, 30] if h in res['horizons']]
            avg_corrs[res['feature']] = np.mean(corrs)

        best_feature = max(avg_corrs.items(), key=lambda x: x[1])
        logger.info(f"\nBest overall feature: {best_feature[0]}")
        logger.info(f"Average |correlation|: {best_feature[1]:.4f}")

        # Add best feature to database if it's a combination
        if best_feature[0].startswith('combo_'):
            logger.info(f"\n📊 Adding {best_feature[0]} to master table...")
            conn.register('feature_data', df[['bar_index', best_feature[0]]])

            # Add column
            try:
                conn.execute(f"ALTER TABLE master ADD COLUMN {best_feature[0]} DOUBLE")
                logger.info(f"  ✅ Added column: {best_feature[0]}")
            except:
                logger.info(f"  ⚠️  Column {best_feature[0]} already exists, updating...")

            # Update values
            conn.execute(f"""
                UPDATE master
                SET {best_feature[0]} = feature_data.{best_feature[0]}
                FROM feature_data
                WHERE master.bar_index = feature_data.bar_index
            """)
            logger.info(f"  ✅ Updated {len(df):,} rows")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    logger.info("\n✅ Feature combination testing complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
