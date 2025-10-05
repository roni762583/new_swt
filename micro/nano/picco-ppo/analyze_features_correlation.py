#!/usr/bin/env python3
"""
Correlation and Cointegration Analysis for 33 Selected Features
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import seaborn, fall back to matplotlib if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not available, using matplotlib for heatmap")

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")
OUTPUT_DIR = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo")

# 26 ML features (current optimized set)
FEATURES = [
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
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]

def load_features():
    """Load normalized features from database."""
    print("=" * 80)
    print("LOADING NORMALIZED FEATURES")
    print("=" * 80)

    conn = duckdb.connect(str(DB_PATH))

    # Build query
    feature_list = ', '.join(FEATURES)
    query = f"""
        SELECT {feature_list}
        FROM master
        WHERE pretrain_action IS NOT NULL
        ORDER BY bar_index
    """

    print(f"\nLoading {len(FEATURES)} normalized features...")
    df = conn.execute(query).fetch_df()
    conn.close()

    print(f"Loaded {len(df):,} rows")
    print(f"Features: {len(df.columns)}")

    # Drop NaNs
    df_clean = df.dropna()
    print(f"After dropping NaNs: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}%)")

    return df_clean


def plot_correlation_heatmap(df):
    """Generate correlation heatmap."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    # Calculate correlation matrix
    corr = df.corr()

    # Summary statistics
    print(f"\nCorrelation Matrix Shape: {corr.shape}")

    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) > 0.7:  # Threshold for high correlation
                high_corr_pairs.append((corr.columns[i], corr.columns[j], corr_val))

    print(f"\n🔴 HIGH CORRELATION PAIRS (|r| > 0.7):")
    if high_corr_pairs:
        for feat1, feat2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {feat1} ↔ {feat2}: {corr_val:+.3f}")
    else:
        print("  None found - Good feature independence!")

    # Moderate correlation
    mod_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if 0.5 < abs(corr_val) <= 0.7:
                mod_corr_pairs.append((corr.columns[i], corr.columns[j], corr_val))

    print(f"\n🟡 MODERATE CORRELATION PAIRS (0.5 < |r| ≤ 0.7):")
    if mod_corr_pairs:
        for feat1, feat2, corr_val in sorted(mod_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"  {feat1} ↔ {feat2}: {corr_val:+.3f}")

    # Plot heatmap
    print("\n📊 Generating correlation heatmap...")

    plt.figure(figsize=(20, 18))

    if HAS_SEABORN:
        # Use seaborn for nicer heatmap
        sns.heatmap(corr,
                    annot=False,  # Too many features to annotate
                    cmap='RdBu_r',  # Red-Blue diverging
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"})
    else:
        # Use matplotlib imshow
        im = plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, label='Pearson Correlation', shrink=0.8)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, ha='right', fontsize=8)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)

    plt.title('Correlation Heatmap - 26 ML Features', fontsize=16, fontweight='bold')
    if HAS_SEABORN:
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "correlation_heatmap_26features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    # Save correlation matrix to CSV
    csv_path = OUTPUT_DIR / "correlation_matrix_26features.csv"
    corr.to_csv(csv_path)
    print(f"✅ Saved: {csv_path}")

    return corr


def analyze_cointegration(df):
    """
    Check for cointegration between feature pairs.

    Note: Cointegration typically applies to non-stationary time series.
    Most of our features are already stationary (returns, z-scores).
    We'll focus on features that might have unit roots.
    """
    print("\n" + "=" * 80)
    print("COINTEGRATION ANALYSIS")
    print("=" * 80)

    # Features that might be non-stationary
    non_stationary_candidates = [
        'atr_14',  # Raw volatility measure
        'efficiency_ratio_h1',  # Could have trends
        'vol_ratio_deviation',  # Deviation measure
    ]

    available_candidates = [f for f in non_stationary_candidates if f in df.columns]

    if len(available_candidates) < 2:
        print("\n⚠️  Not enough non-stationary candidates for cointegration testing")
        print("   Most features are already stationary (returns, z-scores)")
        return

    print(f"\nTesting {len(available_candidates)} potentially non-stationary features:")
    for feat in available_candidates:
        print(f"  - {feat}")

    try:
        from statsmodels.tsa.stattools import coint

        print("\n📊 Pairwise Cointegration Tests (Engle-Granger):")
        print("   H0: No cointegration (p > 0.05)")
        print("   H1: Cointegrated (p ≤ 0.05)")

        cointegrated_pairs = []

        for i, feat1 in enumerate(available_candidates):
            for feat2 in available_candidates[i+1:]:
                _, p_value, _ = coint(df[feat1], df[feat2])

                if p_value <= 0.05:
                    cointegrated_pairs.append((feat1, feat2, p_value))
                    print(f"\n  ✅ COINTEGRATED: {feat1} ↔ {feat2}")
                    print(f"     p-value: {p_value:.6f}")

        if not cointegrated_pairs:
            print("\n  ✅ No cointegration detected - Features are independent")

    except ImportError:
        print("\n⚠️  statsmodels not available, skipping cointegration test")
        print("   Install with: pip install statsmodels")


def feature_importance_by_variance(df):
    """Analyze feature importance by variance contribution."""
    print("\n" + "=" * 80)
    print("FEATURE VARIANCE ANALYSIS")
    print("=" * 80)

    variances = df.var().sort_values(ascending=False)

    print("\n📊 Top 10 Features by Variance:")
    for feat, var in variances.head(10).items():
        print(f"  {feat}: {var:.6f}")

    print("\n📊 Bottom 10 Features by Variance:")
    for feat, var in variances.tail(10).items():
        print(f"  {feat}: {var:.6f}")

    # Plot variance distribution
    plt.figure(figsize=(14, 6))
    variances.plot(kind='bar')
    plt.title('Feature Variance Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "feature_variance_26features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 80)
    print("FEATURE CORRELATION & COINTEGRATION ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing {len(FEATURES)} normalized features")
    print(f"Database: {DB_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Load data
    df = load_features()

    # Correlation analysis
    corr = plot_correlation_heatmap(df)

    # Variance analysis
    feature_importance_by_variance(df)

    # Cointegration analysis
    analyze_cointegration(df)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - correlation_heatmap_26features.png")
    print(f"  - correlation_matrix_26features.csv")
    print(f"  - feature_variance_26features.png")


if __name__ == "__main__":
    main()
