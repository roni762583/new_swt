#!/usr/bin/env python3
"""
SWT Feature Analysis for Micro Trading System
Computes Wavelet Scattering Transform features from price data
and tests their predictive value for various return horizons
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import duckdb
from typing import Dict, List, Tuple
import warnings
import pywt
from scipy import signal
warnings.filterwarnings('ignore')


class SWTFeatureAnalyzer:
    """Analyze predictive power of SWT features for different return horizons."""

    def __init__(self, db_path: str = "/data/micro_features.duckdb"):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

        # SWT parameters
        self.wavelet = 'db4'  # Daubechies-4 wavelet
        self.levels = 5  # 5 levels of decomposition
        self.horizons = [1, 2, 3, 4, 5, 10, 15, 30]

    def compute_swt_features(self, prices: np.ndarray, window: int = 60) -> np.ndarray:
        """
        Compute SWT features from price series.

        Simple version using PyWavelets for wavelet decomposition.
        Returns energy and statistics at each scale.
        """
        features_list = []

        for i in range(len(prices) - window + 1):
            window_prices = prices[i:i+window]

            # Normalize prices
            normalized = (window_prices - np.mean(window_prices)) / (np.std(window_prices) + 1e-8)

            # Compute discrete wavelet transform
            coeffs = pywt.wavedec(normalized, self.wavelet, level=self.levels)

            features = []

            # For each level, compute statistical features
            for j, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    # Energy at this scale
                    energy = np.sum(coeff ** 2) / len(coeff)

                    # Mean absolute value (L1 norm)
                    mean_abs = np.mean(np.abs(coeff))

                    # Standard deviation
                    std = np.std(coeff)

                    # Maximum coefficient magnitude
                    max_coeff = np.max(np.abs(coeff))

                    features.extend([energy, mean_abs, std, max_coeff])

            # Add cross-scale features
            if len(coeffs) >= 2:
                # Ratio of energies between scales
                for j in range(len(coeffs) - 1):
                    if len(coeffs[j]) > 0 and len(coeffs[j+1]) > 0:
                        energy_ratio = np.sum(coeffs[j]**2) / (np.sum(coeffs[j+1]**2) + 1e-8)
                        features.append(np.log1p(energy_ratio))

            features_list.append(features)

        # Pad with zeros for the initial window
        padding = [[0] * len(features_list[0])] * (window - 1)
        features_list = padding + features_list

        return np.array(features_list, dtype=np.float32)

    def load_data_with_swt(self, sample_size: int = 50000) -> pd.DataFrame:
        """Load price data and compute SWT features."""
        print(f"Loading {sample_size} samples from database...")

        # Load raw prices and base features for comparison
        query = f"""
        SELECT
            bar_index,
            close,
            position_in_range_60_0,
            min_max_scaled_momentum_60_0,
            min_max_scaled_rolling_range_0,
            min_max_scaled_momentum_5_0,
            price_change_pips_0
        FROM micro_features
        WHERE bar_index > 100
        ORDER BY bar_index
        LIMIT {sample_size + 100}
        """

        df = self.conn.execute(query).df()
        print(f"Loaded {len(df)} rows")

        # Compute SWT features
        print("Computing SWT features...")
        prices = df['close'].values
        swt_features = self.compute_swt_features(prices)

        # Add SWT features to dataframe
        n_swt_features = swt_features.shape[1]
        for i in range(n_swt_features):
            df[f'swt_{i}'] = swt_features[:, i]

        print(f"Generated {n_swt_features} SWT features")

        # Calculate forward returns for each horizon
        print("Calculating forward returns...")
        for h in self.horizons:
            # Forward return in pips (cumulative price change over h bars)
            df[f'ret_{h}'] = df['price_change_pips_0'].rolling(h).sum().shift(-h)

        # Remove rows with NaN returns
        df = df.dropna()
        print(f"Final dataset: {len(df)} rows with {n_swt_features} SWT features")

        return df, n_swt_features

    def run_comparative_tests(self, df: pd.DataFrame, n_swt_features: int) -> pd.DataFrame:
        """Run regression tests comparing base features vs SWT features."""
        results = []

        print("\nRunning comparative regression tests...")
        print("-" * 60)

        # Define feature sets
        base_features = [
            'position_in_range_60_0',
            'min_max_scaled_momentum_60_0',
            'min_max_scaled_rolling_range_0',
            'min_max_scaled_momentum_5_0',
            'price_change_pips_0'
        ]

        swt_features = [f'swt_{i}' for i in range(n_swt_features)]

        for h in self.horizons:
            print(f"Horizon: {h} minutes")

            # Target variable
            y = df[f'ret_{h}'].values

            # Model A: Base features only (5D)
            X_base = df[base_features].values
            X_base = sm.add_constant(X_base)

            model_base = sm.OLS(y, X_base, missing='drop').fit()
            pred_base = model_base.predict(X_base)
            r2_base = r2_score(y, pred_base)
            mse_base = mean_squared_error(y, pred_base)

            # Model B: SWT features only
            X_swt = df[swt_features].values
            X_swt = sm.add_constant(X_swt)

            model_swt = sm.OLS(y, X_swt, missing='drop').fit()
            pred_swt = model_swt.predict(X_swt)
            r2_swt = r2_score(y, pred_swt)
            mse_swt = mean_squared_error(y, pred_swt)

            # Model C: Combined (Base + SWT)
            all_features = base_features + swt_features
            X_combined = df[all_features].values
            X_combined = sm.add_constant(X_combined)

            model_combined = sm.OLS(y, X_combined, missing='drop').fit()
            pred_combined = model_combined.predict(X_combined)
            r2_combined = r2_score(y, pred_combined)
            mse_combined = mean_squared_error(y, pred_combined)

            # Calculate improvements
            swt_vs_base = r2_swt - r2_base
            combined_vs_base = r2_combined - r2_base
            combined_vs_swt = r2_combined - r2_swt

            results.append({
                'horizon_min': h,
                'r2_base_5d': r2_base,
                'r2_swt': r2_swt,
                'r2_combined': r2_combined,
                'swt_vs_base_gain': swt_vs_base,
                'combined_vs_base_gain': combined_vs_base,
                'combined_vs_swt_gain': combined_vs_swt,
                'mse_base': mse_base,
                'mse_swt': mse_swt,
                'mse_combined': mse_combined,
                'n_base_features': len(base_features),
                'n_swt_features': n_swt_features,
                'n_combined_features': len(all_features)
            })

            print(f"  Base(5D) R²: {r2_base:.4f}, SWT R²: {r2_swt:.4f}, Combined R²: {r2_combined:.4f}")
            print(f"  SWT gain over base: {swt_vs_base:+.4f}")

        return pd.DataFrame(results)

    def generate_summary(self, results: pd.DataFrame) -> str:
        """Generate verbal interpretation of SWT analysis results."""
        summary = []
        summary.append("=" * 70)
        summary.append("SWT (WAVELET SCATTERING) FEATURE ANALYSIS SUMMARY")
        summary.append("=" * 70)
        summary.append("")

        # Overall statistics
        avg_r2_base = results['r2_base_5d'].mean()
        avg_r2_swt = results['r2_swt'].mean()
        avg_r2_combined = results['r2_combined'].mean()

        summary.append(f"📊 OVERALL PERFORMANCE:")
        summary.append(f"  • Average R² (5 base features): {avg_r2_base:.4f}")
        summary.append(f"  • Average R² (SWT features): {avg_r2_swt:.4f}")
        summary.append(f"  • Average R² (Combined): {avg_r2_combined:.4f}")
        summary.append("")

        # Best horizons for SWT
        best_swt = results.nlargest(3, 'r2_swt')
        summary.append(f"🌊 BEST HORIZONS FOR SWT FEATURES:")
        for _, row in best_swt.iterrows():
            summary.append(f"  • {row['horizon_min']} min: R² = {row['r2_swt']:.4f}")
        summary.append("")

        # Where SWT beats base features
        swt_wins = results[results['swt_vs_base_gain'] > 0]
        if len(swt_wins) > 0:
            summary.append(f"✅ SWT OUTPERFORMS BASE FEATURES AT:")
            for _, row in swt_wins.iterrows():
                gain_pct = (row['swt_vs_base_gain'] / max(row['r2_base_5d'], 0.0001)) * 100
                summary.append(f"  • {row['horizon_min']} min: +{row['swt_vs_base_gain']:.4f} R² ({gain_pct:.1f}% improvement)")
        else:
            summary.append(f"⚠️ SWT does not outperform base features at any horizon")
        summary.append("")

        # Combined model performance
        summary.append(f"🔄 COMBINED MODEL INSIGHTS:")
        best_combined = results.nlargest(3, 'combined_vs_base_gain')
        for _, row in best_combined.iterrows():
            summary.append(f"  • {row['horizon_min']} min: Combined R² = {row['r2_combined']:.4f} (+{row['combined_vs_base_gain']:.4f} over base)")
        summary.append("")

        # Feature efficiency
        summary.append(f"📈 FEATURE EFFICIENCY:")
        n_swt = results['n_swt_features'].iloc[0]
        summary.append(f"  • Base features: 5 dimensions")
        summary.append(f"  • SWT features: {n_swt} dimensions")

        # R² per feature dimension
        r2_per_dim_base = avg_r2_base / 5
        r2_per_dim_swt = avg_r2_swt / n_swt
        summary.append(f"  • R² per dimension (base): {r2_per_dim_base:.6f}")
        summary.append(f"  • R² per dimension (SWT): {r2_per_dim_swt:.6f}")

        if r2_per_dim_swt > r2_per_dim_base:
            summary.append(f"  • SWT features are {r2_per_dim_swt/r2_per_dim_base:.1f}x more efficient per dimension")
        else:
            summary.append(f"  • Base features are {r2_per_dim_base/r2_per_dim_swt:.1f}x more efficient per dimension")

        # Trading implications
        summary.append("")
        summary.append("=" * 70)
        summary.append("💡 TRADING IMPLICATIONS:")
        summary.append("-" * 50)

        if avg_r2_swt > avg_r2_base * 1.5:
            summary.append("✅ SWT features show significant improvement over base features")
            summary.append(f"   Average improvement: {(avg_r2_swt/avg_r2_base - 1)*100:.1f}%")
        elif avg_r2_swt > avg_r2_base:
            summary.append("🟡 SWT features show marginal improvement over base features")
            summary.append(f"   Average improvement: {(avg_r2_swt/avg_r2_base - 1)*100:.1f}%")
        else:
            summary.append("⚠️ SWT features do not improve upon base features")
            summary.append("   Consider alternative feature engineering approaches")

        return "\n".join(summary)


def main():
    """Run complete SWT feature analysis."""
    print("🔬 SWT (WAVELET SCATTERING) FEATURE ANALYSIS")
    print("=" * 70)

    # Initialize analyzer
    analyzer = SWTFeatureAnalyzer()

    # Load data and compute SWT features
    df, n_swt_features = analyzer.load_data_with_swt(sample_size=50000)

    # Run comparative regression tests
    results = analyzer.run_comparative_tests(df, n_swt_features)

    # Display results table
    print("\n📊 RESULTS TABLE:")
    print("-" * 70)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(results[['horizon_min', 'r2_base_5d', 'r2_swt', 'r2_combined', 'swt_vs_base_gain']])

    # Generate and print summary
    summary = analyzer.generate_summary(results)
    print("\n" + summary)

    # Save results
    results.to_csv('swt_analysis_results.csv', index=False)

    with open('/workspace/swt_analysis_summary.txt', 'w') as f:
        f.write(summary)

    print("\n✅ Results saved to:")
    print("  • swt_analysis_results.csv")
    print("  • swt_analysis_summary.txt")

    return results


if __name__ == "__main__":
    results = main()