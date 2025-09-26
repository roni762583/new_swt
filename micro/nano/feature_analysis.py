#!/usr/bin/env python3
"""
Feature Analysis for Micro Trading System
Tests predictive value of 5 base features with/without time encodings
for various return horizons (1-30 minutes ahead)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import duckdb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """Analyze predictive power of features for different return horizons."""

    def __init__(self, db_path: str = "/data/micro_features.duckdb"):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

        # Define feature sets (using lag 0 = current values)
        self.base_features = [
            'position_in_range_60_0',
            'min_max_scaled_momentum_60_0',
            'min_max_scaled_rolling_range_0',
            'min_max_scaled_momentum_5_0',
            'price_change_pips_0'
        ]

        self.time_features = [
            'hour_sin_final_0',
            'hour_cos_final_0',
            'dow_sin_final_0',
            'dow_cos_final_0'
        ]

        self.horizons = [1, 2, 3, 4, 5, 10, 15, 30]

    def load_data(self, sample_size: int = 100000) -> pd.DataFrame:
        """Load features and calculate forward returns."""
        print(f"Loading {sample_size} samples from database...")

        # Load base features and time encodings
        query = f"""
        SELECT
            bar_index,
            {', '.join(self.base_features)},
            {', '.join(self.time_features)},
            price_change_pips_0
        FROM micro_features
        WHERE bar_index > 100
        ORDER BY bar_index
        LIMIT {sample_size + 50}
        """

        df = self.conn.execute(query).df()
        print(f"Loaded {len(df)} rows")

        # Calculate forward returns for each horizon
        print("Calculating forward returns...")
        for h in self.horizons:
            # Forward return in pips (cumulative price change over h bars)
            df[f'ret_{h}'] = df['price_change_pips_0'].rolling(h).sum().shift(-h)

        # Remove rows with NaN returns
        df = df.dropna()
        print(f"Final dataset: {len(df)} rows")

        return df

    def run_linear_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run OLS regression tests for each horizon."""
        results = []

        print("\nRunning regression tests...")
        print("-" * 60)

        for h in self.horizons:
            print(f"Horizon: {h} minutes")

            # Target variable
            y = df[f'ret_{h}'].values

            # Model A: Base features only
            X_base = df[self.base_features].values
            X_base = sm.add_constant(X_base)

            model_base = sm.OLS(y, X_base, missing='drop').fit()
            pred_base = model_base.predict(X_base)
            r2_base = r2_score(y, pred_base)
            mse_base = mean_squared_error(y, pred_base)

            # Model B: Base + Time features
            all_features = self.base_features + self.time_features
            X_full = df[all_features].values
            X_full = sm.add_constant(X_full)

            model_full = sm.OLS(y, X_full, missing='drop').fit()
            pred_full = model_full.predict(X_full)
            r2_full = r2_score(y, pred_full)
            mse_full = mean_squared_error(y, pred_full)

            # Calculate improvements
            r2_gain = r2_full - r2_base
            mse_reduction = (mse_base - mse_full) / mse_base * 100  # % reduction

            results.append({
                'horizon_min': h,
                'r2_base': r2_base,
                'r2_with_time': r2_full,
                'r2_gain': r2_gain,
                'r2_gain_pct': (r2_gain / max(r2_base, 0.001)) * 100,
                'mse_base': mse_base,
                'mse_with_time': mse_full,
                'mse_reduction_pct': mse_reduction,
                'base_significant': model_base.f_pvalue < 0.05,
                'time_adds_value': r2_gain > 0.001  # Meaningful improvement
            })

            print(f"  Base R²: {r2_base:.4f}, With Time R²: {r2_full:.4f}, Gain: {r2_gain:.4f}")

        return pd.DataFrame(results)

    def generate_summary(self, results: pd.DataFrame) -> str:
        """Generate verbal interpretation of results."""
        summary = []
        summary.append("=" * 70)
        summary.append("FEATURE PREDICTIVE POWER ANALYSIS SUMMARY")
        summary.append("=" * 70)
        summary.append("")

        # Overall statistics
        avg_r2_base = results['r2_base'].mean()
        avg_r2_full = results['r2_with_time'].mean()

        summary.append(f"📊 OVERALL PERFORMANCE:")
        summary.append(f"  • Average R² (base features): {avg_r2_base:.4f}")
        summary.append(f"  • Average R² (with time): {avg_r2_full:.4f}")
        summary.append(f"  • Average improvement: {(avg_r2_full - avg_r2_base):.4f}")
        summary.append("")

        # Best horizons for base features
        best_base = results.nlargest(3, 'r2_base')
        summary.append(f"🎯 BEST HORIZONS FOR BASE FEATURES:")
        for _, row in best_base.iterrows():
            summary.append(f"  • {row['horizon_min']} min: R² = {row['r2_base']:.4f}")
        summary.append("")

        # Where time features help most
        best_gains = results.nlargest(3, 'r2_gain')
        summary.append(f"⏰ TIME FEATURES MOST VALUABLE AT:")
        for _, row in best_gains.iterrows():
            summary.append(f"  • {row['horizon_min']} min: +{row['r2_gain']:.4f} R² ({row['r2_gain_pct']:.1f}% improvement)")
        summary.append("")

        # Interpretation by horizon
        summary.append("📈 HORIZON-BY-HORIZON INTERPRETATION:")
        summary.append("-" * 50)

        for _, row in results.iterrows():
            h = row['horizon_min']
            interpretation = self._interpret_horizon(row)
            summary.append(f"\n{h}-MINUTE HORIZON:")
            summary.append(f"  R² Base: {row['r2_base']:.4f} | With Time: {row['r2_with_time']:.4f}")
            summary.append(f"  MSE Reduction: {row['mse_reduction_pct']:.1f}%")
            summary.append(f"  Interpretation: {interpretation}")

        # Trading implications
        summary.append("")
        summary.append("=" * 70)
        summary.append("💡 TRADING IMPLICATIONS:")
        summary.append("-" * 50)

        # Short-term predictability
        short_term = results[results['horizon_min'] <= 5]
        if short_term['r2_base'].mean() > 0.01:
            summary.append("✅ Short-term (1-5 min): Features show predictive value")
            summary.append(f"   Average R²: {short_term['r2_base'].mean():.4f}")
        else:
            summary.append("⚠️ Short-term (1-5 min): Limited predictive power")
            summary.append(f"   Average R²: {short_term['r2_base'].mean():.4f}")

        # Medium-term predictability
        medium_term = results[results['horizon_min'].between(10, 30)]
        if medium_term['r2_base'].mean() > 0.01:
            summary.append("✅ Medium-term (10-30 min): Features show predictive value")
            summary.append(f"   Average R²: {medium_term['r2_base'].mean():.4f}")
        else:
            summary.append("⚠️ Medium-term (10-30 min): Limited predictive power")
            summary.append(f"   Average R²: {medium_term['r2_base'].mean():.4f}")

        # Time encoding value
        if results['r2_gain'].mean() > 0.005:
            summary.append("✅ Time encodings: Significantly improve predictions")
            summary.append(f"   Average gain: {results['r2_gain'].mean():.4f}")
        else:
            summary.append("⚠️ Time encodings: Minimal impact on predictions")
            summary.append(f"   Average gain: {results['r2_gain'].mean():.4f}")

        return "\n".join(summary)

    def _interpret_horizon(self, row: Dict) -> str:
        """Generate interpretation for a specific horizon."""
        r2 = row['r2_base']
        gain = row['r2_gain']

        # Base feature strength
        if r2 < 0.001:
            strength = "No predictive power"
        elif r2 < 0.01:
            strength = "Very weak"
        elif r2 < 0.05:
            strength = "Weak"
        elif r2 < 0.1:
            strength = "Moderate"
        else:
            strength = "Strong"

        # Time feature impact
        if gain < 0.001:
            time_impact = "no benefit from time"
        elif gain < 0.005:
            time_impact = "slight time benefit"
        elif gain < 0.01:
            time_impact = "moderate time benefit"
        else:
            time_impact = "strong time benefit"

        return f"{strength} predictability, {time_impact}"

    def plot_results(self, results: pd.DataFrame):
        """Create visualization of results."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: R² by horizon
        ax = axes[0, 0]
        x = results['horizon_min']
        ax.plot(x, results['r2_base'], 'o-', label='Base Features', linewidth=2)
        ax.plot(x, results['r2_with_time'], 's-', label='Base + Time', linewidth=2)
        ax.set_xlabel('Horizon (minutes)')
        ax.set_ylabel('R² Score')
        ax.set_title('Predictive Power by Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: R² Gain from time features
        ax = axes[0, 1]
        colors = ['green' if g > 0 else 'red' for g in results['r2_gain']]
        ax.bar(x, results['r2_gain'], color=colors, alpha=0.7)
        ax.set_xlabel('Horizon (minutes)')
        ax.set_ylabel('R² Gain from Time Features')
        ax.set_title('Time Feature Contribution')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)

        # Plot 3: MSE comparison
        ax = axes[1, 0]
        width = 0.35
        x_pos = np.arange(len(results))
        ax.bar(x_pos - width/2, results['mse_base'], width, label='Base', alpha=0.7)
        ax.bar(x_pos + width/2, results['mse_with_time'], width, label='With Time', alpha=0.7)
        ax.set_xlabel('Horizon (minutes)')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Prediction Error by Horizon')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results['horizon_min'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Percentage improvements
        ax = axes[1, 1]
        ax.plot(results['horizon_min'], results['r2_gain_pct'], 'go-', label='R² Gain %', linewidth=2)
        ax.plot(results['horizon_min'], results['mse_reduction_pct'], 'bs-', label='MSE Reduction %', linewidth=2)
        ax.set_xlabel('Horizon (minutes)')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Percentage Improvements from Time Features')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig('feature_analysis_results.png', dpi=150)
        plt.show()

        print("\n📊 Plots saved to feature_analysis_results.png")


def main():
    """Run complete feature analysis."""
    print("🔬 NANO FEATURE ANALYSIS")
    print("=" * 70)

    # Initialize analyzer
    analyzer = FeatureAnalyzer()

    # Load and prepare data
    df = analyzer.load_data(sample_size=100000)

    # Run regression tests
    results = analyzer.run_linear_tests(df)

    # Display results table
    print("\n📊 RESULTS TABLE:")
    print("-" * 70)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(results)

    # Generate and print summary
    summary = analyzer.generate_summary(results)
    print("\n" + summary)

    # Save results
    results.to_csv('feature_analysis_results.csv', index=False)

    with open('/workspace/feature_analysis_summary.txt', 'w') as f:
        f.write(summary)

    print("\n✅ Results saved to:")
    print("  • feature_analysis_results.csv")
    print("  • feature_analysis_summary.txt")

    # Create visualizations
    try:
        analyzer.plot_results(results)
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

    return results


if __name__ == "__main__":
    results = main()