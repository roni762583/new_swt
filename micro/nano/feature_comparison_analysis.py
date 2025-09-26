#!/usr/bin/env python3
"""
Feature Comparison Analysis:
- Reactive/LessReactive features vs Swing-based regime detection
Compares predictive power, stability, and RL suitability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
# from scipy import stats  # Not available
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureComparisonAnalyzer:
    """Compare reactive/lessreactive features with swing regimes"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.data = None

    def load_data(self, start_idx: int = 10000, end_idx: int = 14000):
        """Load data and compute both feature sets"""
        conn = duckdb.connect(self.db_path, read_only=True)

        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {end_idx}
        ORDER BY bar_index
        """

        self.data = pd.read_sql(query, conn)
        conn.close()

        print(f"Loaded {len(self.data)} bars")

    def compute_reactive_features(self) -> pd.DataFrame:
        """Compute reactive/lessreactive/ratio features"""
        df = self.data.copy()

        # Calculate SMAs
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma200'] = df['close'].rolling(200).mean()

        # Reactive and LessReactive features
        df['reactive'] = df['close'] - df['sma200']
        df['lessreactive'] = df['sma20'] - df['sma200']

        # Ratio with safety for division
        df['ratio'] = np.where(
            np.abs(df['lessreactive']) > 0.0001,
            df['reactive'] / df['lessreactive'],
            0
        )

        # Clip extreme values
        df['ratio'] = np.clip(df['ratio'], -10, 10)

        return df

    def compute_swing_regimes(self) -> pd.DataFrame:
        """Compute swing-based regime features using our existing approach"""
        from swing_state_tracker import SwingStateTracker, SwingState

        df = self.data.copy()
        tracker = SwingStateTracker(k=3)

        # Get swing analysis
        results = tracker.analyze(df)

        # Create regime encoding
        regime_map = {
            SwingState.HHHL: 2,   # Bullish
            SwingState.LHLL: -2,  # Bearish
            SwingState.HHLL: 1,   # Expansion
            SwingState.LHHL: -1,  # Contraction
            SwingState.UNDEFINED: 0
        }

        # Map states to numerical values
        df['swing_regime'] = 0  # Default

        # Apply state history
        for bar_idx, state in results['state_history']:
            idx = df[df['bar_index'] == bar_idx].index
            if len(idx) > 0:
                pos = idx[0]
                df.loc[pos:, 'swing_regime'] = regime_map[state]

        # Add swing-based features
        df['confirmed_swings'] = 0
        for swing in results['confirmed_swings']:
            idx = df[df['bar_index'] == swing.index].index
            if len(idx) > 0:
                df.loc[idx[0], 'confirmed_swings'] = 1 if swing.type == 'high' else -1

        # Distance to last swing
        df['bars_since_swing'] = 0
        last_swing_idx = 0
        for i in range(len(df)):
            if df.iloc[i]['confirmed_swings'] != 0:
                last_swing_idx = i
            df.iloc[i, df.columns.get_loc('bars_since_swing')] = i - last_swing_idx

        return df

    def compute_predictive_power(self, features_df: pd.DataFrame,
                                feature_cols: List[str],
                                horizon: int = 20) -> Dict:
        """Measure predictive power of features for future returns"""

        # Calculate future returns
        features_df['future_return'] = features_df['close'].shift(-horizon) / features_df['close'] - 1

        # Drop NaN rows
        clean_df = features_df[feature_cols + ['future_return']].dropna()

        results = {}

        for col in feature_cols:
            # Correlation with future returns
            corr = clean_df[col].corr(clean_df['future_return'])

            # Information Coefficient (IC) - using rank correlation approximation
            # Rank-based correlation without scipy
            rank_x = clean_df[col].rank()
            rank_y = clean_df['future_return'].rank()
            ic = rank_x.corr(rank_y)  # Pearson on ranks approximates Spearman

            # Predictive R-squared using correlation
            r2 = corr ** 2  # R-squared from correlation

            # Simple feature importance using correlation magnitude
            feature_importance = abs(corr)

            results[col] = {
                'correlation': corr if not np.isnan(corr) else 0,
                'ic': ic if not np.isnan(ic) else 0,
                'r2': r2 if not np.isnan(r2) else 0,
                'importance': feature_importance if not np.isnan(feature_importance) else 0
            }

        return results

    def compute_stability_metrics(self, features_df: pd.DataFrame,
                                 feature_cols: List[str]) -> Dict:
        """Measure stability and noise characteristics of features"""

        results = {}

        for col in feature_cols:
            if col not in features_df.columns:
                continue

            series = features_df[col].dropna()

            # Stability metrics
            results[col] = {
                'std': series.std(),
                'cv': series.std() / (series.mean() + 1e-10),  # Coefficient of variation
                'autocorr_1': series.autocorr(1) if len(series) > 1 else 0,  # Lag-1 autocorrelation
                'autocorr_5': series.autocorr(5) if len(series) > 5 else 0,
                'noise_ratio': 1 - abs(series.autocorr(1)) if len(series) > 1 else 1,  # Higher = noisier
                'zero_crossings': np.sum(np.diff(np.sign(series)) != 0) / len(series)  # Oscillation frequency
            }

        return results

    def compute_rl_suitability(self, features_df: pd.DataFrame,
                              feature_sets: Dict[str, List[str]]) -> Dict:
        """Assess suitability for RL based on various criteria"""

        results = {}

        for name, feature_cols in feature_sets.items():
            # Markov property test - does current state contain sufficient info?
            markov_score = self._test_markov_property(features_df, feature_cols)

            # Stationarity test
            stationarity_score = self._test_stationarity(features_df, feature_cols)

            # Signal-to-noise ratio
            snr_score = self._compute_snr(features_df, feature_cols)

            # Dimensionality efficiency (information per feature)
            dim_efficiency = self._compute_dimensional_efficiency(features_df, feature_cols)

            results[name] = {
                'markov_score': markov_score,
                'stationarity': stationarity_score,
                'snr': snr_score,
                'dim_efficiency': dim_efficiency,
                'overall_score': np.mean([markov_score, stationarity_score, snr_score, dim_efficiency])
            }

        return results

    def _test_markov_property(self, df: pd.DataFrame, features: List[str]) -> float:
        """Test if features satisfy Markov property (current state predicts future)"""
        clean_df = df[features].dropna()
        if len(clean_df) < 100:
            return 0.0

        # Can current features predict next state better than just using history mean?
        # Simplified version using correlation instead of RF

        # Predict next price move
        y = df['close'].shift(-1).fillna(method='ffill')
        X_current = clean_df

        # Split data
        split = int(0.7 * len(clean_df))
        X_train, X_test = X_current[:split], X_current[split:]
        y_train, y_test = y[:split], y[split:]

        # Simplified: use correlation as proxy for predictive power
        correlations = []
        for col in features:
            if col in X_train.columns:
                corr = X_train[col].corr(y_train)
                correlations.append(abs(corr))
        score_features = np.mean(correlations) if correlations else 0

        # Baseline: historical mean
        baseline_score = 1 - np.mean((y_test - y_train.mean())**2) / np.var(y_test)

        # Markov score: how much better than baseline?
        markov_score = max(0, (score_features - baseline_score) / (abs(baseline_score) + 1e-10))
        return np.clip(markov_score, 0, 1)

    def _test_stationarity(self, df: pd.DataFrame, features: List[str]) -> float:
        """Test stationarity using rolling statistics"""
        scores = []
        for col in features:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 100:
                continue

            # Check if rolling mean and std are stable
            rolling_mean = series.rolling(window=50).mean()
            rolling_std = series.rolling(window=50).std()

            # Measure variation in rolling statistics
            mean_stability = 1 - (rolling_mean.std() / (abs(rolling_mean.mean()) + 1e-10))
            std_stability = 1 - (rolling_std.std() / (rolling_std.mean() + 1e-10))

            score = (mean_stability + std_stability) / 2
            scores.append(np.clip(score, 0, 1))

        return np.mean(scores) if scores else 0.0

    def _compute_snr(self, df: pd.DataFrame, features: List[str]) -> float:
        """Compute signal-to-noise ratio"""
        snr_values = []

        for col in features:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) < 20:
                continue

            # Signal: low-frequency component (smoothed)
            signal = series.rolling(20, center=True).mean().fillna(method='bfill').fillna(method='ffill')

            # Noise: high-frequency component
            noise = series - signal

            # SNR in dB
            signal_power = np.mean(signal**2)
            noise_power = np.mean(noise**2)

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # Normalize to 0-1 scale (assuming SNR typically between -10 and 20 dB)
                snr_norm = (snr_db + 10) / 30
                snr_values.append(np.clip(snr_norm, 0, 1))

        return np.mean(snr_values) if snr_values else 0.0

    def _compute_dimensional_efficiency(self, df: pd.DataFrame, features: List[str]) -> float:
        """Compute information efficiency per dimension"""
        # Simplified version without sklearn PCA

        clean_df = df[features].dropna()
        if len(clean_df) < 100 or len(features) == 0:
            return 0.0

        # Standardize features manually
        X_scaled = clean_df.copy()
        for col in X_scaled.columns:
            mean = X_scaled[col].mean()
            std = X_scaled[col].std()
            if std > 0:
                X_scaled[col] = (X_scaled[col] - mean) / std

        # Measure correlation between features (high correlation = information concentration)
        corr_matrix = X_scaled.corr()
        # Average absolute correlation as proxy for concentration
        off_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        concentration = np.mean(np.abs(off_diagonal)) if len(off_diagonal) > 0 else 0

        # Penalize for too many features (efficiency)
        efficiency_penalty = 1.0 / (1 + np.log(len(features)))

        return concentration * efficiency_penalty

    def run_comparison(self):
        """Run complete comparison analysis"""
        print("="*60)
        print("FEATURE COMPARISON ANALYSIS")
        print("="*60)

        # Compute both feature sets
        print("\n1. Computing reactive/lessreactive features...")
        reactive_df = self.compute_reactive_features()

        print("2. Computing swing regime features...")
        swing_df = self.compute_swing_regimes()

        # Merge dataframes keeping close price
        merged_df = pd.concat([
            reactive_df[['close', 'reactive', 'lessreactive', 'ratio']],
            swing_df[['swing_regime', 'confirmed_swings', 'bars_since_swing']]
        ], axis=1)

        # Define feature sets
        reactive_features = ['reactive', 'lessreactive', 'ratio']
        swing_features = ['swing_regime', 'bars_since_swing']

        # 1. Predictive Power Analysis
        print("\n3. Analyzing predictive power...")
        print("-" * 40)

        for horizon in [5, 20, 60]:  # 5min, 20min, 60min
            print(f"\nHorizon: {horizon} bars")

            reactive_pred = self.compute_predictive_power(merged_df, reactive_features, horizon)
            swing_pred = self.compute_predictive_power(merged_df, swing_features, horizon)

            print("\nReactive Features:")
            for feat, metrics in reactive_pred.items():
                print(f"  {feat:15s}: IC={metrics['ic']:+.3f}, R²={metrics['r2']:.4f}")

            print("\nSwing Features:")
            for feat, metrics in swing_pred.items():
                print(f"  {feat:15s}: IC={metrics['ic']:+.3f}, R²={metrics['r2']:.4f}")

        # 2. Stability Analysis
        print("\n4. Analyzing stability...")
        print("-" * 40)

        reactive_stability = self.compute_stability_metrics(merged_df, reactive_features)
        swing_stability = self.compute_stability_metrics(merged_df, swing_features)

        print("\nReactive Features Stability:")
        for feat, metrics in reactive_stability.items():
            print(f"  {feat:15s}: Noise={metrics['noise_ratio']:.3f}, "
                  f"AutoCorr={metrics['autocorr_1']:.3f}, "
                  f"Oscillation={metrics['zero_crossings']:.3f}")

        print("\nSwing Features Stability:")
        for feat, metrics in swing_stability.items():
            print(f"  {feat:15s}: Noise={metrics['noise_ratio']:.3f}, "
                  f"AutoCorr={metrics['autocorr_1']:.3f}, "
                  f"Oscillation={metrics['zero_crossings']:.3f}")

        # 3. RL Suitability Analysis
        print("\n5. Analyzing RL suitability...")
        print("-" * 40)

        feature_sets = {
            'Reactive/LessReactive': reactive_features,
            'Swing Regimes': swing_features,
            'Combined': reactive_features + swing_features
        }

        rl_scores = self.compute_rl_suitability(merged_df, feature_sets)

        print("\nRL Suitability Scores (0-1, higher=better):")
        print("-" * 40)
        for name, scores in rl_scores.items():
            print(f"\n{name}:")
            print(f"  Markov Property: {scores['markov_score']:.3f}")
            print(f"  Stationarity:    {scores['stationarity']:.3f}")
            print(f"  Signal/Noise:    {scores['snr']:.3f}")
            print(f"  Efficiency:      {scores['dim_efficiency']:.3f}")
            print(f"  OVERALL SCORE:   {scores['overall_score']:.3f}")

        # 4. Create visualization
        self.create_comparison_plots(merged_df, reactive_features, swing_features)

        return merged_df, rl_scores

    def create_comparison_plots(self, df: pd.DataFrame,
                               reactive_features: List[str],
                               swing_features: List[str]):
        """Create visualization comparing both feature sets"""

        fig, axes = plt.subplots(4, 2, figsize=(15, 12))

        # Sample for plotting
        plot_df = df.iloc[200:600]  # Skip initial NaN values

        # 1. Reactive features
        ax = axes[0, 0]
        ax.plot(plot_df.index, plot_df['reactive'], label='Reactive', alpha=0.7)
        ax.plot(plot_df.index, plot_df['lessreactive'], label='LessReactive', alpha=0.7)
        ax.set_title('Reactive/LessReactive Features')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Ratio feature
        ax = axes[0, 1]
        ax.plot(plot_df.index, plot_df['ratio'], label='Ratio', color='purple', alpha=0.7)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Ratio Feature (Reactive/LessReactive)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Swing regime
        ax = axes[1, 0]
        ax.plot(plot_df.index, plot_df['swing_regime'], label='Regime', drawstyle='steps-post')
        ax.fill_between(plot_df.index, 0, plot_df['swing_regime'],
                        where=(plot_df['swing_regime'] > 0), color='green', alpha=0.3)
        ax.fill_between(plot_df.index, 0, plot_df['swing_regime'],
                        where=(plot_df['swing_regime'] < 0), color='red', alpha=0.3)
        ax.set_title('Swing Regime States')
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['Bear', 'Contract', 'Undef', 'Expand', 'Bull'])
        ax.grid(True, alpha=0.3)

        # 4. Bars since swing
        ax = axes[1, 1]
        ax.plot(plot_df.index, plot_df['bars_since_swing'], label='Bars Since Swing', color='orange')
        ax.set_title('Distance from Last Swing')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Feature distributions
        ax = axes[2, 0]
        ax.hist([plot_df['reactive'].dropna(), plot_df['lessreactive'].dropna()],
               bins=30, label=['Reactive', 'LessReactive'], alpha=0.5)
        ax.set_title('Feature Distributions (Reactive)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()

        ax = axes[2, 1]
        ax.hist(plot_df['swing_regime'].dropna(), bins=5, alpha=0.7, color='purple')
        ax.set_title('Regime Distribution')
        ax.set_xlabel('Regime State')
        ax.set_ylabel('Frequency')

        # 6. Correlation with price changes
        ax = axes[3, 0]
        price_change = df['close'].pct_change(20)
        ax.scatter(plot_df['ratio'].dropna(), price_change[plot_df.index].dropna(),
                  alpha=0.5, s=1)
        ax.set_title('Ratio vs 20-bar Returns')
        ax.set_xlabel('Ratio')
        ax.set_ylabel('20-bar Return')
        ax.grid(True, alpha=0.3)

        ax = axes[3, 1]
        for regime in plot_df['swing_regime'].unique():
            if pd.isna(regime):
                continue
            mask = plot_df['swing_regime'] == regime
            regime_returns = price_change[plot_df.index][mask].dropna()
            ax.hist(regime_returns, bins=20, alpha=0.5, label=f'Regime {regime:.0f}')
        ax.set_title('Returns by Regime')
        ax.set_xlabel('20-bar Return')
        ax.set_ylabel('Frequency')
        ax.legend()

        plt.suptitle('Feature Comparison: Reactive/LessReactive vs Swing Regimes', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to feature_comparison.png")


def main():
    """Run feature comparison analysis"""
    analyzer = FeatureComparisonAnalyzer()

    # Load data
    print("Loading data...")
    analyzer.load_data(start_idx=10000, end_idx=14000)

    # Run comparison
    merged_df, rl_scores = analyzer.run_comparison()

    # Save results
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR RL IMPLEMENTATION")
    print("="*60)

    best_approach = max(rl_scores.items(), key=lambda x: x[1]['overall_score'])
    print(f"\n✓ Best approach: {best_approach[0]} (score: {best_approach[1]['overall_score']:.3f})")

    print("\nKey Findings:")
    print("1. Reactive features provide continuous signals but may be noisy")
    print("2. Swing regimes are discrete/stable but may miss nuances")
    print("3. Combined approach leverages both perspectives")

    print("\nFor RL Agent State Space:")
    print("- Use reactive/ratio for continuous market distance measures")
    print("- Use swing regimes for discrete context/mode switching")
    print("- Consider hierarchical RL with regimes as high-level states")


if __name__ == "__main__":
    main()