#!/usr/bin/env python3
"""
Pandas-TA Feature Analysis for Micro Trading System
Tests predictive value of technical indicators without TA-Lib compilation issues
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import duckdb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PandasTAFeatureAnalyzer:
    """Analyze predictive power of technical indicators using pandas calculations."""

    def __init__(self, db_path: str = "/data/micro_features.duckdb"):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.horizons = [1, 2, 3, 4, 5, 10, 15, 30]

    def compute_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators using pandas.

        Indicators:
        - SMA: Simple Moving Averages
        - EMA: Exponential Moving Averages
        - RSI: Relative Strength Index
        - MACD: Moving Average Convergence Divergence
        - BB: Bollinger Bands
        - Momentum indicators
        """

        close = df['close']
        features = pd.DataFrame(index=df.index)

        # === TREND INDICATORS ===
        # Simple Moving Averages
        features['sma_5'] = close.rolling(window=5).mean()
        features['sma_10'] = close.rolling(window=10).mean()
        features['sma_20'] = close.rolling(window=20).mean()
        features['sma_50'] = close.rolling(window=50).mean()

        # Exponential Moving Averages
        features['ema_5'] = close.ewm(span=5, adjust=False).mean()
        features['ema_10'] = close.ewm(span=10, adjust=False).mean()
        features['ema_20'] = close.ewm(span=20, adjust=False).mean()

        # Price relative to moving averages
        features['price_to_sma5'] = close / features['sma_5'] - 1
        features['price_to_sma20'] = close / features['sma_20'] - 1
        features['sma5_to_sma20'] = features['sma_5'] / features['sma_20'] - 1

        # === MOMENTUM INDICATORS ===
        # RSI (Relative Strength Index)
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        features['rsi_7'] = calculate_rsi(close, 7)
        features['rsi_14'] = calculate_rsi(close, 14)
        features['rsi_21'] = calculate_rsi(close, 21)

        # RSI zones
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(float)
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(float)

        # Rate of Change (ROC)
        features['roc_5'] = (close / close.shift(5) - 1) * 100
        features['roc_10'] = (close / close.shift(10) - 1) * 100
        features['roc_20'] = (close / close.shift(20) - 1) * 100

        # Momentum
        features['mom_5'] = close - close.shift(5)
        features['mom_10'] = close - close.shift(10)
        features['mom_20'] = close - close.shift(20)

        # === MACD ===
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma_bb = close.rolling(window=bb_period).mean()
        std_bb = close.rolling(window=bb_period).std()
        features['bb_upper'] = sma_bb + (bb_std * std_bb)
        features['bb_lower'] = sma_bb - (bb_std * std_bb)
        features['bb_middle'] = sma_bb
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_width'] + 1e-10)

        # Standard Deviation (volatility proxy)
        features['std_5'] = close.rolling(window=5).std()
        features['std_10'] = close.rolling(window=10).std()
        features['std_20'] = close.rolling(window=20).std()

        # === PATTERN INDICATORS ===
        # Price position in range
        features['price_position_20'] = (close - close.rolling(20).min()) / \
                                        (close.rolling(20).max() - close.rolling(20).min() + 1e-10)
        features['price_position_50'] = (close - close.rolling(50).min()) / \
                                        (close.rolling(50).max() - close.rolling(50).min() + 1e-10)

        # === COMPOSITE FEATURES ===
        # Trend strength
        features['trend_strength'] = (features['sma_5'] - features['sma_20']).abs() / features['std_20']

        # Volatility-adjusted momentum
        features['vol_adj_mom'] = features['mom_10'] / (features['std_10'] + 1e-10)

        return features

    def load_data_with_ta(self, sample_size: int = 100000) -> Tuple[pd.DataFrame, List[str]]:
        """Load price data and compute technical indicators."""
        print(f"Loading {sample_size} samples from database...")

        # Load data
        query = f"""
        SELECT
            bar_index,
            close,
            price_change_pips_0
        FROM micro_features
        WHERE bar_index > 100
        ORDER BY bar_index
        LIMIT {sample_size + 100}
        """

        df = self.conn.execute(query).df()
        print(f"Loaded {len(df)} rows")

        # Compute TA features
        print("Computing technical indicators...")
        ta_features = self.compute_ta_features(df)

        # Combine with original data
        df = pd.concat([df, ta_features], axis=1)

        # Get list of valid feature columns (>50% non-NaN)
        feature_cols = [col for col in ta_features.columns
                       if df[col].notna().sum() > len(df) * 0.5]

        print(f"Generated {len(feature_cols)} technical indicators")

        # Calculate forward returns
        print("Calculating forward returns...")
        for h in self.horizons:
            df[f'ret_{h}'] = df['price_change_pips_0'].rolling(h).sum().shift(-h)

        # Remove NaN rows
        df = df.dropna(subset=feature_cols + [f'ret_{h}' for h in self.horizons])
        print(f"Final dataset: {len(df)} rows with {len(feature_cols)} features")

        return df, feature_cols

    def run_indicator_tests(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Test predictive power of technical indicators."""
        results = []

        print("\nRunning technical indicator regression tests...")
        print("-" * 60)

        # Group indicators by type
        feature_groups = {
            'Trend': [c for c in feature_cols if any(x in c for x in ['sma', 'ema', 'price_to'])],
            'Momentum': [c for c in feature_cols if any(x in c for x in ['rsi', 'roc', 'mom', 'macd'])],
            'Volatility': [c for c in feature_cols if any(x in c for x in ['bb_', 'std_'])],
            'Pattern': [c for c in feature_cols if 'position' in c],
            'All': feature_cols
        }

        for h in self.horizons:
            print(f"\nHorizon: {h} minutes")
            y = df[f'ret_{h}'].values
            horizon_results = {'horizon_min': h}

            for group_name, group_features in feature_groups.items():
                if len(group_features) == 0:
                    continue

                X = df[group_features].values
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X, missing='drop').fit()
                    pred = model.predict(X)
                    r2 = r2_score(y, pred)
                    mse = mean_squared_error(y, pred)

                    horizon_results[f'r2_{group_name.lower()}'] = r2
                    horizon_results[f'mse_{group_name.lower()}'] = mse
                    horizon_results[f'n_{group_name.lower()}'] = len(group_features)

                    if group_name == 'All':
                        print(f"  All indicators R²: {r2:.4f} ({len(group_features)} features)")
                except:
                    horizon_results[f'r2_{group_name.lower()}'] = 0.0
                    horizon_results[f'mse_{group_name.lower()}'] = np.nan

            results.append(horizon_results)

        return pd.DataFrame(results)

    def find_best_indicators(self, df: pd.DataFrame, feature_cols: List[str], top_n: int = 10) -> pd.DataFrame:
        """Find the most predictive individual indicators."""
        print("\n" + "=" * 60)
        print("TOP INDIVIDUAL INDICATORS")
        print("=" * 60)

        indicator_scores = []

        for feature in feature_cols:
            scores = []
            for h in self.horizons:
                y = df[f'ret_{h}'].values
                X = df[[feature]].values
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X, missing='drop').fit()
                    pred = model.predict(X)
                    r2 = r2_score(y, pred)
                    scores.append(r2)
                except:
                    scores.append(0.0)

            if scores:
                indicator_scores.append({
                    'indicator': feature,
                    'avg_r2': np.mean(scores),
                    'max_r2': np.max(scores),
                    'best_horizon': self.horizons[np.argmax(scores)]
                })

        best = pd.DataFrame(indicator_scores).sort_values('avg_r2', ascending=False).head(top_n)

        print("\nTop 10 Most Predictive Indicators:")
        for _, row in best.iterrows():
            print(f"  {row['indicator']:25s} | Avg R²: {row['avg_r2']:.4f} | Best @ {row['best_horizon']} min")

        return best


def main():
    """Run technical indicator analysis."""
    print("🔬 TECHNICAL INDICATORS ANALYSIS (Pandas Implementation)")
    print("=" * 70)

    analyzer = PandasTAFeatureAnalyzer()
    df, feature_cols = analyzer.load_data_with_ta(sample_size=100000)
    results = analyzer.run_indicator_tests(df, feature_cols)
    best = analyzer.find_best_indicators(df, feature_cols)

    # Display results
    print("\n📊 RESULTS BY HORIZON:")
    print("-" * 70)
    display_cols = ['horizon_min', 'r2_trend', 'r2_momentum', 'r2_volatility', 'r2_all']
    display_cols = [c for c in display_cols if c in results.columns]
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(results[display_cols])

    # Summary
    print("\n" + "=" * 70)
    print("💡 KEY FINDINGS:")
    print("-" * 70)

    avg_r2 = results['r2_all'].mean()
    best_horizon = results.loc[results['r2_all'].idxmax(), 'horizon_min']
    best_r2 = results['r2_all'].max()

    print(f"• Average R² across all horizons: {avg_r2:.4f}")
    print(f"• Best horizon: {best_horizon} minutes (R² = {best_r2:.4f})")
    print(f"• Top indicator: {best.iloc[0]['indicator']} (Avg R² = {best.iloc[0]['avg_r2']:.4f})")

    if avg_r2 > 0.001:
        print(f"✅ Technical indicators show {avg_r2/0.0007:.1f}x better predictive power than base features")
    else:
        print("⚠️ Technical indicators show limited predictive power")

    # Save results
    results.to_csv('pandas_ta_results.csv', index=False)
    best.to_csv('pandas_ta_best.csv', index=False)
    print("\n✅ Results saved to pandas_ta_results.csv and pandas_ta_best.csv")

    return results, best


if __name__ == "__main__":
    results, best = main()