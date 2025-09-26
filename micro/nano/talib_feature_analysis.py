#!/usr/bin/env python3
"""
TA-Lib Feature Analysis for Micro Trading System
Tests predictive value of popular technical indicators for various return horizons
"""

import pandas as pd
import numpy as np
import talib
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import duckdb
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TALibFeatureAnalyzer:
    """Analyze predictive power of TA-Lib indicators for different return horizons."""

    def __init__(self, db_path: str = "/data/micro_features.duckdb"):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)
        self.horizons = [1, 2, 3, 4, 5, 10, 15, 30]

    def compute_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute popular TA-Lib indicators.

        Selected indicators covering different categories:
        - Trend: SMA, EMA, MACD
        - Momentum: RSI, STOCH, MOM
        - Volatility: ATR, BBANDS
        - Volume: OBV, AD (if volume available)
        """

        # Extract OHLCV data
        high = df['close'].values  # Using close as high for now
        low = df['close'].values   # Using close as low for now
        close = df['close'].values
        volume = np.ones_like(close) * 1000  # Dummy volume if not available

        # Calculate multiple timeframes for key indicators
        features = pd.DataFrame(index=df.index)

        # === TREND INDICATORS ===
        # Simple Moving Averages
        features['sma_5'] = talib.SMA(close, timeperiod=5)
        features['sma_20'] = talib.SMA(close, timeperiod=20)
        features['sma_50'] = talib.SMA(close, timeperiod=50)

        # Exponential Moving Averages
        features['ema_5'] = talib.EMA(close, timeperiod=5)
        features['ema_20'] = talib.EMA(close, timeperiod=20)

        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist

        # === MOMENTUM INDICATORS ===
        # RSI (Relative Strength Index)
        features['rsi_14'] = talib.RSI(close, timeperiod=14)
        features['rsi_7'] = talib.RSI(close, timeperiod=7)

        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close,
                                    fastk_period=14, slowk_period=3,
                                    slowk_matype=0, slowd_period=3, slowd_matype=0)
        features['stoch_k'] = slowk
        features['stoch_d'] = slowd

        # Momentum
        features['mom_10'] = talib.MOM(close, timeperiod=10)
        features['roc_10'] = talib.ROC(close, timeperiod=10)

        # CCI (Commodity Channel Index)
        features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

        # === VOLATILITY INDICATORS ===
        # ATR (Average True Range)
        features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = upper - lower
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)

        # === VOLUME INDICATORS ===
        # OBV (On Balance Volume)
        features['obv'] = talib.OBV(close, volume)

        # AD (Accumulation/Distribution)
        features['ad'] = talib.AD(high, low, close, volume)

        # === PATTERN INDICATORS ===
        # Normalized price position
        features['price_position'] = (close - talib.MIN(close, 60)) / (talib.MAX(close, 60) - talib.MIN(close, 60) + 1e-10)

        # === DERIVED FEATURES ===
        # Price relative to moving averages
        features['price_to_sma20'] = close / (features['sma_20'] + 1e-10) - 1
        features['sma5_to_sma20'] = features['sma_5'] / (features['sma_20'] + 1e-10) - 1

        # RSI zones
        features['rsi_oversold'] = (features['rsi_14'] < 30).astype(float)
        features['rsi_overbought'] = (features['rsi_14'] > 70).astype(float)

        return features

    def load_data_with_ta(self, sample_size: int = 100000) -> Tuple[pd.DataFrame, List[str]]:
        """Load price data and compute TA-Lib features."""
        print(f"Loading {sample_size} samples from database...")

        # Load data with enough history for indicators
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
        print("Computing TA-Lib indicators...")
        ta_features = self.compute_ta_features(df)

        # Combine with original data
        df = pd.concat([df, ta_features], axis=1)

        # Get list of feature columns (excluding NaN columns)
        feature_cols = [col for col in ta_features.columns
                       if df[col].notna().sum() > len(df) * 0.5]  # Keep features with >50% valid data

        print(f"Generated {len(feature_cols)} TA-Lib features")

        # Calculate forward returns for each horizon
        print("Calculating forward returns...")
        for h in self.horizons:
            df[f'ret_{h}'] = df['price_change_pips_0'].rolling(h).sum().shift(-h)

        # Remove rows with NaN in features or returns
        df = df.dropna(subset=feature_cols + [f'ret_{h}' for h in self.horizons])
        print(f"Final dataset: {len(df)} rows with {len(feature_cols)} TA features")

        return df, feature_cols

    def run_ta_tests(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Run regression tests for TA indicators."""
        results = []

        print("\nRunning TA-Lib regression tests...")
        print("-" * 60)

        # Test different feature groups
        feature_groups = {
            'Trend': [col for col in feature_cols if any(x in col for x in ['sma', 'ema', 'macd'])],
            'Momentum': [col for col in feature_cols if any(x in col for x in ['rsi', 'stoch', 'mom', 'roc', 'cci'])],
            'Volatility': [col for col in feature_cols if any(x in col for x in ['atr', 'bb_'])],
            'Volume': [col for col in feature_cols if any(x in col for x in ['obv', 'ad'])],
            'All': feature_cols
        }

        for h in self.horizons:
            print(f"\nHorizon: {h} minutes")

            # Target variable
            y = df[f'ret_{h}'].values

            horizon_results = {'horizon_min': h}

            for group_name, group_features in feature_groups.items():
                if len(group_features) == 0:
                    continue

                # Prepare features
                X = df[group_features].values
                X = sm.add_constant(X)

                try:
                    # Fit model
                    model = sm.OLS(y, X, missing='drop').fit()
                    pred = model.predict(X)
                    r2 = r2_score(y, pred)
                    mse = mean_squared_error(y, pred)

                    horizon_results[f'r2_{group_name.lower()}'] = r2
                    horizon_results[f'mse_{group_name.lower()}'] = mse
                    horizon_results[f'n_features_{group_name.lower()}'] = len(group_features)

                    if group_name == 'All':
                        print(f"  All TA features R²: {r2:.4f} ({len(group_features)} features)")
                except:
                    horizon_results[f'r2_{group_name.lower()}'] = 0.0
                    horizon_results[f'mse_{group_name.lower()}'] = np.nan
                    horizon_results[f'n_features_{group_name.lower()}'] = len(group_features)

            results.append(horizon_results)

        return pd.DataFrame(results)

    def find_best_indicators(self, df: pd.DataFrame, feature_cols: List[str], top_n: int = 10) -> pd.DataFrame:
        """Find the most predictive individual indicators."""
        print("\n" + "=" * 60)
        print("FINDING BEST INDIVIDUAL INDICATORS")
        print("=" * 60)

        indicator_scores = []

        for feature in feature_cols:
            feature_r2_scores = []

            for h in self.horizons:
                y = df[f'ret_{h}'].values
                X = df[[feature]].values

                # Skip if too many NaNs
                if pd.isna(X).sum() > len(X) * 0.5:
                    continue

                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X, missing='drop').fit()
                    pred = model.predict(X)
                    r2 = r2_score(y, pred)
                    feature_r2_scores.append(r2)
                except:
                    feature_r2_scores.append(0.0)

            if feature_r2_scores:
                indicator_scores.append({
                    'indicator': feature,
                    'avg_r2': np.mean(feature_r2_scores),
                    'max_r2': np.max(feature_r2_scores),
                    'best_horizon': self.horizons[np.argmax(feature_r2_scores)]
                })

        # Sort by average R²
        best_indicators = pd.DataFrame(indicator_scores)
        best_indicators = best_indicators.sort_values('avg_r2', ascending=False).head(top_n)

        print("\nTop 10 Most Predictive Indicators:")
        print("-" * 50)
        for _, row in best_indicators.iterrows():
            print(f"{row['indicator']:25s} | Avg R²: {row['avg_r2']:.4f} | Best at {row['best_horizon']} min")

        return best_indicators

    def generate_summary(self, results: pd.DataFrame, best_indicators: pd.DataFrame) -> str:
        """Generate summary of TA-Lib analysis."""
        summary = []
        summary.append("=" * 70)
        summary.append("TA-LIB TECHNICAL INDICATORS ANALYSIS SUMMARY")
        summary.append("=" * 70)
        summary.append("")

        # Overall performance by indicator group
        summary.append("📊 PERFORMANCE BY INDICATOR GROUP:")

        for group in ['trend', 'momentum', 'volatility', 'all']:
            col = f'r2_{group}'
            if col in results.columns:
                avg_r2 = results[col].mean()
                best_horizon = results.loc[results[col].idxmax(), 'horizon_min']
                best_r2 = results[col].max()
                summary.append(f"  • {group.upper():12s}: Avg R² = {avg_r2:.4f}, Best = {best_r2:.4f} @ {best_horizon} min")

        summary.append("")

        # Best individual indicators
        summary.append("🎯 TOP 5 INDIVIDUAL INDICATORS:")
        for i, row in best_indicators.head(5).iterrows():
            summary.append(f"  {i+1}. {row['indicator']:20s}: R² = {row['avg_r2']:.4f}")

        summary.append("")

        # Horizon analysis
        summary.append("⏱️ BEST HORIZONS FOR TA INDICATORS:")
        if 'r2_all' in results.columns:
            best_3 = results.nlargest(3, 'r2_all')
            for _, row in best_3.iterrows():
                summary.append(f"  • {row['horizon_min']} min: R² = {row['r2_all']:.4f}")

        summary.append("")
        summary.append("=" * 70)
        summary.append("💡 TRADING IMPLICATIONS:")
        summary.append("-" * 50)

        # Interpret results
        avg_r2_all = results['r2_all'].mean() if 'r2_all' in results.columns else 0

        if avg_r2_all > 0.01:
            summary.append(f"✅ TA indicators show meaningful predictive power (Avg R² = {avg_r2_all:.4f})")
        else:
            summary.append(f"⚠️ TA indicators show limited predictive power (Avg R² = {avg_r2_all:.4f})")
            summary.append("   Consider non-linear models or alternative features")

        # Compare to base features
        summary.append("")
        summary.append("📈 COMPARISON TO BASE FEATURES:")
        summary.append(f"  • Base 5D features: ~0.0007 R² (from previous test)")
        summary.append(f"  • TA-Lib indicators: {avg_r2_all:.4f} R²")

        if avg_r2_all > 0.0007:
            improvement = (avg_r2_all / 0.0007 - 1) * 100
            summary.append(f"  • TA indicators are {improvement:.0f}% better than base features")

        return "\n".join(summary)


def main():
    """Run complete TA-Lib feature analysis."""
    print("🔬 TA-LIB TECHNICAL INDICATORS ANALYSIS")
    print("=" * 70)

    # Initialize analyzer
    analyzer = TALibFeatureAnalyzer()

    # Load data and compute TA features
    df, feature_cols = analyzer.load_data_with_ta(sample_size=100000)

    # Run regression tests
    results = analyzer.run_ta_tests(df, feature_cols)

    # Find best individual indicators
    best_indicators = analyzer.find_best_indicators(df, feature_cols)

    # Display results
    print("\n📊 RESULTS TABLE:")
    print("-" * 70)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # Show key columns
    display_cols = ['horizon_min']
    for col in results.columns:
        if col.startswith('r2_'):
            display_cols.append(col)

    print(results[display_cols])

    # Generate and print summary
    summary = analyzer.generate_summary(results, best_indicators)
    print("\n" + summary)

    # Save results
    results.to_csv('talib_analysis_results.csv', index=False)
    best_indicators.to_csv('talib_best_indicators.csv', index=False)

    with open('/workspace/talib_analysis_summary.txt', 'w') as f:
        f.write(summary)

    print("\n✅ Results saved to:")
    print("  • talib_analysis_results.csv")
    print("  • talib_best_indicators.csv")
    print("  • talib_analysis_summary.txt")

    return results, best_indicators


if __name__ == "__main__":
    results, best = main()