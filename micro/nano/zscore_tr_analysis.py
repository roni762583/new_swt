#!/usr/bin/env python3
"""
Z-Score of True Range Predictive Analysis
Calculates: zscore_20_500 = (TR - SMA(TR, 20)) / rolling_std(TR, 500)
Tests predictive power for various return horizons
"""

import numpy as np
import pandas as pd
import duckdb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class ZScoreTRAnalysis:
    def __init__(self, db_path="/data/master.duckdb"):
        self.conn = duckdb.connect(db_path, read_only=True)

    def calculate_zscore_tr(self, limit=100000):
        """Calculate z-score of True Range."""
        print("📊 Loading OHLCV data from master table...")

        # Load data
        query = f"""
        SELECT
            rowid,
            open,
            high,
            low,
            close,
            volume
        FROM master
        ORDER BY rowid
        LIMIT {limit}
        """

        df = self.conn.execute(query).df()
        print(f"Loaded {len(df)} rows")

        # Calculate True Range
        print("\n📈 Calculating True Range...")
        df['prev_close'] = df['close'].shift(1)

        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = np.abs(df['high'] - df['prev_close'])
        df['tr3'] = np.abs(df['low'] - df['prev_close'])

        # True Range is the maximum of the three
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate SMA(20) of TR
        print("📊 Calculating SMA(20) of True Range...")
        df['tr_sma20'] = df['true_range'].rolling(window=20, min_periods=1).mean()

        # Calculate rolling std(500) of TR
        print("📊 Calculating rolling std(500) of True Range...")
        df['tr_std500'] = df['true_range'].rolling(window=500, min_periods=50).std()

        # Calculate Z-score
        print("📊 Calculating Z-score (TR - SMA20) / std500...")
        df['zscore_20_500'] = (df['true_range'] - df['tr_sma20']) / df['tr_std500']

        # Handle edge cases
        df['zscore_20_500'] = df['zscore_20_500'].replace([np.inf, -np.inf], np.nan)

        # Stats
        print("\n📊 Z-Score Statistics:")
        print(f"  Mean: {df['zscore_20_500'].mean():.4f}")
        print(f"  Std: {df['zscore_20_500'].std():.4f}")
        print(f"  Min: {df['zscore_20_500'].min():.4f}")
        print(f"  Max: {df['zscore_20_500'].max():.4f}")
        print(f"  NaN count: {df['zscore_20_500'].isna().sum()} (first 500 rows expected)")

        # Add close price for return calculations
        df['close_price'] = df['close']

        return df

    def test_predictive_power(self, df, horizons=[1, 2, 3, 4, 5, 15, 30]):
        """Test predictive power of zscore_20_500 for various horizons."""

        print("\n🔬 Testing Predictive Power of zscore_20_500")
        print("=" * 60)

        results = []

        for horizon in horizons:
            # Calculate future returns
            df[f'return_{horizon}'] = df['close_price'].shift(-horizon) / df['close_price'] - 1

            # Remove NaN values
            valid_mask = ~(df['zscore_20_500'].isna() | df[f'return_{horizon}'].isna())
            X = df.loc[valid_mask, ['zscore_20_500']].values
            y = df.loc[valid_mask, f'return_{horizon}'].values

            if len(X) < 100:
                print(f"Insufficient data for {horizon}-min horizon")
                continue

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            # Calculate correlation
            correlation = np.corrcoef(X.flatten(), y)[0, 1]

            results.append({
                'horizon': horizon,
                'r2': r2,
                'correlation': correlation,
                'coef': model.coef_[0],
                'samples': len(X)
            })

            print(f"\n{horizon}-min ahead:")
            print(f"  R²: {r2:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            print(f"  Coefficient: {model.coef_[0]:.8f}")
            print(f"  Samples: {len(X):,}")

        return pd.DataFrame(results)

    def analyze_extreme_values(self, df):
        """Analyze returns during extreme z-score values."""
        print("\n🎯 Analyzing Extreme Z-Score Events")
        print("=" * 60)

        # Define thresholds
        thresholds = [2, -2, 3, -3]

        for thresh in thresholds:
            if thresh > 0:
                mask = df['zscore_20_500'] > thresh
                label = f"Z-Score > {thresh}"
            else:
                mask = df['zscore_20_500'] < thresh
                label = f"Z-Score < {thresh}"

            if mask.sum() == 0:
                print(f"\n{label}: No occurrences")
                continue

            print(f"\n{label}: {mask.sum()} occurrences ({mask.sum()/len(df)*100:.2f}%)")

            # Calculate average returns for different horizons
            for horizon in [1, 5, 15, 30]:
                ret_col = f'return_{horizon}'
                if ret_col in df.columns:
                    avg_return = df.loc[mask, ret_col].mean() * 10000  # Convert to pips
                    std_return = df.loc[mask, ret_col].std() * 10000
                    print(f"  {horizon:2d}-min return: {avg_return:+.1f} ± {std_return:.1f} pips")

    def run_complete_analysis(self):
        """Run complete z-score analysis."""
        print("🔬 Z-SCORE TRUE RANGE ANALYSIS")
        print("=" * 70)

        # Calculate z-score
        df = self.calculate_zscore_tr(limit=100000)

        # Test predictive power
        results_df = self.test_predictive_power(df)

        # Analyze extreme values
        self.analyze_extreme_values(df)

        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)

        if not results_df.empty:
            best_horizon = results_df.loc[results_df['r2'].idxmax()]
            print(f"Best predictive horizon: {best_horizon['horizon']}-min")
            print(f"Best R²: {best_horizon['r2']:.6f}")
            print(f"Best correlation: {best_horizon['correlation']:.6f}")

            print("\n📈 R² by Horizon:")
            for _, row in results_df.iterrows():
                bar_length = int(row['r2'] * 100000)  # Scale for visibility
                bar = '█' * bar_length if bar_length > 0 else '|'
                print(f"  {int(row['horizon']):2d} min: {bar} {row['r2']:.6f}")

        return df, results_df

def main():
    analyzer = ZScoreTRAnalysis()
    df, results = analyzer.run_complete_analysis()

    # Save results
    if not results.empty:
        results.to_csv('zscore_tr_results.csv', index=False)
        print("\n📁 Results saved to zscore_tr_results.csv")

    return df, results

if __name__ == "__main__":
    main()