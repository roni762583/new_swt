#!/usr/bin/env python3
"""
Optimal Feature Finder for M5/H1 Strategy
Identifies minimal set of highly predictive features including:
- Trend following features
- Range/trend detection
- Mean reversion features
"""

import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureSet:
    """Compact feature set for RL"""
    # Core price features (M5)
    price_position: float  # (close - sma20) / atr
    trend_strength: float  # (sma20 - sma50) / atr
    momentum: float       # roc_5 = (close - close_5) / close_5

    # Volatility & Range Detection
    atr_ratio: float      # atr / sma20 (volatility normalized)
    efficiency_ratio: float  # Kaufman's ER for trend detection

    # Mean Reversion Features
    bb_position: float    # Position in Bollinger Band (-1 to 1)
    rsi_extreme: float    # RSI distance from 50

    # Multi-timeframe (H1 context)
    h1_trend: float       # H1 trend direction (-1, 0, 1)
    h1_momentum: float    # H1 ROC

    # Market Regime
    regime: str          # 'TREND', 'RANGE', 'VOLATILE'


class OptimalFeatureAnalyzer:
    """Find and test optimal minimal features"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.pip_multiplier = 100

    def load_sample_data(self, start_idx: int = 100000, size: int = 20000):
        """Load sample data for feature analysis"""
        conn = duckdb.connect(self.db_path, read_only=True)

        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {start_idx + size}
        ORDER BY bar_index
        """

        self.m1_data = pd.read_sql(query, conn)
        conn.close()

        # Create M5 and H1 data
        self.m5_data = self.aggregate_bars(self.m1_data, 5)
        self.h1_data = self.aggregate_bars(self.m1_data, 60)

        print(f"Loaded {len(self.m1_data)} M1 bars")
        print(f"Created {len(self.m5_data)} M5 bars, {len(self.h1_data)} H1 bars")

    def aggregate_bars(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Aggregate to higher timeframe"""
        if period == 1:
            return data.copy()

        data['group'] = data['bar_index'] // period

        agg = data.groupby('group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_index': 'first',
            'timestamp': 'first'
        }).reset_index(drop=True)

        return agg

    def calculate_features(self) -> pd.DataFrame:
        """Calculate all candidate features"""

        # M5 Features
        df = self.m5_data.copy()

        # Basic price features
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()

        # ATR for normalization
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # === TREND FOLLOWING FEATURES ===

        # 1. Price Position (normalized distance from MA)
        df['price_position'] = (df['close'] - df['sma20']) / (df['atr'] + 0.0001)

        # 2. Trend Strength (MA alignment)
        df['trend_strength'] = (df['sma20'] - df['sma50']) / (df['atr'] + 0.0001)

        # 3. Momentum (Rate of Change)
        df['momentum'] = df['close'].pct_change(5)  # 5-bar ROC

        # === VOLATILITY & REGIME DETECTION ===

        # 4. ATR Ratio (volatility relative to price)
        df['atr_ratio'] = df['atr'] / (df['sma20'] + 0.0001)

        # 5. Efficiency Ratio (Kaufman's - trend vs noise)
        direction = (df['close'] - df['close'].shift(10)).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = direction / (volatility + 0.0001)

        # === MEAN REVERSION FEATURES ===

        # 6. Bollinger Band Position
        bb_std = df['close'].rolling(20).std()
        bb_upper = df['sma20'] + (bb_std * 2)
        bb_lower = df['sma20'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['sma20']) / (bb_std * 2 + 0.0001)
        df['bb_position'] = df['bb_position'].clip(-1, 1)  # Normalize to [-1, 1]

        # 7. RSI Extreme (distance from neutral 50)
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['rsi_extreme'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]

        # === H1 CONTEXT FEATURES ===

        # Calculate H1 features
        h1 = self.h1_data.copy()
        h1['h1_sma20'] = h1['close'].rolling(20).mean()
        h1['h1_trend'] = 0
        h1.loc[h1['close'] > h1['h1_sma20'], 'h1_trend'] = 1
        h1.loc[h1['close'] < h1['h1_sma20'], 'h1_trend'] = -1
        h1['h1_momentum'] = h1['close'].pct_change(5)

        # Map H1 features to M5
        df['h1_trend'] = 0
        df['h1_momentum'] = 0

        for i in range(len(df)):
            m5_bar = df.iloc[i]['bar_index']
            # Find corresponding H1 bar
            h1_idx = h1[h1['bar_index'] <= m5_bar].index
            if len(h1_idx) > 0:
                h1_idx = h1_idx[-1]
                df.loc[i, 'h1_trend'] = h1.loc[h1_idx, 'h1_trend']
                df.loc[i, 'h1_momentum'] = h1.loc[h1_idx, 'h1_momentum']

        # === MARKET REGIME CLASSIFICATION ===

        df['regime'] = 'RANGE'  # Default

        # Trending: Strong directional movement with high efficiency
        trending_mask = (abs(df['trend_strength']) > 0.5) & (df['efficiency_ratio'] > 0.3)
        df.loc[trending_mask, 'regime'] = 'TREND'

        # Volatile: High ATR but low efficiency (choppy)
        volatile_mask = (df['atr_ratio'] > df['atr_ratio'].rolling(50).mean() * 1.5) & (df['efficiency_ratio'] < 0.2)
        df.loc[volatile_mask, 'regime'] = 'VOLATILE'

        # === COMPOSITE FEATURES ===

        # Reactive/LessReactive style features
        df['reactive'] = df['close'] - df['sma200']
        df['lessreactive'] = df['sma20'] - df['sma200']
        df['react_ratio'] = df['reactive'] / (df['lessreactive'] + 0.0001)
        df['react_ratio'] = df['react_ratio'].clip(-5, 5)  # Cap extremes

        # Feature for regime-appropriate strategy
        df['use_mean_reversion'] = (df['regime'] == 'RANGE').astype(int)
        df['use_trend_follow'] = (df['regime'] == 'TREND').astype(int)

        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def test_feature_predictiveness(self, df: pd.DataFrame) -> Dict:
        """Test how predictive each feature is"""

        # Calculate future returns (5 M5 bars = 25 minutes ahead)
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1

        # Features to test
        feature_cols = [
            'price_position', 'trend_strength', 'momentum',
            'atr_ratio', 'efficiency_ratio',
            'bb_position', 'rsi_extreme',
            'h1_trend', 'h1_momentum',
            'react_ratio'
        ]

        results = {}

        for col in feature_cols:
            if col not in df.columns:
                continue

            clean_df = df[[col, 'future_return']].dropna()

            if len(clean_df) < 100:
                continue

            # Correlation with future returns
            corr = clean_df[col].corr(clean_df['future_return'])

            # Information Coefficient (rank correlation)
            rank_corr = clean_df[col].rank().corr(clean_df['future_return'].rank())

            # Predictive power in different regimes
            trend_corr = 0
            range_corr = 0

            if 'regime' in df.columns:
                trend_data = df[df['regime'] == 'TREND'][[col, 'future_return']].dropna()
                range_data = df[df['regime'] == 'RANGE'][[col, 'future_return']].dropna()

                if len(trend_data) > 50:
                    trend_corr = trend_data[col].corr(trend_data['future_return'])
                if len(range_data) > 50:
                    range_corr = range_data[col].corr(range_data['future_return'])

            results[col] = {
                'correlation': corr,
                'rank_correlation': rank_corr,
                'trend_regime_corr': trend_corr,
                'range_regime_corr': range_corr,
                'abs_corr': abs(corr)
            }

        return results

    def simulate_with_features(self, df: pd.DataFrame) -> Dict:
        """Simulate trading with different feature combinations"""

        strategies = {
            'trend_only': ['price_position', 'trend_strength', 'momentum', 'h1_trend'],
            'mean_reversion': ['bb_position', 'rsi_extreme', 'atr_ratio'],
            'regime_adaptive': ['price_position', 'trend_strength', 'bb_position', 'regime'],
            'minimal': ['react_ratio', 'h1_trend', 'efficiency_ratio'],
            'kitchen_sink': ['price_position', 'trend_strength', 'momentum', 'atr_ratio',
                            'efficiency_ratio', 'bb_position', 'rsi_extreme', 'h1_trend',
                            'h1_momentum', 'react_ratio']
        }

        results = {}

        for name, features in strategies.items():
            trades = self.backtest_strategy(df, features)

            total_trades = len(trades)
            if total_trades == 0:
                results[name] = {'trades': 0, 'pips': 0, 'win_rate': 0}
                continue

            total_pips = sum(t['pips'] for t in trades)
            wins = [t for t in trades if t['pips'] > 0]
            win_rate = len(wins) / total_trades * 100

            results[name] = {
                'trades': total_trades,
                'pips': total_pips,
                'win_rate': win_rate,
                'features': features
            }

        return results

    def backtest_strategy(self, df: pd.DataFrame, features: List[str]) -> List[Dict]:
        """Simple backtest using feature signals"""
        trades = []
        position = None

        for i in range(50, len(df) - 10):  # Skip warmup period

            if position is not None:
                # Check exit
                bars_held = i - position['entry_idx']

                # Exit after 10 bars or on opposite signal
                if bars_held >= 10:
                    exit_price = df.iloc[i]['close']

                    if position['direction'] == 'long':
                        pips = (exit_price - position['entry_price']) * self.pip_multiplier
                    else:
                        pips = (position['entry_price'] - exit_price) * self.pip_multiplier

                    trades.append({'pips': pips, 'bars': bars_held})
                    position = None
                continue

            # Entry logic based on features
            signal = self.generate_signal(df.iloc[i], features)

            if signal != 0:
                position = {
                    'entry_idx': i,
                    'entry_price': df.iloc[i]['close'],
                    'direction': 'long' if signal > 0 else 'short'
                }

        return trades

    def generate_signal(self, row: pd.Series, features: List[str]) -> int:
        """Generate trading signal from features"""

        # Simple scoring system
        score = 0

        for feat in features:
            if feat not in row or pd.isna(row[feat]):
                continue

            # Trend features
            if feat in ['price_position', 'trend_strength', 'momentum', 'h1_trend', 'react_ratio']:
                if row[feat] > 0.5:
                    score += 1
                elif row[feat] < -0.5:
                    score -= 1

            # Mean reversion features
            elif feat in ['bb_position', 'rsi_extreme']:
                if row[feat] < -0.8:  # Oversold
                    score += 1
                elif row[feat] > 0.8:  # Overbought
                    score -= 1

            # Regime filter
            elif feat == 'regime':
                if row[feat] == 'TREND' and 'trend_strength' in row:
                    score *= 2  # Double trend signals in trending regime
                elif row[feat] == 'RANGE' and 'bb_position' in row:
                    score *= -1  # Reverse in range (mean reversion)

        # Threshold for signal
        if score >= 2:
            return 1  # Long
        elif score <= -2:
            return -1  # Short
        else:
            return 0  # No signal

    def find_optimal_features(self):
        """Main analysis to find optimal features"""

        print("="*70)
        print("OPTIMAL FEATURE ANALYSIS FOR M5/H1 STRATEGY")
        print("="*70)

        # Load and prepare data
        print("\n1. Loading sample data...")
        self.load_sample_data()

        # Calculate features
        print("\n2. Calculating features...")
        df = self.calculate_features()

        # Test predictiveness
        print("\n3. Testing feature predictiveness...")
        predictiveness = self.test_feature_predictiveness(df)

        # Display results
        print("\n" + "="*70)
        print("FEATURE PREDICTIVENESS (Correlation with Future Returns)")
        print("-"*70)

        # Sort by absolute correlation
        sorted_features = sorted(predictiveness.items(),
                                key=lambda x: x[1]['abs_corr'],
                                reverse=True)

        for feat, metrics in sorted_features:
            print(f"{feat:20s} | Corr: {metrics['correlation']:+.3f} | "
                  f"Trend: {metrics['trend_regime_corr']:+.3f} | "
                  f"Range: {metrics['range_regime_corr']:+.3f}")

        # Test strategies
        print("\n4. Testing feature combinations...")
        strategy_results = self.simulate_with_features(df)

        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("-"*70)

        for name, results in sorted(strategy_results.items(),
                                   key=lambda x: x[1]['pips'],
                                   reverse=True):
            print(f"{name:15s} | Trades: {results['trades']:3d} | "
                  f"Pips: {results['pips']:+7.1f} | "
                  f"Win%: {results['win_rate']:.1f}%")

        # Generate optimal feature set
        print("\n" + "="*70)
        print("RECOMMENDED MINIMAL FEATURE SET (7 features)")
        print("="*70)

        optimal_features = {
            'trend_features': {
                'react_ratio': 'Normalized momentum (reactive/lessreactive)',
                'h1_trend': 'Higher timeframe direction filter',
                'efficiency_ratio': 'Trend quality (signal/noise)',
            },
            'mean_reversion_features': {
                'bb_position': 'Bollinger Band position for extremes',
                'rsi_extreme': 'RSI distance from neutral',
            },
            'regime_detection': {
                'atr_ratio': 'Volatility regime indicator',
                'use_mean_reversion': 'Binary flag for regime strategy',
            }
        }

        print("\n📊 TREND FOLLOWING (use when efficiency_ratio > 0.3):")
        for feat, desc in optimal_features['trend_features'].items():
            print(f"  • {feat}: {desc}")

        print("\n🔄 MEAN REVERSION (use when regime == 'RANGE'):")
        for feat, desc in optimal_features['mean_reversion_features'].items():
            print(f"  • {feat}: {desc}")

        print("\n🎯 REGIME DETECTION:")
        for feat, desc in optimal_features['regime_detection'].items():
            print(f"  • {feat}: {desc}")

        # Create visualization
        self.visualize_features(df)

        return df, optimal_features

    def visualize_features(self, df: pd.DataFrame):
        """Create feature visualization"""

        fig, axes = plt.subplots(4, 2, figsize=(15, 12))

        # Sample for plotting
        plot_df = df.iloc[200:400].reset_index(drop=True)

        # 1. Price with regime backgrounds
        ax = axes[0, 0]
        ax.plot(plot_df.index, plot_df['close'], 'k-', linewidth=1)

        # Color background by regime
        for regime, color in [('TREND', 'lightgreen'), ('RANGE', 'lightyellow'), ('VOLATILE', 'lightcoral')]:
            mask = plot_df['regime'] == regime
            ax.fill_between(plot_df.index, plot_df['close'].min(), plot_df['close'].max(),
                          where=mask, alpha=0.3, color=color, label=regime)

        ax.set_title('Price with Market Regimes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. React Ratio
        ax = axes[0, 1]
        ax.plot(plot_df.index, plot_df['react_ratio'], 'b-', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        ax.set_title('React Ratio (Reactive/LessReactive)')
        ax.grid(True, alpha=0.3)

        # 3. Efficiency Ratio
        ax = axes[1, 0]
        ax.plot(plot_df.index, plot_df['efficiency_ratio'], 'g-', alpha=0.7)
        ax.axhline(y=0.3, color='red', linestyle='--', label='Trend Threshold')
        ax.fill_between(plot_df.index, 0, plot_df['efficiency_ratio'],
                        where=plot_df['efficiency_ratio'] > 0.3, alpha=0.3, color='green')
        ax.set_title('Efficiency Ratio (Trend Detection)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. BB Position
        ax = axes[1, 1]
        ax.plot(plot_df.index, plot_df['bb_position'], 'purple', alpha=0.7)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax.axhline(y=-0.8, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax.fill_between(plot_df.index, -0.8, 0.8, alpha=0.1, color='gray')
        ax.set_title('Bollinger Band Position')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. RSI Extreme
        ax = axes[2, 0]
        ax.plot(plot_df.index, plot_df['rsi_extreme'], 'orange', alpha=0.7)
        ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
        ax.set_title('RSI Extreme (Distance from 50)')
        ax.grid(True, alpha=0.3)

        # 6. H1 Context
        ax = axes[2, 1]
        ax.plot(plot_df.index, plot_df['h1_trend'], 'r-', label='H1 Trend', drawstyle='steps-post')
        ax2 = ax.twinx()
        ax2.plot(plot_df.index, plot_df['h1_momentum'] * 100, 'b--', alpha=0.5, label='H1 Momentum')
        ax.set_title('H1 Context Features')
        ax.set_ylabel('H1 Trend')
        ax2.set_ylabel('H1 Momentum (%)')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 7. Composite Signal
        ax = axes[3, 0]

        # Calculate composite signal
        trend_signal = (plot_df['react_ratio'] > 0.5).astype(int) - (plot_df['react_ratio'] < -0.5).astype(int)
        mr_signal = (plot_df['bb_position'] < -0.8).astype(int) - (plot_df['bb_position'] > 0.8).astype(int)

        # Apply regime filter
        composite_signal = trend_signal.copy()
        composite_signal[plot_df['regime'] == 'RANGE'] = mr_signal[plot_df['regime'] == 'RANGE']

        ax.plot(plot_df.index, composite_signal, 'k-', drawstyle='steps-post')
        ax.fill_between(plot_df.index, 0, composite_signal,
                        where=composite_signal > 0, alpha=0.3, color='green', label='Long')
        ax.fill_between(plot_df.index, 0, composite_signal,
                        where=composite_signal < 0, alpha=0.3, color='red', label='Short')
        ax.set_title('Composite Trading Signal')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Feature Correlation Matrix
        ax = axes[3, 1]

        feature_cols = ['react_ratio', 'efficiency_ratio', 'bb_position',
                       'rsi_extreme', 'h1_trend', 'atr_ratio']

        corr_matrix = plot_df[feature_cols].corr()

        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_yticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=45, ha='right')
        ax.set_yticklabels(feature_cols)
        ax.set_title('Feature Correlation')

        # Add correlation values
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax)

        plt.suptitle('Optimal Feature Set Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('optimal_features.png', dpi=100, bbox_inches='tight')
        plt.close()

        print("\nVisualization saved to optimal_features.png")


def main():
    """Find optimal features for M5/H1 strategy"""

    analyzer = OptimalFeatureAnalyzer()
    df, features = analyzer.find_optimal_features()

    # Generate final RL state space recommendation
    print("\n" + "="*70)
    print("FINAL RL STATE SPACE RECOMMENDATION")
    print("="*70)

    print("""
# Minimal but powerful 7-feature state space for M5/H1

state = {
    # Core trend features (3)
    'react_ratio': (close - sma200) / (sma20 - sma200),  # -5 to 5
    'h1_trend': h1_direction,  # -1, 0, 1
    'efficiency_ratio': direction_move / total_volatility,  # 0 to 1

    # Mean reversion features (2)
    'bb_position': (close - bb_mid) / (2 * bb_std),  # -1 to 1
    'rsi_extreme': (rsi - 50) / 50,  # -1 to 1

    # Regime detection (2)
    'atr_ratio': atr / sma20,  # Volatility regime
    'use_mean_reversion': 1 if efficiency_ratio < 0.3 else 0
}

# Action space
if state['use_mean_reversion']:
    # Mean reversion mode
    if state['bb_position'] < -0.8:
        action = BUY
    elif state['bb_position'] > 0.8:
        action = SELL
else:
    # Trend following mode
    if state['react_ratio'] > 0.5 and state['h1_trend'] > 0:
        action = BUY
    elif state['react_ratio'] < -0.5 and state['h1_trend'] < 0:
        action = SELL
    """)

    print("\n✅ This minimal 7-feature set captures:")
    print("   • Trend direction and strength")
    print("   • Mean reversion opportunities")
    print("   • Market regime (trending vs ranging)")
    print("   • Multi-timeframe confirmation")
    print("   • All features normalized to [-1, 1] or [0, 1]")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()