#!/usr/bin/env python3
"""
Quick Timeframe Comparison - Samples from dataset for efficiency
Tests multiple timeframe combinations on representative data samples
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QuickTFResult:
    tf_pair: str
    sample_size: int
    trades: int
    win_rate: float
    total_pips: float
    pips_per_trade: float
    profit_factor: float
    trade_frequency: float  # trades per 1000 bars


class QuickTFAnalyzer:
    """Quick timeframe comparison using data samples"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.pip_multiplier = 100

    def get_sample_ranges(self, n_samples: int = 5, sample_size: int = 10000):
        """Get evenly distributed sample ranges from dataset"""
        conn = duckdb.connect(self.db_path, read_only=True)

        # Get total range
        query = "SELECT MIN(bar_index), MAX(bar_index) FROM master"
        min_idx, max_idx = conn.execute(query).fetchone()
        conn.close()

        total_range = max_idx - min_idx
        step = total_range // (n_samples + 1)

        samples = []
        for i in range(1, n_samples + 1):
            start = min_idx + (i * step)
            end = start + sample_size
            if end <= max_idx:
                samples.append((start, end))

        print(f"Created {len(samples)} samples of {sample_size} bars each")
        print(f"Total dataset: {min_idx:,} to {max_idx:,} ({total_range:,} bars)")

        return samples

    def quick_test_tf(self, exec_tf: int, context_tf: int,
                     start_idx: int, end_idx: int) -> Dict:
        """Quick test of a timeframe pair on a data sample"""

        conn = duckdb.connect(self.db_path, read_only=True)

        # Load sample
        query = f"""
        SELECT bar_index, close, high, low
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {end_idx}
        ORDER BY bar_index
        """

        data = pd.read_sql(query, conn)
        conn.close()

        if len(data) < context_tf * 4:
            return {'trades': 0, 'pips': 0}

        # Simple aggregation
        exec_data = self.aggregate_simple(data, exec_tf)
        context_data = self.aggregate_simple(data, context_tf)

        # Simple trend detection
        exec_trend = self.detect_trend(exec_data)
        context_trend = self.detect_trend(context_data)

        # Simulate trades
        trades = []
        position = None

        for i in range(10, len(exec_data) - 1):
            # Map to context
            exec_bar = exec_data.iloc[i]['bar_index']
            ctx_idx = len(context_data[context_data['bar_index'] <= exec_bar]) - 1

            if ctx_idx < 0 or ctx_idx >= len(context_trend):
                continue

            exec_t = exec_trend[i] if i < len(exec_trend) else 0
            context_t = context_trend[ctx_idx]

            # Entry logic
            if position is None:
                # Bullish alignment
                if context_t > 0 and exec_t > 0:
                    position = {
                        'type': 'long',
                        'entry': exec_data.iloc[i]['close'],
                        'entry_idx': i
                    }
                # Bearish alignment
                elif context_t < 0 and exec_t < 0:
                    position = {
                        'type': 'short',
                        'entry': exec_data.iloc[i]['close'],
                        'entry_idx': i
                    }

            # Exit logic (simple: opposite signal or 20 bars)
            elif position:
                bars_held = i - position['entry_idx']
                exit_signal = False

                if position['type'] == 'long':
                    if exec_t < 0 or bars_held > 20:
                        pips = (exec_data.iloc[i]['close'] - position['entry']) * self.pip_multiplier
                        exit_signal = True
                else:  # short
                    if exec_t > 0 or bars_held > 20:
                        pips = (position['entry'] - exec_data.iloc[i]['close']) * self.pip_multiplier
                        exit_signal = True

                if exit_signal:
                    trades.append({'pips': pips, 'bars': bars_held})
                    position = None

        return {
            'trades': len(trades),
            'pips': sum(t['pips'] for t in trades),
            'wins': len([t for t in trades if t['pips'] > 0])
        }

    def aggregate_simple(self, data: pd.DataFrame, period: int):
        """Simple aggregation without full OHLC"""
        if period == 1:
            return data

        data['group'] = data['bar_index'] // period
        return data.groupby('group').agg({
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'bar_index': 'first'
        }).reset_index(drop=True)

    def detect_trend(self, data: pd.DataFrame) -> List[int]:
        """Simple trend detection: 1=up, 0=neutral, -1=down"""
        if len(data) < 20:
            return [0] * len(data)

        # Use simple moving average crossover
        sma_fast = data['close'].rolling(5, min_periods=1).mean()
        sma_slow = data['close'].rolling(20, min_periods=1).mean()

        trend = []
        for i in range(len(data)):
            if pd.isna(sma_slow.iloc[i]):
                trend.append(0)
            elif sma_fast.iloc[i] > sma_slow.iloc[i]:
                trend.append(1)
            elif sma_fast.iloc[i] < sma_slow.iloc[i]:
                trend.append(-1)
            else:
                trend.append(0)

        return trend

    def run_comparison(self):
        """Run quick comparison across all timeframes"""

        # Get sample ranges
        samples = self.get_sample_ranges(n_samples=3, sample_size=10000)

        # Timeframe combinations
        tf_combinations = [
            (1, 60, 'M1/H1'),
            (1, 240, 'M1/H4'),
            (5, 60, 'M5/H1'),
            (5, 240, 'M5/H4'),
            (15, 60, 'M15/H1'),
            (15, 240, 'M15/H4'),
            (30, 240, 'M30/H4'),
            (60, 240, 'H1/H4'),
        ]

        print("\n" + "="*70)
        print("QUICK TIMEFRAME COMPARISON (Sampled Data)")
        print("="*70)

        all_results = []

        for exec_tf, context_tf, name in tf_combinations:
            print(f"\nTesting {name}...")

            total_trades = 0
            total_pips = 0
            total_wins = 0

            for sample_idx, (start, end) in enumerate(samples):
                result = self.quick_test_tf(exec_tf, context_tf, start, end)
                total_trades += result['trades']
                total_pips += result['pips']
                total_wins += result['wins']

                print(f"  Sample {sample_idx+1}: {result['trades']} trades, {result['pips']:.1f} pips")

            # Calculate statistics
            if total_trades > 0:
                win_rate = (total_wins / total_trades) * 100
                pips_per_trade = total_pips / total_trades
                profit_factor = abs(total_pips / (total_trades - total_wins)) if total_wins < total_trades else 999

                # Trade frequency per 1000 execution bars
                total_exec_bars = len(samples) * 10000 / exec_tf
                trade_freq = (total_trades / total_exec_bars) * 1000
            else:
                win_rate = 0
                pips_per_trade = 0
                profit_factor = 0
                trade_freq = 0

            result = QuickTFResult(
                tf_pair=name,
                sample_size=len(samples) * 10000,
                trades=total_trades,
                win_rate=win_rate,
                total_pips=total_pips,
                pips_per_trade=pips_per_trade,
                profit_factor=profit_factor,
                trade_frequency=trade_freq
            )

            all_results.append(result)

        # Display results table
        self.display_results(all_results)

        return all_results

    def display_results(self, results: List[QuickTFResult]):
        """Display comparison results"""

        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)

        # Sort by total pips
        results.sort(key=lambda x: x.total_pips, reverse=True)

        # Header
        print(f"{'Rank':<5} {'TF Pair':<10} {'Trades':<8} {'Win%':<8} "
              f"{'Total Pips':<12} {'Pips/Trade':<12} {'Freq/1000':<10}")
        print("-"*70)

        for i, r in enumerate(results, 1):
            rank_symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

            print(f"{rank_symbol:<5} {r.tf_pair:<10} {r.trades:<8} "
                  f"{r.win_rate:<7.1f}% {r.total_pips:<11.1f} "
                  f"{r.pips_per_trade:<11.2f} {r.trade_frequency:<9.1f}")

        print("-"*70)

        # Analysis
        best = results[0]
        worst = results[-1]

        print(f"\n📊 KEY FINDINGS:")
        print(f"✅ Best Performer: {best.tf_pair}")
        print(f"   • {best.total_pips:.1f} total pips")
        print(f"   • {best.pips_per_trade:.2f} pips per trade")
        print(f"   • {best.win_rate:.1f}% win rate")

        print(f"\n❌ Worst Performer: {worst.tf_pair}")
        print(f"   • {worst.total_pips:.1f} total pips")
        print(f"   • {worst.trades} trades (may be too few)")

        # Insights
        print(f"\n💡 INSIGHTS:")

        # Check if lower TFs are better
        low_tf = [r for r in results if 'M1/' in r.tf_pair or 'M5/' in r.tf_pair]
        high_tf = [r for r in results if 'M30/' in r.tf_pair or 'H1/' in r.tf_pair]

        avg_low = np.mean([r.total_pips for r in low_tf])
        avg_high = np.mean([r.total_pips for r in high_tf])

        if avg_low > avg_high:
            print("• Lower timeframes (M1/M5) perform better than higher (M30/H1)")
        else:
            print("• Higher timeframes (M30/H1) perform better than lower (M1/M5)")

        # Check H1 vs H4 context
        h1_context = [r for r in results if '/H1' in r.tf_pair]
        h4_context = [r for r in results if '/H4' in r.tf_pair]

        avg_h1 = np.mean([r.total_pips for r in h1_context])
        avg_h4 = np.mean([r.total_pips for r in h4_context])

        if avg_h1 > avg_h4:
            print("• H1 context timeframe works better than H4")
        else:
            print("• H4 context timeframe works better than H1")

        # Trade frequency insight
        high_freq = [r for r in results if r.trade_frequency > 5]
        if high_freq:
            print(f"• High frequency setups: {', '.join([r.tf_pair for r in high_freq])}")

        print("\n" + "="*70)


def main():
    """Run quick timeframe comparison"""

    print("Quick Timeframe Comparison Analysis")
    print("Using sampled data for efficiency")
    print("-"*40)

    analyzer = QuickTFAnalyzer()
    results = analyzer.run_comparison()

    # Save results
    print("\nSaving results...")
    df = pd.DataFrame([{
        'timeframe': r.tf_pair,
        'trades': r.trades,
        'win_rate': r.win_rate,
        'total_pips': r.total_pips,
        'pips_per_trade': r.pips_per_trade,
        'trade_frequency': r.trade_frequency
    } for r in results])

    df.to_csv('quick_tf_results.csv', index=False)
    print("Results saved to quick_tf_results.csv")


if __name__ == "__main__":
    main()