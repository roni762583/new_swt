#!/usr/bin/env python3
"""
Test simple trading strategies with potential for better metrics.
"""

import numpy as np
import pandas as pd
import duckdb

def momentum_reversal_strategy(df, lookback=20, reversal_threshold=2.0, pip_size=0.01, spread_pips=4):
    """
    Trade momentum reversals after extreme moves.
    Entry: When price reverses after moving >2 std devs from mean
    Exit: Return to mean or fixed stop/target
    """
    closes = df['close'].values
    trades = []

    # Calculate rolling mean and std
    for i in range(lookback, len(closes) - 100):
        window = closes[i-lookback:i]
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            continue

        z_score = (closes[i] - mean) / std

        # Look for extreme moves
        if abs(z_score) > reversal_threshold:
            # Wait for reversal signal
            if z_score > reversal_threshold:  # Overbought - look for short
                # Check if price starts moving down
                if closes[i+1] < closes[i]:
                    entry = closes[i+1]
                    stop = closes[i] + (2 * std)  # Stop above recent high
                    target = mean  # Target the mean

                    # Simulate trade
                    for j in range(i+2, min(i+100, len(closes))):
                        if closes[j] >= stop:
                            exit_price = stop
                            break
                        elif closes[j] <= target:
                            exit_price = target
                            break
                    else:
                        exit_price = closes[min(i+100, len(closes)-1)]

                    pnl = (entry - exit_price) / pip_size - spread_pips
                    trades.append({
                        'type': 'momentum_reversal',
                        'direction': 'short',
                        'entry_idx': i+1,
                        'pnl_pips': pnl,
                        'z_score': z_score
                    })

            elif z_score < -reversal_threshold:  # Oversold - look for long
                if closes[i+1] > closes[i]:
                    entry = closes[i+1]
                    stop = closes[i] - (2 * std)
                    target = mean

                    for j in range(i+2, min(i+100, len(closes))):
                        if closes[j] <= stop:
                            exit_price = stop
                            break
                        elif closes[j] >= target:
                            exit_price = target
                            break
                    else:
                        exit_price = closes[min(i+100, len(closes)-1)]

                    pnl = (exit_price - entry) / pip_size - spread_pips
                    trades.append({
                        'type': 'momentum_reversal',
                        'direction': 'long',
                        'entry_idx': i+1,
                        'pnl_pips': pnl,
                        'z_score': z_score
                    })

    return pd.DataFrame(trades)


def volatility_breakout_strategy(df, atr_period=14, atr_mult=1.5, pip_size=0.01, spread_pips=4):
    """
    Trade volatility breakouts with ATR-based stops/targets.
    Entry: Break of previous high/low + ATR filter
    Exit: 2:1 risk/reward or stop
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Calculate ATR
    tr = np.maximum(highs - lows,
                    np.maximum(abs(highs - np.roll(closes, 1)),
                              abs(lows - np.roll(closes, 1))))
    atr = pd.Series(tr).rolling(atr_period).mean().values

    trades = []
    i = atr_period + 10

    while i < len(closes) - 100:
        # Breakout conditions
        prev_high = max(highs[i-10:i])
        prev_low = min(lows[i-10:i])
        current_atr = atr[i]

        if current_atr > 0:
            # Long breakout
            if closes[i] > prev_high + (current_atr * 0.5):
                entry = closes[i]
                stop = entry - (current_atr * atr_mult)
                target = entry + (current_atr * atr_mult * 2)  # 2:1 RR

                for j in range(i+1, min(i+200, len(closes))):
                    if closes[j] <= stop:
                        exit_price = stop
                        exit_idx = j
                        break
                    elif closes[j] >= target:
                        exit_price = target
                        exit_idx = j
                        break
                else:
                    exit_price = closes[min(i+200, len(closes)-1)]
                    exit_idx = min(i+200, len(closes)-1)

                pnl = (exit_price - entry) / pip_size - spread_pips
                trades.append({
                    'type': 'volatility_breakout',
                    'direction': 'long',
                    'entry_idx': i,
                    'pnl_pips': pnl,
                    'atr': current_atr
                })
                i = exit_idx + 10

            # Short breakout
            elif closes[i] < prev_low - (current_atr * 0.5):
                entry = closes[i]
                stop = entry + (current_atr * atr_mult)
                target = entry - (current_atr * atr_mult * 2)

                for j in range(i+1, min(i+200, len(closes))):
                    if closes[j] >= stop:
                        exit_price = stop
                        exit_idx = j
                        break
                    elif closes[j] <= target:
                        exit_price = target
                        exit_idx = j
                        break
                else:
                    exit_price = closes[min(i+200, len(closes)-1)]
                    exit_idx = min(i+200, len(closes)-1)

                pnl = (entry - exit_price) / pip_size - spread_pips
                trades.append({
                    'type': 'volatility_breakout',
                    'direction': 'short',
                    'entry_idx': i,
                    'pnl_pips': pnl,
                    'atr': current_atr
                })
                i = exit_idx + 10
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(trades)


def simple_ma_cross_strategy(df, fast_ma=10, slow_ma=30, pip_size=0.01, spread_pips=4):
    """
    Classic MA crossover with fixed risk/reward.
    Entry: Fast MA crosses Slow MA
    Exit: Fixed 20 pip stop, 40 pip target (2:1)
    """
    closes = df['close'].values

    # Calculate MAs
    fast = pd.Series(closes).rolling(fast_ma).mean().values
    slow = pd.Series(closes).rolling(slow_ma).mean().values

    trades = []
    i = slow_ma

    while i < len(closes) - 100:
        # Check for crossover
        if i > 0:
            # Bullish cross
            if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
                entry = closes[i]
                stop = entry - (20 * pip_size)  # 20 pip stop
                target = entry + (40 * pip_size)  # 40 pip target

                for j in range(i+1, min(i+500, len(closes))):
                    if closes[j] <= stop:
                        exit_price = stop
                        exit_idx = j
                        break
                    elif closes[j] >= target:
                        exit_price = target
                        exit_idx = j
                        break
                else:
                    exit_price = closes[min(i+500, len(closes)-1)]
                    exit_idx = min(i+500, len(closes)-1)

                pnl = (exit_price - entry) / pip_size - spread_pips
                trades.append({
                    'type': 'ma_cross',
                    'direction': 'long',
                    'entry_idx': i,
                    'pnl_pips': pnl
                })
                i = exit_idx + 1

            # Bearish cross
            elif fast[i-1] >= slow[i-1] and fast[i] < slow[i]:
                entry = closes[i]
                stop = entry + (20 * pip_size)
                target = entry - (40 * pip_size)

                for j in range(i+1, min(i+500, len(closes))):
                    if closes[j] >= stop:
                        exit_price = stop
                        exit_idx = j
                        break
                    elif closes[j] <= target:
                        exit_price = target
                        exit_idx = j
                        break
                else:
                    exit_price = closes[min(i+500, len(closes)-1)]
                    exit_idx = min(i+500, len(closes)-1)

                pnl = (entry - exit_price) / pip_size - spread_pips
                trades.append({
                    'type': 'ma_cross',
                    'direction': 'short',
                    'entry_idx': i,
                    'pnl_pips': pnl
                })
                i = exit_idx + 1
            else:
                i += 1
        else:
            i += 1

    return pd.DataFrame(trades)


def analyze_strategy_performance(trades_df, strategy_name):
    """Analyze and print strategy performance metrics."""
    if len(trades_df) == 0:
        print(f"\n{strategy_name}: No trades generated")
        return None

    print(f"\n{'='*60}")
    print(f"{strategy_name.upper()}")
    print('='*60)

    # Basic metrics
    total_trades = len(trades_df)
    wins = (trades_df['pnl_pips'] > 0).sum()
    win_rate = wins / total_trades * 100
    avg_pnl = trades_df['pnl_pips'].mean()

    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg P&L: {avg_pnl:.1f} pips")

    # Calculate expectancy
    losses = trades_df[trades_df['pnl_pips'] < 0]['pnl_pips']
    if len(losses) > 0:
        R = abs(losses.mean())
        expectancy = avg_pnl / R
        print(f"R (avg loss): {R:.1f} pips")
        print(f"**Expectancy: {expectancy:.3f} R**")

    print(f"Best trade: {trades_df['pnl_pips'].max():.1f} pips")
    print(f"Worst trade: {trades_df['pnl_pips'].min():.1f} pips")

    return {
        'strategy': strategy_name,
        'trades': total_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'expectancy': expectancy if len(losses) > 0 else 0
    }


def main():
    print("🔬 TESTING SIMPLE TRADING STRATEGIES")
    print("=" * 70)

    # Connect to database
    conn = duckdb.connect("/data/master.duckdb", read_only=True)

    # Load data
    print("\n📊 Loading OHLCV data...")
    query = """
    SELECT open, high, low, close, volume
    FROM master
    ORDER BY rowid
    LIMIT 50000
    """
    df = conn.execute(query).df()
    print(f"Loaded {len(df)} bars")

    results = []

    # Test Strategy 1: Momentum Reversal
    print("\nTesting Momentum Reversal Strategy...")
    trades1 = momentum_reversal_strategy(df)
    result1 = analyze_strategy_performance(trades1, "Momentum Reversal")
    if result1:
        results.append(result1)

    # Test Strategy 2: Volatility Breakout
    print("\nTesting Volatility Breakout Strategy...")
    trades2 = volatility_breakout_strategy(df)
    result2 = analyze_strategy_performance(trades2, "Volatility Breakout")
    if result2:
        results.append(result2)

    # Test Strategy 3: MA Cross
    print("\nTesting MA Cross Strategy...")
    trades3 = simple_ma_cross_strategy(df)
    result3 = analyze_strategy_performance(trades3, "MA Cross (10/30)")
    if result3:
        results.append(result3)

    # Summary
    print("\n" + "="*70)
    print("📊 STRATEGY COMPARISON")
    print("="*70)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('expectancy', ascending=False)

        print("\nRanked by Expectancy:")
        for _, row in results_df.iterrows():
            print(f"\n{row['strategy']}:")
            print(f"  Expectancy: {row['expectancy']:.3f} R")
            print(f"  Win rate: {row['win_rate']:.1f}%")
            print(f"  Avg P&L: {row['avg_pnl']:.1f} pips")
            print(f"  Trades: {row['trades']}")

            # Monthly projection
            trades_per_day = row['trades'] / (50000 / 1440)  # Approximate
            monthly_pips = trades_per_day * 22 * row['avg_pnl']
            print(f"  Est. monthly: {monthly_pips:.0f} pips")

    return results_df if results else None


if __name__ == "__main__":
    main()