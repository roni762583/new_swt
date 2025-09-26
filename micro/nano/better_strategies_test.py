#!/usr/bin/env python3
"""
Test improved trading strategies focusing on market structure.
"""

import numpy as np
import pandas as pd
import duckdb

def range_compression_breakout(df, lookback=30, compression_pct=0.6, pip_size=0.01, spread_pips=4):
    """
    Trade breakouts from compressed ranges (low volatility to high volatility).
    Logic: When recent range is <60% of normal range, expect expansion.
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    trades = []

    for i in range(lookback*2, len(closes) - 100):
        # Calculate average range for lookback period
        recent_ranges = highs[i-lookback:i] - lows[i-lookback:i]
        historical_ranges = highs[i-lookback*2:i-lookback] - lows[i-lookback*2:i-lookback]

        avg_recent = np.mean(recent_ranges)
        avg_historical = np.mean(historical_ranges)

        if avg_historical > 0:
            compression_ratio = avg_recent / avg_historical

            # Look for compressed range
            if compression_ratio < compression_pct:
                # Setup for breakout - use recent high/low as triggers
                recent_high = max(highs[i-5:i])
                recent_low = min(lows[i-5:i])
                range_size = recent_high - recent_low

                # Wait for breakout (max 20 bars)
                for j in range(i, min(i+20, len(closes)-80)):
                    # Long breakout
                    if closes[j] > recent_high:
                        entry = closes[j]
                        stop = recent_low  # Stop at opposite side of range
                        target = entry + (range_size * 2)  # Target 2x the range

                        # Simulate trade
                        for k in range(j+1, min(j+80, len(closes))):
                            if closes[k] <= stop:
                                exit_price = stop
                                break
                            elif closes[k] >= target:
                                exit_price = target
                                break
                        else:
                            exit_price = closes[min(j+80, len(closes)-1)]

                        pnl = (exit_price - entry) / pip_size - spread_pips
                        trades.append({
                            'type': 'range_compression',
                            'direction': 'long',
                            'entry_idx': j,
                            'pnl_pips': pnl,
                            'compression': compression_ratio
                        })
                        break

                    # Short breakout
                    elif closes[j] < recent_low:
                        entry = closes[j]
                        stop = recent_high
                        target = entry - (range_size * 2)

                        for k in range(j+1, min(j+80, len(closes))):
                            if closes[k] >= stop:
                                exit_price = stop
                                break
                            elif closes[k] <= target:
                                exit_price = target
                                break
                        else:
                            exit_price = closes[min(j+80, len(closes)-1)]

                        pnl = (entry - exit_price) / pip_size - spread_pips
                        trades.append({
                            'type': 'range_compression',
                            'direction': 'short',
                            'entry_idx': j,
                            'pnl_pips': pnl,
                            'compression': compression_ratio
                        })
                        break

    return pd.DataFrame(trades)


def inside_bar_breakout(df, min_bars=3, pip_size=0.01, spread_pips=4):
    """
    Trade breakouts from inside bar patterns.
    Inside bar = bar completely within range of previous bar.
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    trades = []
    i = 1

    while i < len(closes) - 100:
        # Count consecutive inside bars
        inside_count = 0
        mother_high = highs[i-1]
        mother_low = lows[i-1]

        for j in range(i, min(i+10, len(closes))):
            if highs[j] <= mother_high and lows[j] >= mother_low:
                inside_count += 1
            else:
                break

        # If we have enough inside bars, trade the breakout
        if inside_count >= min_bars:
            last_inside = i + inside_count - 1

            # Wait for breakout
            for k in range(last_inside + 1, min(last_inside + 10, len(closes) - 80)):
                # Long breakout
                if closes[k] > mother_high:
                    entry = closes[k]
                    stop = mother_low
                    range_size = mother_high - mother_low
                    target = entry + (range_size * 1.5)  # 1.5:1 RR

                    for m in range(k+1, min(k+80, len(closes))):
                        if closes[m] <= stop:
                            exit_price = stop
                            break
                        elif closes[m] >= target:
                            exit_price = target
                            break
                    else:
                        exit_price = closes[min(k+80, len(closes)-1)]

                    pnl = (exit_price - entry) / pip_size - spread_pips
                    trades.append({
                        'type': 'inside_bar',
                        'direction': 'long',
                        'entry_idx': k,
                        'pnl_pips': pnl,
                        'inside_bars': inside_count
                    })
                    i = k + 10
                    break

                # Short breakout
                elif closes[k] < mother_low:
                    entry = closes[k]
                    stop = mother_high
                    range_size = mother_high - mother_low
                    target = entry - (range_size * 1.5)

                    for m in range(k+1, min(k+80, len(closes))):
                        if closes[m] >= stop:
                            exit_price = stop
                            break
                        elif closes[m] <= target:
                            exit_price = target
                            break
                    else:
                        exit_price = closes[min(k+80, len(closes)-1)]

                    pnl = (entry - exit_price) / pip_size - spread_pips
                    trades.append({
                        'type': 'inside_bar',
                        'direction': 'short',
                        'entry_idx': k,
                        'pnl_pips': pnl,
                        'inside_bars': inside_count
                    })
                    i = k + 10
                    break
            else:
                i += inside_count
        else:
            i += 1

    return pd.DataFrame(trades)


def session_open_momentum(df, session_bars=60, momentum_threshold=0.001, pip_size=0.01, spread_pips=4):
    """
    Trade momentum at session opens (every 4 hours = 240 bars).
    If first hour moves strongly, trade in that direction.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    trades = []

    # Check every 240 bars (4-hour sessions)
    for i in range(240, len(closes) - 200, 240):
        # Measure first hour momentum
        session_open = closes[i]
        hour_close = closes[min(i + session_bars, len(closes)-1)]
        momentum = (hour_close - session_open) / session_open

        if abs(momentum) > momentum_threshold:
            # Strong momentum detected
            if momentum > 0:  # Bullish
                entry = hour_close
                recent_low = min(lows[i:i+session_bars])
                stop = recent_low - (10 * pip_size)  # 10 pip buffer
                target = entry + abs(entry - stop) * 2  # 2:1 RR

                for j in range(i+session_bars+1, min(i+200, len(closes))):
                    if closes[j] <= stop:
                        exit_price = stop
                        break
                    elif closes[j] >= target:
                        exit_price = target
                        break
                else:
                    exit_price = closes[min(i+200, len(closes)-1)]

                pnl = (exit_price - entry) / pip_size - spread_pips
                trades.append({
                    'type': 'session_momentum',
                    'direction': 'long',
                    'entry_idx': i+session_bars,
                    'pnl_pips': pnl,
                    'momentum': momentum
                })

            else:  # Bearish
                entry = hour_close
                recent_high = max(highs[i:i+session_bars])
                stop = recent_high + (10 * pip_size)
                target = entry - abs(stop - entry) * 2

                for j in range(i+session_bars+1, min(i+200, len(closes))):
                    if closes[j] >= stop:
                        exit_price = stop
                        break
                    elif closes[j] <= target:
                        exit_price = target
                        break
                else:
                    exit_price = closes[min(i+200, len(closes)-1)]

                pnl = (entry - exit_price) / pip_size - spread_pips
                trades.append({
                    'type': 'session_momentum',
                    'direction': 'short',
                    'entry_idx': i+session_bars,
                    'pnl_pips': pnl,
                    'momentum': momentum
                })

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
    else:
        expectancy = 0
        R = 0

    print(f"Best trade: {trades_df['pnl_pips'].max():.1f} pips")
    print(f"Worst trade: {trades_df['pnl_pips'].min():.1f} pips")

    # Additional analysis
    if 'inside_bars' in trades_df.columns:
        print(f"Avg inside bars: {trades_df['inside_bars'].mean():.1f}")
    if 'compression' in trades_df.columns:
        print(f"Avg compression: {trades_df['compression'].mean():.2f}")

    return {
        'strategy': strategy_name,
        'trades': total_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'expectancy': expectancy,
        'R': R
    }


def main():
    print("🔬 TESTING IMPROVED TRADING STRATEGIES")
    print("=" * 70)

    # Connect to database
    conn = duckdb.connect("/data/master.duckdb", read_only=True)

    # Load data with OHLC
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

    # Test Strategy 1: Range Compression Breakout
    print("\nTesting Range Compression Breakout...")
    trades1 = range_compression_breakout(df)
    result1 = analyze_strategy_performance(trades1, "Range Compression")
    if result1:
        results.append(result1)

    # Test Strategy 2: Inside Bar Breakout
    print("\nTesting Inside Bar Breakout...")
    trades2 = inside_bar_breakout(df)
    result2 = analyze_strategy_performance(trades2, "Inside Bar Breakout")
    if result2:
        results.append(result2)

    # Test Strategy 3: Session Momentum
    print("\nTesting Session Open Momentum...")
    trades3 = session_open_momentum(df)
    result3 = analyze_strategy_performance(trades3, "Session Momentum")
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
            print(f"  **Expectancy: {row['expectancy']:.3f} R**")
            print(f"  Win rate: {row['win_rate']:.1f}%")
            print(f"  Avg P&L: {row['avg_pnl']:.1f} pips")
            print(f"  R (avg loss): {row['R']:.1f} pips")
            print(f"  Trades: {row['trades']}")

            # Monthly projection
            trades_per_day = row['trades'] / (50000 / 1440)  # Approximate
            monthly_pips = trades_per_day * 22 * row['avg_pnl']
            print(f"  Est. monthly: {monthly_pips:.0f} pips ({trades_per_day*22:.0f} trades)")

        # Find best strategy
        best = results_df.iloc[0]
        if best['expectancy'] > 0:
            print(f"\n✅ BEST STRATEGY: {best['strategy']}")
            print(f"   Positive expectancy of {best['expectancy']:.3f}R")
        else:
            print("\n❌ All strategies have negative expectancy")
            print("   Consider different market conditions or parameters")

    return results_df if results else None


if __name__ == "__main__":
    main()