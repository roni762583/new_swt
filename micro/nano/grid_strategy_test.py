#!/usr/bin/env python3
"""
Test grid trading strategy - works well in ranging markets.
"""

import numpy as np
import pandas as pd
import duckdb

def simple_grid_strategy(df, grid_size=20, max_positions=5, pip_size=0.01, spread_pips=4):
    """
    Simple grid trading - place orders at fixed intervals.
    Close when price returns to entry or hits stop.
    """
    closes = df['close'].values
    trades = []

    # Grid levels in pips
    grid_pips = grid_size * pip_size

    i = 100  # Start after some warmup
    open_positions = []

    while i < len(closes) - 100:
        current_price = closes[i]

        # Check if we should open a new position
        can_open = True
        for pos in open_positions:
            # Don't open if too close to existing position
            if abs(current_price - pos['entry']) < grid_pips:
                can_open = False
                break

        if can_open and len(open_positions) < max_positions:
            # Determine direction based on recent momentum
            momentum = closes[i] - closes[i-10]

            if momentum < 0:  # Price falling, buy
                entry = current_price
                take_profit = entry + grid_pips
                stop_loss = entry - (grid_pips * 2)

                open_positions.append({
                    'direction': 'long',
                    'entry': entry,
                    'tp': take_profit,
                    'sl': stop_loss,
                    'entry_idx': i
                })

            elif momentum > 0:  # Price rising, sell
                entry = current_price
                take_profit = entry - grid_pips
                stop_loss = entry + (grid_pips * 2)

                open_positions.append({
                    'direction': 'short',
                    'entry': entry,
                    'tp': take_profit,
                    'sl': stop_loss,
                    'entry_idx': i
                })

        # Check exits for open positions
        positions_to_close = []
        for j, pos in enumerate(open_positions):
            if pos['direction'] == 'long':
                if closes[i] >= pos['tp']:
                    pnl = (pos['tp'] - pos['entry']) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'bars_held': i - pos['entry_idx'],
                        'exit_type': 'tp'
                    })
                    positions_to_close.append(j)
                elif closes[i] <= pos['sl']:
                    pnl = (pos['sl'] - pos['entry']) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'bars_held': i - pos['entry_idx'],
                        'exit_type': 'sl'
                    })
                    positions_to_close.append(j)

            else:  # short
                if closes[i] <= pos['tp']:
                    pnl = (pos['entry'] - pos['tp']) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'bars_held': i - pos['entry_idx'],
                        'exit_type': 'tp'
                    })
                    positions_to_close.append(j)
                elif closes[i] >= pos['sl']:
                    pnl = (pos['entry'] - pos['sl']) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'bars_held': i - pos['entry_idx'],
                        'exit_type': 'sl'
                    })
                    positions_to_close.append(j)

        # Remove closed positions
        for j in sorted(positions_to_close, reverse=True):
            del open_positions[j]

        i += 1

    return pd.DataFrame(trades)


def adaptive_grid_strategy(df, base_grid=15, atr_period=20, pip_size=0.01, spread_pips=4):
    """
    Adaptive grid that adjusts size based on volatility (ATR).
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    # Calculate ATR
    tr = []
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        ))
    atr = pd.Series(tr).rolling(atr_period).mean().values

    trades = []
    i = atr_period + 1

    while i < len(closes) - 100:
        current_atr = atr[i-1] if i > 0 and not np.isnan(atr[i-1]) else 0.001

        # Adaptive grid size based on ATR
        grid_size = max(base_grid, int(current_atr / pip_size))
        grid_pips = grid_size * pip_size

        # Simple reversion strategy
        sma20 = np.mean(closes[max(0, i-20):i])
        distance_from_mean = closes[i] - sma20

        # If price is far from mean, trade back to mean
        if abs(distance_from_mean) > grid_pips:
            if distance_from_mean > 0:  # Price above mean, short
                entry = closes[i]
                target = sma20
                stop = entry + (grid_pips * 2)

                # Simulate trade
                for j in range(i+1, min(i+100, len(closes))):
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
                    'pnl_pips': pnl,
                    'grid_size': grid_size,
                    'direction': 'short'
                })
                i += 20  # Skip ahead

            else:  # Price below mean, long
                entry = closes[i]
                target = sma20
                stop = entry - (grid_pips * 2)

                for j in range(i+1, min(i+100, len(closes))):
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
                    'pnl_pips': pnl,
                    'grid_size': grid_size,
                    'direction': 'long'
                })
                i += 20
        else:
            i += 1

    return pd.DataFrame(trades)


def mean_reversion_grid(df, threshold_std=1.5, profit_target=10, pip_size=0.01, spread_pips=4):
    """
    Pure mean reversion - enter when price is X std devs from mean.
    Fixed profit target, no stop loss (grid averaging).
    """
    closes = df['close'].values
    trades = []

    for i in range(50, len(closes) - 100):
        # Calculate rolling stats
        window = closes[max(0, i-50):i]
        mean = np.mean(window)
        std = np.std(window)

        if std > 0:
            z_score = (closes[i] - mean) / std

            # Only trade extreme deviations
            if abs(z_score) > threshold_std:
                if z_score > threshold_std:  # Overbought, short
                    entry = closes[i]
                    target = entry - (profit_target * pip_size)

                    # No stop, just target
                    for j in range(i+1, min(i+200, len(closes))):
                        if closes[j] <= target:
                            exit_price = target
                            exit_idx = j
                            break
                    else:
                        exit_price = closes[min(i+200, len(closes)-1)]
                        exit_idx = min(i+200, len(closes)-1)

                    pnl = (entry - exit_price) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'z_score': z_score,
                        'bars_held': exit_idx - i,
                        'direction': 'short'
                    })

                elif z_score < -threshold_std:  # Oversold, long
                    entry = closes[i]
                    target = entry + (profit_target * pip_size)

                    for j in range(i+1, min(i+200, len(closes))):
                        if closes[j] >= target:
                            exit_price = target
                            exit_idx = j
                            break
                    else:
                        exit_price = closes[min(i+200, len(closes)-1)]
                        exit_idx = min(i+200, len(closes)-1)

                    pnl = (exit_price - entry) / pip_size - spread_pips
                    trades.append({
                        'pnl_pips': pnl,
                        'z_score': z_score,
                        'bars_held': exit_idx - i,
                        'direction': 'long'
                    })

    return pd.DataFrame(trades)


def analyze_strategy(trades_df, name):
    """Analyze strategy results."""
    if len(trades_df) == 0:
        print(f"\n{name}: No trades")
        return None

    print(f"\n{'='*60}")
    print(f"{name.upper()}")
    print('='*60)

    wins = (trades_df['pnl_pips'] > 0).sum()
    total = len(trades_df)
    win_rate = wins / total * 100
    avg_pnl = trades_df['pnl_pips'].mean()

    print(f"Trades: {total}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg P&L: {avg_pnl:.1f} pips")

    losses = trades_df[trades_df['pnl_pips'] < 0]['pnl_pips']
    if len(losses) > 0:
        R = abs(losses.mean())
        expectancy = avg_pnl / R
        print(f"R: {R:.1f} pips")
        print(f"**Expectancy: {expectancy:.3f} R**")
    else:
        expectancy = float('inf')  # All winners
        R = 0
        print("**All winning trades!**")

    print(f"Best: {trades_df['pnl_pips'].max():.1f} pips")
    print(f"Worst: {trades_df['pnl_pips'].min():.1f} pips")

    if 'bars_held' in trades_df.columns:
        print(f"Avg hold: {trades_df['bars_held'].mean():.0f} bars")

    return {'name': name, 'trades': total, 'win_rate': win_rate,
            'avg_pnl': avg_pnl, 'expectancy': expectancy, 'R': R}


def main():
    print("🔬 TESTING GRID AND MEAN REVERSION STRATEGIES")
    print("=" * 70)

    conn = duckdb.connect("/data/master.duckdb", read_only=True)
    query = """
    SELECT open, high, low, close, volume
    FROM master
    ORDER BY rowid
    LIMIT 30000
    """
    df = conn.execute(query).df()
    print(f"Loaded {len(df)} bars\n")

    results = []

    # Test strategies
    trades1 = simple_grid_strategy(df)
    r1 = analyze_strategy(trades1, "Simple Grid")
    if r1: results.append(r1)

    trades2 = adaptive_grid_strategy(df)
    r2 = analyze_strategy(trades2, "Adaptive Grid (ATR)")
    if r2: results.append(r2)

    trades3 = mean_reversion_grid(df)
    r3 = analyze_strategy(trades3, "Mean Reversion (No Stop)")
    if r3: results.append(r3)

    # Summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - BEST STRATEGIES")
    print("="*70)

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('expectancy', ascending=False)

        for _, row in df_results.iterrows():
            if row['expectancy'] > 0 or row['expectancy'] == float('inf'):
                print(f"\n✅ {row['name']}:")
                if row['expectancy'] == float('inf'):
                    print(f"   **Perfect record - all wins!**")
                else:
                    print(f"   **Expectancy: {row['expectancy']:.3f} R**")
                print(f"   Win rate: {row['win_rate']:.1f}%")
                print(f"   Avg P&L: {row['avg_pnl']:.1f} pips")
                print(f"   Trades: {row['trades']}")

                # Monthly projection
                trades_per_day = row['trades'] / (30000 / 1440)
                monthly = trades_per_day * 22 * row['avg_pnl']
                print(f"   Monthly est: {monthly:.0f} pips")


if __name__ == "__main__":
    main()