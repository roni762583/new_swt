#!/usr/bin/env python3
"""
Calculate theoretical pips accumulated by trading ZigZag swings.

Strategy:
- Enter 2 bars after each pivot confirmation
- Exit 1 bar before next pivot
- Direction: Follow ZigZag trend (buy on uptrend, sell on downtrend)
- Calculate pips gained/lost per trade and cumulative performance
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

DB_PATH = Path("/home/aharon/projects/new_swt/micro/nano/picco-ppo/master.duckdb")

def calculate_zigzag_trades(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Calculate all theoretical ZigZag trades with entry/exit prices and pips."""

    print("="*80)
    print("ZIGZAG THEORETICAL TRADING PERFORMANCE")
    print("="*80)

    # Fetch data
    print("\nFetching data...")
    df = conn.execute("""
        SELECT
            bar_index,
            timestamp,
            open,
            high,
            low,
            close,
            is_zigzag_pivot,
            zigzag_price,
            zigzag_direction
        FROM master
        ORDER BY bar_index
    """).fetch_df()

    print(f"Loaded {len(df):,} bars")

    # Find pivot indices
    pivot_indices = df[df['is_zigzag_pivot'] == True].index.tolist()
    print(f"Found {len(pivot_indices):,} ZigZag pivots")

    if len(pivot_indices) < 2:
        print("Not enough pivots to calculate trades")
        return pd.DataFrame()

    # Calculate trades
    trades = []

    for i in range(len(pivot_indices) - 1):
        pivot_idx = pivot_indices[i]
        next_pivot_idx = pivot_indices[i + 1]

        # Entry: 2 bars after pivot
        entry_idx = pivot_idx + 2
        if entry_idx >= len(df):
            continue

        # Exit: 1 bar before next pivot
        exit_idx = next_pivot_idx - 1
        if exit_idx <= entry_idx:
            continue

        # Get entry/exit data
        entry_bar = df.iloc[entry_idx]
        exit_bar = df.iloc[exit_idx]

        # Direction at entry
        direction = entry_bar['zigzag_direction']

        if direction == 0:
            continue  # Skip undefined direction

        # Entry/exit prices (use close for consistency)
        entry_price = entry_bar['close']
        exit_price = exit_bar['close']

        # Calculate pips
        if direction == 1:  # Long (buy)
            pips = (exit_price - entry_price) * 100
            trade_type = 'LONG'
        elif direction == -1:  # Short (sell)
            pips = (entry_price - exit_price) * 100
            trade_type = 'SHORT'
        else:
            continue

        # Duration
        duration_bars = exit_idx - entry_idx
        entry_time = entry_bar['timestamp']
        exit_time = exit_bar['timestamp']

        trades.append({
            'trade_num': len(trades) + 1,
            'type': trade_type,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pips': pips,
            'duration_bars': duration_bars,
            'pivot_idx': pivot_idx,
            'next_pivot_idx': next_pivot_idx,
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        print("No valid trades generated")
        return trades_df

    # Calculate statistics
    print("\n" + "="*80)
    print("TRADE STATISTICS")
    print("="*80)

    total_trades = len(trades_df)
    winners = trades_df[trades_df['pips'] > 0]
    losers = trades_df[trades_df['pips'] <= 0]

    total_pips = trades_df['pips'].sum()
    avg_pips = trades_df['pips'].mean()
    win_rate = len(winners) / total_trades * 100

    print(f"\nTotal trades: {total_trades:,}")
    print(f"Winners: {len(winners):,} ({len(winners)/total_trades*100:.1f}%)")
    print(f"Losers: {len(losers):,} ({len(losers)/total_trades*100:.1f}%)")

    print(f"\n📊 PROFIT/LOSS:")
    print(f"  Total pips: {total_pips:+.1f}")
    print(f"  Average pips/trade: {avg_pips:+.2f}")
    print(f"  Best trade: {trades_df['pips'].max():+.1f} pips")
    print(f"  Worst trade: {trades_df['pips'].min():+.1f} pips")

    if len(winners) > 0:
        print(f"\n✅ WINNERS:")
        print(f"  Count: {len(winners):,}")
        print(f"  Total pips: {winners['pips'].sum():+.1f}")
        print(f"  Avg pips: {winners['pips'].mean():+.2f}")
        print(f"  Avg duration: {winners['duration_bars'].mean():.0f} bars")

    if len(losers) > 0:
        print(f"\n❌ LOSERS:")
        print(f"  Count: {len(losers):,}")
        print(f"  Total pips: {losers['pips'].sum():+.1f}")
        print(f"  Avg pips: {losers['pips'].mean():+.2f}")
        print(f"  Avg duration: {losers['duration_bars'].mean():.0f} bars")

    # Type breakdown
    longs = trades_df[trades_df['type'] == 'LONG']
    shorts = trades_df[trades_df['type'] == 'SHORT']

    print(f"\n📈 TRADE TYPES:")
    print(f"  Longs: {len(longs):,} ({len(longs)/total_trades*100:.1f}%), {longs['pips'].sum():+.1f} pips")
    print(f"  Shorts: {len(shorts):,} ({len(shorts)/total_trades*100:.1f}%), {shorts['pips'].sum():+.1f} pips")

    # Cumulative performance
    trades_df['cumulative_pips'] = trades_df['pips'].cumsum()

    print(f"\n📊 CUMULATIVE PERFORMANCE:")
    print(f"  Starting: 0 pips")
    print(f"  Ending: {trades_df['cumulative_pips'].iloc[-1]:+.1f} pips")
    print(f"  Peak: {trades_df['cumulative_pips'].max():+.1f} pips")
    print(f"  Trough: {trades_df['cumulative_pips'].min():+.1f} pips")

    # Risk metrics
    max_dd = 0
    peak = 0
    for cum_pips in trades_df['cumulative_pips']:
        if cum_pips > peak:
            peak = cum_pips
        dd = peak - cum_pips
        if dd > max_dd:
            max_dd = dd

    print(f"\n⚠️  RISK METRICS:")
    print(f"  Max drawdown: {max_dd:.1f} pips")

    # Expectancy
    if len(losers) > 0:
        expectancy = (len(winners)/total_trades * winners['pips'].mean()) - \
                     (len(losers)/total_trades * abs(losers['pips'].mean()))
        print(f"  Expectancy: {expectancy:+.2f} pips/trade")

    # Profit factor
    if losers['pips'].sum() != 0:
        profit_factor = abs(winners['pips'].sum() / losers['pips'].sum())
        print(f"  Profit factor: {profit_factor:.2f}")

    print("\n" + "="*80)

    return trades_df


def save_trades_to_csv(trades_df: pd.DataFrame, output_path: Path):
    """Save trades to CSV for further analysis."""
    if len(trades_df) > 0:
        trades_df.to_csv(output_path, index=False)
        print(f"\n✅ Trades saved to: {output_path}")
        print(f"   {len(trades_df):,} trades exported")


def main():
    """Main execution."""
    conn = duckdb.connect(str(DB_PATH))

    # Calculate trades
    trades_df = calculate_zigzag_trades(conn)

    # Save to CSV
    if len(trades_df) > 0:
        output_path = DB_PATH.parent / "zigzag_theoretical_trades.csv"
        save_trades_to_csv(trades_df, output_path)

        # Show first and last few trades
        print("\n" + "="*80)
        print("FIRST 10 TRADES:")
        print("="*80)
        print(trades_df[['trade_num', 'type', 'entry_time', 'entry_price', 'exit_price', 'pips', 'cumulative_pips']].head(10).to_string(index=False))

        print("\n" + "="*80)
        print("LAST 10 TRADES:")
        print("="*80)
        print(trades_df[['trade_num', 'type', 'entry_time', 'entry_price', 'exit_price', 'pips', 'cumulative_pips']].tail(10).to_string(index=False))

    conn.close()
    print("\n✅ ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
