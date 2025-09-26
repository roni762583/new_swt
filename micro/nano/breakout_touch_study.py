#!/usr/bin/env python3
"""
Breakout Touch Study - Analyze relationship between zone touches and breakout success
Tests the hypothesis that more touches of a zone lead to stronger breakouts
"""

import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt

def breakout_touch_study(df, pip_size=0.01, zone_bars=120, breakout_pips=5, spread_pips=4):
    """
    df: DataFrame with 'close' column (M1 bars)
    pip_size: instrument pip size (0.01 for GBP/JPY, 0.0001 for majors)
    zone_bars: number of minutes to form initial zone (default 120 = 2h)
    breakout_pips: confirmation threshold in pips

    Returns: DataFrame of trades with touch counts, pnl, etc.
    """
    closes = df['close'].values
    trades = []

    i = zone_bars
    while i < len(closes):
        # --- Define zone ---
        zone_high = closes[i-zone_bars:i].max()
        zone_low = closes[i-zone_bars:i].min()
        breakout_thresh = breakout_pips * pip_size

        # --- Count touches until breakout ---
        touches_high = touches_low = 0
        j, breakout_dir, entry_price = i, None, None
        while j < len(closes):
            price = closes[j]
            if abs(price - zone_high) <= breakout_thresh:
                touches_high += 1
            if abs(price - zone_low) <= breakout_thresh:
                touches_low += 1

            if price > zone_high + breakout_thresh:
                breakout_dir, entry_price, entry_idx = "long", price, j
                break
            elif price < zone_low - breakout_thresh:
                breakout_dir, entry_price, entry_idx = "short", price, j
                break
            j += 1

        if breakout_dir is None:
            break  # no breakout until end of data

        # --- Simulate trade exit ---
        exit_idx, exit_price = None, None
        if breakout_dir == "long":
            last_low, last_high = entry_price, entry_price
            for k in range(entry_idx+1, min(entry_idx+1000, len(closes))):  # Cap at 1000 bars
                if closes[k] > last_high:
                    last_high = closes[k]  # new HH
                elif closes[k] < last_low:
                    last_low = closes[k]   # new low
                # trend reversal: lower high + lower low
                if closes[k] < last_low and closes[k] < entry_price:
                    exit_idx, exit_price = k, closes[k]
                    break
        else:  # short
            last_low, last_high = entry_price, entry_price
            for k in range(entry_idx+1, min(entry_idx+1000, len(closes))):  # Cap at 1000 bars
                if closes[k] < last_low:
                    last_low = closes[k]  # new LL
                elif closes[k] > last_high:
                    last_high = closes[k] # new high
                # trend reversal: higher high + higher low
                if closes[k] > last_high and closes[k] > entry_price:
                    exit_idx, exit_price = k, closes[k]
                    break

        if exit_idx is None:
            exit_idx = min(entry_idx + 1000, len(closes)-1)
            exit_price = closes[exit_idx]

        # Calculate P&L with spread cost
        raw_pnl = (exit_price - entry_price) / pip_size * (1 if breakout_dir=="long" else -1)
        net_pnl = raw_pnl - spread_pips  # Subtract spread cost

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "direction": breakout_dir,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_pnl_pips": raw_pnl,
            "pnl_pips": net_pnl,  # Net after spread
            "touches_high": touches_high,
            "touches_low": touches_low,
            "touches_total": touches_high + touches_low,
            "bars_held": exit_idx - entry_idx
        })

        # Reset after trade completes
        i = exit_idx + zone_bars

    return pd.DataFrame(trades)


def analyze_touch_correlation(trades_df):
    """Analyze correlation between touches and PnL"""
    print("\n📊 TOUCH COUNT vs PNL ANALYSIS")
    print("=" * 60)

    # Group by touch buckets
    touch_buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000)]

    for low, high in touch_buckets:
        mask = (trades_df['touches_total'] >= low) & (trades_df['touches_total'] < high)
        bucket_trades = trades_df[mask]

        if len(bucket_trades) > 0:
            avg_pnl = bucket_trades['pnl_pips'].mean()
            win_rate = (bucket_trades['pnl_pips'] > 0).mean() * 100
            count = len(bucket_trades)

            print(f"\nTouches [{low:3d}-{high:3d}): {count:4d} trades")
            print(f"  Avg PnL: {avg_pnl:+.1f} pips")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Avg Hold: {bucket_trades['bars_held'].mean():.0f} bars")


def run_comprehensive_analysis():
    """Run complete breakout touch study"""
    print("🔬 BREAKOUT TOUCH STUDY ANALYSIS")
    print("=" * 70)

    # Connect to database
    conn = duckdb.connect("/data/master.duckdb", read_only=True)

    # Load data
    print("\n📊 Loading OHLCV data...")
    query = """
    SELECT close
    FROM master
    ORDER BY rowid
    LIMIT 100000
    """
    df = conn.execute(query).df()
    print(f"Loaded {len(df)} bars")

    # Test different parameter combinations
    param_sets = [
        {'zone_bars': 60, 'breakout_pips': 3, 'name': 'Fast (1H zones, 3 pips)'},
        {'zone_bars': 120, 'breakout_pips': 5, 'name': 'Standard (2H zones, 5 pips)'},
        {'zone_bars': 240, 'breakout_pips': 8, 'name': 'Slow (4H zones, 8 pips)'},
    ]

    all_results = []

    for params in param_sets:
        print(f"\n{'='*60}")
        print(f"Testing: {params['name']}")
        print('='*60)

        trades_df = breakout_touch_study(
            df,
            pip_size=0.01,  # GBP/JPY pip size
            zone_bars=params['zone_bars'],
            breakout_pips=params['breakout_pips'],
            spread_pips=4  # 4-pip spread cost
        )

        if len(trades_df) > 0:
            print(f"\n📈 Results (with 4-pip spread cost):")
            print(f"  Total trades: {len(trades_df)}")
            print(f"  Avg PnL (net): {trades_df['pnl_pips'].mean():.1f} pips")
            print(f"  Avg PnL (raw): {trades_df['raw_pnl_pips'].mean():.1f} pips")
            print(f"  Spread impact: {(trades_df['raw_pnl_pips'].mean() - trades_df['pnl_pips'].mean()):.1f} pips/trade")
            print(f"  Win rate: {(trades_df['pnl_pips'] > 0).mean() * 100:.1f}%")
            print(f"  Best trade: {trades_df['pnl_pips'].max():.1f} pips")
            print(f"  Worst trade: {trades_df['pnl_pips'].min():.1f} pips")
            print(f"  Avg touches: {trades_df['touches_total'].mean():.1f}")

            # Analyze touch correlation
            analyze_touch_correlation(trades_df)

            # Calculate correlation
            if len(trades_df) > 10:
                corr = trades_df[['touches_total', 'pnl_pips']].corr().iloc[0, 1]
                print(f"\n📊 Correlation (touches vs PnL): {corr:.4f}")

            # Add to results
            trades_df['params'] = params['name']
            all_results.append(trades_df)

    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save results
        combined_df.to_csv('breakout_touch_results.csv', index=False)
        print(f"\n📁 Results saved to breakout_touch_results.csv")

        # Overall statistics
        print("\n" + "="*70)
        print("📊 OVERALL SUMMARY")
        print("="*70)

        print(f"Total trades analyzed: {len(combined_df)}")
        print(f"Overall win rate: {(combined_df['pnl_pips'] > 0).mean() * 100:.1f}%")
        print(f"Average PnL: {combined_df['pnl_pips'].mean():.1f} pips")

        # Best parameter set
        best_params = combined_df.groupby('params')['pnl_pips'].mean().idxmax()
        best_avg = combined_df.groupby('params')['pnl_pips'].mean().max()
        print(f"\nBest parameters: {best_params}")
        print(f"Best average PnL: {best_avg:.1f} pips")

        # Touch analysis summary
        print("\n📊 TOUCH COUNT INSIGHTS:")
        quartiles = combined_df['touches_total'].quantile([0.25, 0.5, 0.75])

        low_touch = combined_df[combined_df['touches_total'] <= quartiles[0.25]]
        high_touch = combined_df[combined_df['touches_total'] >= quartiles[0.75]]

        print(f"Low touch trades (≤{quartiles[0.25]:.0f} touches):")
        print(f"  Avg PnL: {low_touch['pnl_pips'].mean():.1f} pips")
        print(f"  Win rate: {(low_touch['pnl_pips'] > 0).mean() * 100:.1f}%")

        print(f"\nHigh touch trades (≥{quartiles[0.75]:.0f} touches):")
        print(f"  Avg PnL: {high_touch['pnl_pips'].mean():.1f} pips")
        print(f"  Win rate: {(high_touch['pnl_pips'] > 0).mean() * 100:.1f}%")

        return combined_df

    return None


def main():
    results = run_comprehensive_analysis()

    if results is not None:
        print("\n✅ Analysis complete!")
        return results
    else:
        print("\n⚠️ No trades generated")
        return None


if __name__ == "__main__":
    main()