#!/usr/bin/env python3
"""
Terminal-based ASCII graph of rolling expectancy.
Perfect for monitoring during Docker training.
"""

import json
import os
import time
from datetime import datetime

def load_expectancy_data(filepath="results/rolling_expectancy.json"):
    """Load expectancy data from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def create_ascii_bar(value, max_width=40, min_val=-0.5, max_val=1.0):
    """Create ASCII bar chart."""
    # Normalize value to 0-1 range
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))  # Clamp between 0 and 1

    # Calculate bar length
    bar_length = int(normalized * max_width)

    # Choose color/symbol based on value
    if value < 0:
        symbol = '▓'
        color = '\033[91m'  # Red
    elif value < 0.25:
        symbol = '▒'
        color = '\033[93m'  # Yellow
    elif value < 0.5:
        symbol = '█'
        color = '\033[92m'  # Green
    else:
        symbol = '█'
        color = '\033[94m'  # Blue (excellent)

    # Create bar
    bar = color + symbol * bar_length + '\033[0m'
    spaces = ' ' * (max_width - bar_length)

    return bar + spaces

def display_expectancy_terminal(continuous=False):
    """Display expectancy in terminal format."""

    while True:
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        data = load_expectancy_data()

        if data is None:
            print("⏳ Waiting for expectancy data...")
            if continuous:
                time.sleep(5)
                continue
            else:
                break

        # Header
        print("=" * 70)
        print("📊 PPO TRADING PERFORMANCE - ROLLING EXPECTANCY")
        print("=" * 70)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Episode and trades info
        print(f"📈 Episode: {data.get('episode', 0)}")
        print(f"🎯 Total Trades: {data.get('total_trades', 0):,}")
        print(f"💰 Cumulative P&L: {data.get('cumulative_pips', 0):+.1f} pips")
        print()

        # Rolling expectancy bars
        print("ROLLING EXPECTANCY (R-multiples):")
        print("-" * 70)

        windows = [100, 500, 1000]
        for window in windows:
            exp_key = f'expectancy_R_{window}'
            sample_key = f'sample_size_{window}'
            wr_key = f'win_rate_{window}'

            if exp_key in data:
                exp_r = data[exp_key]
                sample = data.get(sample_key, 0)
                win_rate = data.get(wr_key, 0)

                # Quality indicator
                if exp_r > 0.5:
                    quality = "🏆 EXCELLENT"
                elif exp_r > 0.25:
                    quality = "✅ GOOD"
                elif exp_r > 0:
                    quality = "⚠️  ACCEPTABLE"
                else:
                    quality = "🔴 POOR"

                # Create bar
                bar = create_ascii_bar(exp_r)

                # Display
                print(f"{window:4d}-trade: [{bar}] {exp_r:+.3f}R {quality}")
                print(f"           Win Rate: {win_rate:.1f}% | Sample: {sample}/{window}")
                print()

        # Lifetime expectancy
        print("-" * 70)
        lifetime_exp = data.get('lifetime_expectancy_R', 0)
        lifetime_wr = data.get('lifetime_win_rate', 0)
        lifetime_trades = data.get('lifetime_trades', 0)

        bar = create_ascii_bar(lifetime_exp)
        print(f"LIFETIME:  [{bar}] {lifetime_exp:+.3f}R")
        print(f"           Win Rate: {lifetime_wr:.1f}% | Trades: {lifetime_trades:,}")

        # Legend
        print()
        print("=" * 70)
        print("LEGEND: 🔴 <0R (losing) | ⚠️ 0-0.25R | ✅ 0.25-0.5R | 🏆 >0.5R")

        # Performance trend
        if 'expectancy_R_100' in data and 'expectancy_R_1000' in data:
            trend = data['expectancy_R_100'] - data['expectancy_R_1000']
            if trend > 0.05:
                print("📈 TREND: Recent performance IMPROVING")
            elif trend < -0.05:
                print("📉 TREND: Recent performance DECLINING")
            else:
                print("➡️ TREND: Performance STABLE")

        print("=" * 70)

        if not continuous:
            break

        print("\n🔄 Refreshing in 10 seconds... (Press Ctrl+C to exit)")
        time.sleep(10)

def generate_mini_summary():
    """Generate a compact summary for quick checking."""
    data = load_expectancy_data()

    if data is None:
        return "No data available"

    lifetime_exp = data.get('lifetime_expectancy_R', 0)
    total_trades = data.get('total_trades', 0)
    cumulative_pips = data.get('cumulative_pips', 0)

    # Choose emoji based on performance
    if lifetime_exp > 0.5:
        emoji = "🏆"
    elif lifetime_exp > 0.25:
        emoji = "✅"
    elif lifetime_exp > 0:
        emoji = "⚠️"
    else:
        emoji = "🔴"

    return (f"{emoji} Exp: {lifetime_exp:+.3f}R | "
            f"Trades: {total_trades:,} | "
            f"P&L: {cumulative_pips:+.1f} pips")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--mini':
        # Just print mini summary
        print(generate_mini_summary())
    elif len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Display once
        display_expectancy_terminal(continuous=False)
    else:
        # Continuous monitoring
        try:
            display_expectancy_terminal(continuous=True)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")