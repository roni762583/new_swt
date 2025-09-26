#!/usr/bin/env python3
"""
Monitor rolling expectancy during training.
Reads from expectancy log file and displays current status.
"""

import json
import os
import time
from datetime import datetime
import sys

def monitor_expectancy(log_file="results/rolling_expectancy.json", refresh_rate=5):
    """
    Monitor rolling expectancy from log file.

    Args:
        log_file: Path to expectancy log
        refresh_rate: Seconds between updates
    """
    print("=" * 60)
    print("📊 ROLLING EXPECTANCY MONITOR")
    print("=" * 60)
    print(f"Monitoring: {log_file}")
    print(f"Refresh rate: {refresh_rate}s")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    data = json.load(f)

                # Clear screen (works on Unix-like systems)
                os.system('clear' if os.name == 'posix' else 'cls')

                print("=" * 60)
                print(f"📊 ROLLING EXPECTANCY - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)

                # Display metadata
                print(f"\n📈 Training Progress:")
                print(f"  Episode: {data.get('episode', 'N/A')}")
                print(f"  Total Trades: {data.get('total_trades', 0):,}")
                print(f"  Cumulative P&L: {data.get('cumulative_pips', 0):+.1f} pips")

                # Display rolling expectancies
                print(f"\n📊 Rolling Windows:")
                for window_size in [100, 500, 1000]:
                    exp_key = f'expectancy_R_{window_size}'
                    if exp_key in data:
                        exp_R = data[exp_key]
                        sample = data.get(f'sample_size_{window_size}', 0)
                        win_rate = data.get(f'win_rate_{window_size}', 0)

                        # Quality assessment
                        if exp_R > 0.5:
                            quality = "🏆 EXCELLENT"
                        elif exp_R > 0.25:
                            quality = "✅ GOOD"
                        elif exp_R > 0:
                            quality = "⚠️ ACCEPTABLE"
                        else:
                            quality = "🔴 NEEDS IMPROVEMENT"

                        print(f"\n  {window_size}-Trade Window:")
                        print(f"    Expectancy: {exp_R:+.3f}R {quality}")
                        print(f"    Win Rate: {win_rate:.1f}%")
                        print(f"    Sample: {sample}/{window_size}")

                        # Progress bar
                        progress = min(100, (sample / window_size) * 100)
                        bar_length = 20
                        filled = int(bar_length * progress / 100)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"    Progress: [{bar}] {progress:.0f}%")

                # Lifetime stats
                if 'lifetime_expectancy_R' in data:
                    print(f"\n🎯 Lifetime Performance:")
                    print(f"  Expectancy: {data['lifetime_expectancy_R']:+.3f}R")
                    print(f"  Win Rate: {data.get('lifetime_win_rate', 0):.1f}%")
                    print(f"  Total Trades: {data.get('lifetime_trades', 0):,}")

                # Trend indicator
                if 'trend' in data:
                    trend = data['trend']
                    if trend > 0:
                        print(f"\n📈 Trend: IMPROVING (+{trend:.3f}R/100 trades)")
                    elif trend < 0:
                        print(f"\n📉 Trend: DECLINING ({trend:.3f}R/100 trades)")
                    else:
                        print(f"\n➡️ Trend: STABLE")

                print("\n" + "=" * 60)

            else:
                print(f"Waiting for {log_file} to be created...")

            time.sleep(refresh_rate)

        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Error reading log: {e}")
            time.sleep(refresh_rate)


def check_latest_expectancy(log_file="results/rolling_expectancy.json"):
    """Quick check of latest expectancy."""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)

        print("📊 Latest Rolling Expectancy:")
        print("-" * 40)

        for window_size in [100, 500, 1000]:
            exp_key = f'expectancy_R_{window_size}'
            if exp_key in data:
                exp_R = data[exp_key]
                quality = "🏆" if exp_R > 0.5 else "✅" if exp_R > 0.25 else "⚠️" if exp_R > 0 else "🔴"
                print(f"{window_size:4d}-trade: {exp_R:+.3f}R {quality}")

        if 'lifetime_expectancy_R' in data:
            print(f"\nLifetime: {data['lifetime_expectancy_R']:+.3f}R")
            print(f"Total trades: {data.get('lifetime_trades', 0):,}")
    else:
        print(f"No expectancy log found at {log_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_latest_expectancy()
    else:
        monitor_expectancy()