#!/usr/bin/env python3
"""
Simple terminal dashboard for MuZero training.
"""

import subprocess
import time
import os
import re
from datetime import datetime

def get_logs(lines=2000):
    """Get recent container logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "micro_training", "--tail", str(lines)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3
        )
        return result.stdout
    except:
        return ""

def parse_latest_episode(logs):
    """Parse the latest episode line."""
    # Match episode lines with INFO: prefix
    pattern = r'Episode (\d+) \| Steps (\d+) \| EPS: ([\d.]+) \| Loss: ([\d.]+) \| TD: ([\d.]+) \| Exp: ([-\d.]+) \| WR: ([\d.]+)% \| TradeRatio: ([\d.]+)%'
    matches = re.findall(pattern, logs)

    if matches:
        latest = matches[-1]
        return {
            'episode': int(latest[0]),
            'steps': int(latest[1]),
            'eps': float(latest[2]),
            'loss': float(latest[3]),
            'expectancy': float(latest[5]),
            'win_rate': float(latest[6]),
            'trade_ratio': float(latest[7])
        }
    return None

def display_dashboard(data):
    """Display the dashboard."""
    os.system('clear')

    print("=" * 60)
    print(" " * 18 + "MUZERO TRAINING MONITOR")
    print("=" * 60)
    print()

    if not data:
        print("Waiting for training data...")
        print()
        print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        return

    # Progress
    progress = (data['episode'] / 1000000) * 100
    print(f"📊 PROGRESS")
    print(f"  Episode:     {data['episode']:,} / 1,000,000 ({progress:.3f}%)")
    print(f"  Steps:       {data['steps']:,}")
    print(f"  Speed:       {data['eps']:.1f} episodes/sec")

    # ETA
    if data['eps'] > 0:
        remaining = 1000000 - data['episode']
        eta_hours = remaining / (data['eps'] * 3600)
        print(f"  ETA:         {eta_hours/24:.1f} days ({eta_hours:.0f} hours)")

    print()

    # Performance
    print(f"💰 PERFORMANCE")
    print(f"  Expectancy:  {data['expectancy']:.2f} pips/trade")
    print(f"  Win Rate:    {data['win_rate']:.1f}%")
    print(f"  Trade Ratio: {data['trade_ratio']:.1f}%")
    print(f"  Loss:        {data['loss']:.2f}")

    print()

    # Progress bar
    bar_len = 50
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"[{bar}] {progress:.2f}%")

    print()
    print(f"Updated: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to exit")
    print("-" * 60)

def main():
    """Main loop."""
    print("Starting dashboard... Press Ctrl+C to exit")

    while True:
        try:
            logs = get_logs(1000)
            data = parse_latest_episode(logs)
            display_dashboard(data)
            time.sleep(3)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()