#!/usr/bin/env python3
"""
Ultra-simple training dashboard - just the essentials.
"""

import subprocess
import time
import os
from datetime import datetime

def get_metrics():
    """Get latest metrics from container."""
    try:
        # Get last 100 lines of logs
        result = subprocess.run(
            ["docker", "logs", "micro_training", "--tail", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2
        )
        logs = result.stdout

        # Parse latest episode
        metrics = {}
        for line in reversed(logs.split('\n')):
            if 'Episode' in line and '|' in line and 'Exp:' in line:
                # Parse the line
                parts = line.split('|')
                for part in parts:
                    if 'Episode' in part:
                        metrics['episode'] = part.split()[-1]
                    elif 'Exp:' in part:
                        metrics['exp'] = part.split()[-1]
                    elif 'WR:' in part:
                        metrics['wr'] = part.split()[-1]
                    elif 'TradeRatio:' in part:
                        metrics['tr'] = part.split()[-1]
                    elif 'Loss:' in part:
                        metrics['loss'] = part.split()[-1]
                    elif 'EPS:' in part:
                        metrics['eps'] = part.split()[-1]
                break

        # Get action distribution
        for line in reversed(logs.split('\n')):
            if 'Action distribution' in line:
                metrics['actions'] = line.split('-')[1].strip()
                break

        return metrics
    except:
        return {}

def main():
    """Run simple dashboard."""
    print("Starting dashboard... Press Ctrl+C to exit\n")

    while True:
        try:
            os.system('clear')

            # Header
            print("=" * 60)
            print(" " * 20 + "MUZERO TRAINING MONITOR")
            print("=" * 60)
            print()

            # Get metrics
            m = get_metrics()

            if m:
                # Progress
                ep = int(m.get('episode', 0))
                progress = (ep / 1000000) * 100
                print(f"📊 Episode: {ep:,} / 1,000,000 ({progress:.3f}%)")
                print(f"⚡ Speed: {m.get('eps', '?')} eps/sec")
                print()

                # Performance
                print(f"💰 Expectancy: {m.get('exp', '?')} pips")
                print(f"🎯 Win Rate: {m.get('wr', '?')}")
                print(f"📈 Trade Ratio: {m.get('tr', '?')}")
                print(f"📉 Loss: {m.get('loss', '?')}")
                print()

                # Actions
                print(f"🎮 Actions: {m.get('actions', 'N/A')}")
                print()

                # Progress bar
                bar_len = 40
                filled = int(bar_len * progress / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"Progress: [{bar}] {progress:.3f}%")
            else:
                print("Waiting for data...")

            print()
            print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 60)

            time.sleep(2)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            break

if __name__ == "__main__":
    main()