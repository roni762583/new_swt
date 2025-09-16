#!/usr/bin/env python3
"""
Simple training monitor showing latest episodes and validation status
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

def read_latest_episodes(checkpoint_dir):
    """Read recent episode JSON files"""
    checkpoint_dir = Path(checkpoint_dir)
    episodes = {}

    for json_file in sorted(checkpoint_dir.glob("swt_episode_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                ep_num = data.get('episode', 0)
                if 'performance' in data and not isinstance(data['performance'].get('error'), str):
                    episodes[ep_num] = data['performance']
        except:
            pass

    return episodes

def display_status():
    """Display training and validation status"""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("=" * 80)
    print("SWT TRAINING & VALIDATION STATUS".center(80))
    print("=" * 80)
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Read episodes
    episodes = read_latest_episodes('checkpoints')

    if episodes:
        # Show last 5 episodes
        print("RECENT TRAINING EPISODES")
        print("-" * 40)

        recent = sorted(episodes.items(), key=lambda x: x[0])[-5:]
        for ep_num, ep_data in recent:
            sqn = ep_data.get('sqn', 0)
            trades = ep_data.get('trades', 0)
            pnl = ep_data.get('total_pnl', 0)
            win_rate = ep_data.get('win_rate', 0) * 100
            print(f"Episode {ep_num:3d}: SQN={sqn:5.2f} | Trades={trades:3d} | PnL={pnl:7.1f} | Win={win_rate:4.1f}%")

        # Show best episode
        print()
        print("BEST EPISODE")
        print("-" * 40)
        best_sqn = 0
        best_ep = None
        for ep_num, ep_data in episodes.items():
            if ep_data.get('sqn', 0) > best_sqn:
                best_sqn = ep_data['sqn']
                best_ep = (ep_num, ep_data)

        if best_ep:
            ep_num, ep_data = best_ep
            print(f"Episode {ep_num}: SQN={ep_data['sqn']:.2f} ({ep_data.get('sqn_classification', 'Unknown')})")
            print(f"  Trades: {ep_data.get('trades', 0)}")
            print(f"  Total PnL: {ep_data.get('total_pnl', 0):.1f} pips")
            print(f"  Win Rate: {ep_data.get('win_rate', 0)*100:.1f}%")

    # Check for validation results
    print()
    print("VALIDATION STATUS")
    print("-" * 40)

    val_dir = Path('validation_results')
    if val_dir.exists():
        val_files = list(val_dir.glob("validation_*.json"))
        if val_files:
            latest = max(val_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest, 'r') as f:
                    val_data = json.load(f)
                print(f"Last validation: {latest.name}")
                print(f"  Checkpoint: {val_data.get('checkpoint', 'Unknown')}")
                print(f"  Win Rate: {val_data.get('win_rate', 0):.1f}%")
                print(f"  Sharpe Ratio: {val_data.get('sharpe_ratio', 0):.3f}")
            except:
                print("No validation results available")
        else:
            print("No validation results yet")
    else:
        print("Validation directory not found")

    print()
    print("-" * 80)
    print("Press Ctrl+C to exit")

def main():
    """Main monitoring loop"""
    print("Starting simple monitor...")

    while True:
        try:
            display_status()
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()