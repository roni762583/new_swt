#!/usr/bin/env python3
"""
One-shot display of current training stats.
"""

import subprocess
from datetime import datetime

# Get last 100 lines of logs
result = subprocess.run(
    ["docker", "logs", "micro_training", "--tail", "100"],
    capture_output=True,
    text=True,
    timeout=2
)
logs = result.stdout + result.stderr

# Parse latest episode
metrics = {}
for line in reversed(logs.split('\n')):
    if 'Episode' in line and '|' in line and 'Exp:' in line:
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

# Display
print("=" * 60)
print(" " * 20 + "MUZERO TRAINING MONITOR")
print("=" * 60)
print()

if metrics:
    ep = int(metrics.get('episode', 0))
    progress = (ep / 1000000) * 100

    print(f"📊 Episode: {ep:,} / 1,000,000 ({progress:.3f}%)")
    print(f"⚡ Speed: {metrics.get('eps', '?')} eps/sec")
    print()

    print(f"💰 Expectancy: {metrics.get('exp', '?')} pips")
    print(f"🎯 Win Rate: {metrics.get('wr', '?')}")
    print(f"📈 Trade Ratio: {metrics.get('tr', '?')}")
    print(f"📉 Loss: {metrics.get('loss', '?')}")
    print()

    print(f"🎮 Actions: {metrics.get('actions', 'N/A')}")
    print()

    # Progress bar
    bar_len = 40
    filled = int(bar_len * progress / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"Progress: [{bar}] {progress:.3f}%")
    print()

    # ETA calculation
    try:
        eps = float(metrics.get('eps', 0))
        if eps > 0:
            remaining = 1000000 - ep
            eta_hours = remaining / (eps * 3600)
            eta_days = eta_hours / 24
            print(f"⏰ ETA: {eta_days:.1f} days ({eta_hours:.0f} hours)")
    except:
        pass
else:
    print("No data available yet...")

print()
print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
print("-" * 60)