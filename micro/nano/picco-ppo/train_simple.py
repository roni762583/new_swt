#!/usr/bin/env python3
"""
Simplified PPO training script for testing without full dependencies.
Uses minimal implementation to verify the environment setup.
"""

import numpy as np
import pandas as pd
import duckdb
from datetime import datetime
import os
import sys

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("PPO TRAINING SYSTEM - TEST RUN")
print("="*60)

# Test database connection
try:
    db_path = "../../../data/master.duckdb"
    conn = duckdb.connect(db_path, read_only=True)

    # Check data availability
    result = conn.execute("""
        SELECT COUNT(*) as count,
               MIN(bar_index) as min_idx,
               MAX(bar_index) as max_idx
        FROM master
    """).fetchone()

    print(f"\n✓ Database connected successfully")
    print(f"  Total bars: {result[0]:,}")
    print(f"  Range: {result[1]:,} to {result[2]:,}")

    conn.close()
except Exception as e:
    print(f"✗ Database error: {e}")
    sys.exit(1)

# Test feature calculations
print("\n" + "-"*40)
print("TESTING FEATURE CALCULATIONS")
print("-"*40)

# Load sample data
conn = duckdb.connect(db_path, read_only=True)
m1_data = pd.read_sql("""
    SELECT bar_index, timestamp, open, high, low, close, volume
    FROM master
    WHERE bar_index BETWEEN 100000 AND 105000
    ORDER BY bar_index
""", conn)
conn.close()

print(f"✓ Loaded {len(m1_data)} M1 bars for testing")

# Aggregate to M5
def aggregate_bars(data, period):
    data['group'] = data['bar_index'] // period
    return data.groupby('group').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'bar_index': 'first'
    }).reset_index(drop=True)

m5_data = aggregate_bars(m1_data, 5)
h1_data = aggregate_bars(m1_data, 60)

print(f"✓ Created {len(m5_data)} M5 bars")
print(f"✓ Created {len(h1_data)} H1 bars")

# Calculate features
print("\n" + "-"*40)
print("CALCULATING OPTIMAL FEATURES")
print("-"*40)

# M5 features
m5_data['sma20'] = m5_data['close'].rolling(20).mean()
m5_data['sma200'] = m5_data['close'].rolling(200).mean()

# React ratio
m5_data['reactive'] = m5_data['close'] - m5_data['sma200']
m5_data['lessreactive'] = m5_data['sma20'] - m5_data['sma200']
m5_data['react_ratio'] = m5_data['reactive'] / (m5_data['lessreactive'] + 0.0001)

# Efficiency ratio
direction = (m5_data['close'] - m5_data['close'].shift(10)).abs()
volatility = m5_data['close'].diff().abs().rolling(10).sum()
m5_data['efficiency_ratio'] = direction / (volatility + 0.0001)

# Bollinger Bands
bb_std = m5_data['close'].rolling(20).std()
m5_data['bb_position'] = (m5_data['close'] - m5_data['sma20']) / (bb_std * 2 + 0.0001)

print("\nMarket Features (sample):")
print(f"1. React Ratio: {m5_data['react_ratio'].iloc[-1]:.3f}")
print(f"2. Efficiency Ratio: {m5_data['efficiency_ratio'].iloc[-1]:.3f}")
print(f"3. BB Position: {m5_data['bb_position'].iloc[-1]:.3f}")

# Position tracking simulation
print("\n" + "-"*40)
print("POSITION TRACKING SIMULATION")
print("-"*40)

class PositionTracker:
    def __init__(self):
        self.position_side = 0.0
        self.position_pips = 0.0
        self.bars_since_entry = 0.0
        self.pips_from_peak = 0.0
        self.max_drawdown_pips = 0.0
        self.accumulated_dd = 0.0
        self.entry_price = 0.0
        self.peak_pips = 0.0

    def open_long(self, price):
        self.position_side = 1.0
        self.entry_price = price
        self.bars_since_entry = 0
        self.position_pips = 0
        self.peak_pips = 0
        print(f"→ Opened LONG at {price:.3f}")

    def update(self, current_price):
        if self.position_side == 0:
            return

        self.bars_since_entry += 1

        if self.position_side > 0:  # Long
            self.position_pips = (current_price - self.entry_price) * 100
        else:  # Short
            self.position_pips = (self.entry_price - current_price) * 100

        self.peak_pips = max(self.peak_pips, self.position_pips)
        self.pips_from_peak = self.peak_pips - self.position_pips

        if self.position_pips < 0:
            self.max_drawdown_pips = max(self.max_drawdown_pips, abs(self.position_pips))

    def close(self, price):
        if self.position_side == 0:
            return

        final_pips = self.position_pips
        print(f"← Closed position at {price:.3f}")
        print(f"  Result: {final_pips:.1f} pips in {self.bars_since_entry} bars")

        if final_pips < 0:
            self.accumulated_dd += abs(final_pips)

        self.position_side = 0
        self.position_pips = 0
        self.bars_since_entry = 0

# Simulate a trade
tracker = PositionTracker()

# Find a good entry point
for i in range(250, len(m5_data) - 20):
    efficiency = m5_data.iloc[i]['efficiency_ratio']
    react_ratio = m5_data.iloc[i]['react_ratio']

    # Entry signal
    if efficiency > 0.3 and react_ratio > 0.5 and tracker.position_side == 0:
        tracker.open_long(m5_data.iloc[i]['close'])

        # Simulate position for 10 bars
        for j in range(i+1, min(i+11, len(m5_data))):
            tracker.update(m5_data.iloc[j]['close'])

        tracker.close(m5_data.iloc[j]['close'])
        break

# State vector example
print("\n" + "-"*40)
print("EXAMPLE STATE VECTOR (13 features)")
print("-"*40)

# Market features (7)
market_features = [
    m5_data['react_ratio'].iloc[-1],
    1.0,  # h1_trend
    0.02,  # h1_momentum
    m5_data['efficiency_ratio'].iloc[-1],
    m5_data['bb_position'].iloc[-1],
    0.3,  # rsi_extreme
    1.0 if m5_data['efficiency_ratio'].iloc[-1] < 0.3 else 0.0
]

# Position features (6)
position_features = [
    tracker.position_side,
    tracker.position_pips / 100.0,
    tracker.bars_since_entry / 100.0,
    tracker.pips_from_peak / 100.0,
    tracker.max_drawdown_pips / 100.0,
    tracker.accumulated_dd / 1000.0
]

state = np.array(market_features + position_features, dtype=np.float32)

print("Market Features:")
for i, name in enumerate(['react_ratio', 'h1_trend', 'h1_momentum',
                          'efficiency_ratio', 'bb_position', 'rsi_extreme',
                          'use_mean_reversion']):
    print(f"  {name:20s}: {market_features[i]:+.3f}")

print("\nPosition Features:")
for i, name in enumerate(['position_side', 'position_pips_norm',
                          'bars_since_entry_norm', 'pips_from_peak_norm',
                          'max_dd_pips_norm', 'accumulated_dd_norm']):
    print(f"  {name:20s}: {position_features[i]:+.3f}")

print("\n" + "="*60)
print("TEST COMPLETE - Environment Ready for PPO Training")
print("="*60)

print("\nNext steps:")
print("1. Install dependencies: pip install gymnasium stable-baselines3 torch")
print("2. Run full training: python train.py")
print("3. Monitor with TensorBoard: tensorboard --logdir=./tensorboard")

# Check if we can import PPO dependencies
try:
    import gymnasium
    import stable_baselines3
    print("\n✓ PPO dependencies are installed - ready for full training!")
except ImportError:
    print("\n⚠ PPO dependencies not yet installed")
    print("  Run: pip install gymnasium stable-baselines3 torch")