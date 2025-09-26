#!/usr/bin/env python3
"""
Demo PPO Training Script with Rolling Expectancy.
Shows the complete training flow with all improvements.
"""

import numpy as np
from datetime import datetime
from rolling_expectancy import RollingExpectancyTracker

print("=" * 60)
print("🚀 PPO TRAINING DEMONSTRATION")
print("=" * 60)

# Configuration
print("\n📋 Configuration:")
print("- Environment: 17 features (7 market + 6 position + 4 time)")
print("- Reward: AMDDP1 = pnl_pips - 0.01 * cumulative_DD")
print("- Trading Cost: 4 pip spread on opening")
print("- Network: [256, 256] MLP with ReLU")
print("- Training Data: 600k bars (60% of dataset)")
print("- PPO Hyperparameters: γ=0.99, λ=0.95, clip=0.2")

# Initialize rolling expectancy tracker
tracker = RollingExpectancyTracker(window_sizes=[100, 500, 1000])

# Simulate training session
print("\n🎯 Starting Training Session...")
print("-" * 40)

np.random.seed(42)
cumulative_pips = 0.0

# Simulate 3 phases of learning
phases = [
    ("Learning Phase", 100, -1.0, 5.0),      # Poor initial performance
    ("Improvement Phase", 200, 2.0, 4.0),    # Getting better
    ("Optimization Phase", 300, 4.5, 3.0),   # Good performance
]

episode = 0
for phase_name, num_episodes, mean_pips, std_pips in phases:
    print(f"\n📚 {phase_name}:")

    for i in range(num_episodes):
        episode += 1

        # Simulate episode with multiple trades
        episode_trades = []
        for _ in range(np.random.randint(5, 15)):
            # Simulate trade with spread cost
            trade_pips = np.random.normal(mean_pips, std_pips) - 4.0  # 4 pip spread
            episode_trades.append(trade_pips)
            tracker.add_trade(trade_pips)

        episode_pips = sum(episode_trades)
        cumulative_pips += episode_pips

        # Print progress every 50 episodes
        if episode % 50 == 0:
            expectancies = tracker.calculate_expectancies()
            exp_100 = expectancies.get('expectancy_R_100', 0)
            exp_500 = expectancies.get('expectancy_R_500', 0)

            print(f"  Episode {episode:3d}: "
                  f"Cumulative: {cumulative_pips:+7.1f} pips | "
                  f"E_100: {exp_100:+.3f}R | "
                  f"E_500: {exp_500:+.3f}R")

# Final summary
print("\n" + "=" * 60)
print(tracker.get_summary())

# Performance analysis
print("\n📊 Training Results Analysis:")
print("-" * 40)

final_expectancies = tracker.calculate_expectancies()

# Check each window
for size in [100, 500, 1000]:
    exp_key = f'expectancy_R_{size}'
    if exp_key in final_expectancies:
        exp_R = final_expectancies[exp_key]
        quality, emoji = tracker.get_quality_assessment(exp_R)
        print(f"{size:4d}-trade window: {exp_R:+.3f}R {emoji} ({quality})")

# Lifetime performance
if 'lifetime_expectancy_R' in final_expectancies:
    lifetime_R = final_expectancies['lifetime_expectancy_R']
    quality, emoji = tracker.get_quality_assessment(lifetime_R)
    print(f"\nLifetime ({final_expectancies['lifetime_trades']} trades): "
          f"{lifetime_R:+.3f}R {emoji} ({quality})")

print("\n🏁 Training Complete!")
print(f"Total Pips: {cumulative_pips:+.1f}")
print(f"Total Trades: {len(tracker.all_trades)}")

# Key insights
print("\n💡 Key Insights:")
print("1. Rolling windows show performance evolution")
print("2. 100-trade window responds quickly to changes")
print("3. 1000-trade window shows stable long-term performance")
print("4. AMDDP1 reward encourages risk management")
print("5. 4 pip spread makes results realistic")

print("\n✅ Demo Complete!")