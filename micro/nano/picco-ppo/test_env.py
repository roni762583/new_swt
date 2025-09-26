#!/usr/bin/env python3
"""Quick test of the trading environment."""

import sys
import os
import numpy as np

print("Testing PPO Trading Environment")
print("=" * 60)

# Test imports
try:
    import gymnasium as gym
    print("✅ Gymnasium imported")
except ImportError as e:
    print(f"❌ Failed to import gymnasium: {e}")
    sys.exit(1)

# Test environment
try:
    from env.trading_env import TradingEnv
    print("✅ TradingEnv imported")

    # Create environment
    env = TradingEnv(
        data_path="../../../data/master.duckdb",
        initial_balance=0.0,  # Track pips only
        transaction_cost=4.0,  # 4 pip spread
        max_episode_steps=100,
        reward_scaling=0.01
    )
    print("✅ Environment created")

    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful: obs shape = {obs.shape}")
    print(f"   Features: 7 market + 6 position + 4 time = 17 total")

    # Test a few steps
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"✅ Ran 10 steps: total reward = {total_reward:.4f}")

    # Test PPO imports (if available)
    try:
        import torch
        import stable_baselines3
        print("✅ PyTorch and Stable-Baselines3 available")
        print("   Ready for neural network PPO training!")
    except ImportError:
        print("⚠️  PyTorch/SB3 not installed - only minimal training available")

    print("\n🎯 Environment test PASSED!")
    print("   AMDDP1 reward: pnl_pips - 0.01 * cumulative_drawdown")
    print("   4 pip spread cost on position opening")
    print("   17 features with cyclic time encoding")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)