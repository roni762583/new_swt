#!/usr/bin/env python3
"""
Validation script for the minimal PPO implementation.
Tests on validation and test sets separately.
"""

import numpy as np
import pandas as pd
import duckdb
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_minimal import MinimalTradingEnv, SimplePolicy


def validate(data_split="validation"):
    """Run validation on specified data split"""
    print("\n" + "="*60)
    print(f"VALIDATION ON {data_split.upper()} SET")
    print("="*60)

    # Determine data range (from 1M total bars)
    if data_split == "validation":
        start_idx = 700000
        end_idx = 1000000  # 300k bars
    elif data_split == "test":
        start_idx = 1000000
        end_idx = 1100000  # 100k bars
    else:
        raise ValueError("Split must be 'validation' or 'test'")

    # Check for Docker or local path
    db_path = "../../../data/master.duckdb"
    if os.path.exists("/app/data/master.duckdb"):
        db_path = "/app/data/master.duckdb"

    print(f"\n📊 Data Range: bars {start_idx:,} to {end_idx:,}")

    # Create environment
    env = MinimalTradingEnv(
        db_path=db_path,
        start_idx=start_idx,
        end_idx=end_idx
    )

    # Create policy (would load trained model in full implementation)
    policy = SimplePolicy()

    # Run evaluation
    state = env.reset()
    episode_reward = 0
    steps = 0

    while True:
        action = policy.get_action(state)
        next_state, reward, done = env.step(action)

        episode_reward += reward
        steps += 1

        if steps % 500 == 0:
            print(f"  Step {steps}: Equity=${env.equity:.2f}, Trades={len(env.trades)}")

        state = next_state

        if done:
            break

    # Calculate metrics
    total_return = (env.equity - 10000) / 10000 * 100

    print(f"\n{data_split.upper()} RESULTS:")
    print(f"  Final Equity: ${env.equity:.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Total Trades: {len(env.trades)}")

    if env.trades:
        winning_trades = [t for t in env.trades if t['pips'] > 0]
        win_rate = len(winning_trades) / len(env.trades) * 100
        total_pips = sum(t['pips'] for t in env.trades)
        avg_pips = total_pips / len(env.trades)

        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total Pips: {total_pips:.1f}")
        print(f"  Avg Pips/Trade: {avg_pips:.1f}")

        # Calculate expectancy_R (Van Tharp method)
        trades_pnl = [t['pips'] for t in env.trades]
        trades_array = np.array(trades_pnl)
        losses = trades_array[trades_array < 0]

        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 10.0  # R value
        expectancy_pips = avg_pips
        expectancy_R = expectancy_pips / avg_loss

        print(f"  Expectancy: {expectancy_pips:.1f} pips = {expectancy_R:.3f}R (R={avg_loss:.1f})")

        # Calculate Sharpe ratio (simplified)
        returns = [t['pips'] for t in env.trades]
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 0.0001)
            print(f"  Sharpe Ratio: {sharpe:.2f}")

        # System Quality Assessment
        if expectancy_R > 0.5:
            quality = "EXCELLENT 🏆"
        elif expectancy_R > 0.25:
            quality = "GOOD ✅"
        elif expectancy_R > 0:
            quality = "ACCEPTABLE ⚠️"
        else:
            quality = "NEEDS IMPROVEMENT 🔴"

        print(f"  System Quality: {quality}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'split': data_split,
        'return': total_return,
        'trades': len(env.trades),
        'final_equity': env.equity,
        'win_rate': win_rate if env.trades else 0,
        'total_pips': total_pips if env.trades else 0,
        'expectancy_pips': expectancy_pips if env.trades else 0,
        'expectancy_R': expectancy_R if env.trades else 0,
        'avg_loss_R': avg_loss if env.trades else 10.0
    }

    os.makedirs("results", exist_ok=True)
    results_file = f"results/validation_{data_split}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Results saved to {results_file}")

    return results


if __name__ == "__main__":
    print("\n🔬 RUNNING COMPREHENSIVE VALIDATION")
    print("=====================================")

    # Validate on both sets
    val_results = validate("validation")
    test_results = validate("test")

    print("\n" + "="*60)
    print("📊 SUMMARY ACROSS ALL SPLITS")
    print("="*60)
    print(f"Validation Return: {val_results['return']:.2f}%")
    print(f"Test Return: {test_results['return']:.2f}%")
    print(f"\n🎯 PERFORMANCE RATING (Van Tharp):")
    print(f"Validation: {val_results['expectancy_pips']:.1f} pips = {val_results['expectancy_R']:.3f}R")
    print(f"Test:       {test_results['expectancy_pips']:.1f} pips = {test_results['expectancy_R']:.3f}R")

    # Overall quality assessment
    avg_expectancy_R = (val_results['expectancy_R'] + test_results['expectancy_R']) / 2
    if avg_expectancy_R > 0.5:
        print(f"\n🏆 OVERALL: EXCELLENT SYSTEM (Avg {avg_expectancy_R:.3f}R)")
    elif avg_expectancy_R > 0.25:
        print(f"\n✅ OVERALL: GOOD SYSTEM (Avg {avg_expectancy_R:.3f}R)")
    elif avg_expectancy_R > 0:
        print(f"\n⚠️  OVERALL: ACCEPTABLE SYSTEM (Avg {avg_expectancy_R:.3f}R)")
    else:
        print(f"\n🔴 OVERALL: NEEDS IMPROVEMENT (Avg {avg_expectancy_R:.3f}R)")
    print("\n✅ Validation complete!")
    print("\nNote: In production PPO, the policy would be:")
    print("1. Trained on training set (60%)")
    print("2. Hyperparameters tuned on validation set (30%)")
    print("3. Final evaluation on test set (10%)")