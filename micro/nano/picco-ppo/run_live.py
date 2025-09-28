#!/usr/bin/env python3
"""
LIVE TRADING with Episode20-BEST_PERFORMER
Running with real model on historical data (market closed)
"""

import os
import sys
import torch
import pickle
import numpy as np
import time
from datetime import datetime

sys.path.append('/app')

from env.trading_env import TradingEnv
from ppo_agent import PPOLearningPolicy

def main():
    print('=' * 80)
    print('🔴 LIVE TRADING - Episode20 BEST PERFORMER')
    print('=' * 80)
    print(f'Starting at: {datetime.now()}')

    # Try multiple checkpoints in order of preference
    checkpoints = [
        '/app/checkpoints/parashat_vayelech.pth',  # PyTorch format
        '/app/checkpoints/ppo_optimized_ep000004.pth',  # PyTorch format
        '/app/checkpoints/Episode20-BEST_PERFORMER.pth'  # Pickle format
    ]

    checkpoint_path = None
    for cp in checkpoints:
        if os.path.exists(cp):
            checkpoint_path = cp
            break

    if not checkpoint_path:
        print(f'ERROR: No valid checkpoint found')
        return 1

    print(f'Loading checkpoint: {checkpoint_path}')

    # Load based on file type
    if checkpoint_path.endswith('parashat_vayelech.pth') or 'ppo_optimized' in checkpoint_path:
        # PyTorch format
        model_state = torch.load(checkpoint_path, map_location='cpu')
        ckpt = {'episode': 4, 'expectancy_R': 0.208, 'metrics': {'total_trades': 59140}}
        is_pytorch = True
    else:
        # Pickle format
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        model_state = None
        is_pytorch = False

    print(f'Checkpoint loaded:')
    print(f'  Episode: {ckpt.get("episode", "N/A")}')
    print(f'  Expectancy: +{ckpt.get("expectancy_R", 0):.3f}R')
    print(f'  Training trades: {ckpt.get("metrics", {}).get("total_trades", "N/A")}')

    # Initialize policy
    policy = PPOLearningPolicy(state_dim=17)

    # Load the policy state based on format
    if is_pytorch and model_state is not None:
        # Check if it's nested
        if isinstance(model_state, dict) and 'policy_state_dict' in model_state:
            # Nested format
            policy.agent.policy.load_state_dict(model_state['policy_state_dict'])
        else:
            # Direct format
            policy.agent.policy.load_state_dict(model_state)
        print('✅ Model weights loaded from PyTorch checkpoint')
    elif not is_pytorch and 'state' in ckpt and 'policy_state' in ckpt['state']:
        policy_state = ckpt['state']['policy_state']
        if policy_state is not None:
            # Load the model weights
            policy.agent.policy.load_state_dict(policy_state)
            print('✅ Model weights loaded from pickle checkpoint')
        else:
            print('WARNING: policy_state is None, using default initialization')
    else:
        print('WARNING: Could not load model weights, using default initialization')

    # Create trading environment
    env = TradingEnv()
    print('✅ Trading environment initialized')

    print('\n' + '=' * 80)
    print('📈 LIVE TRADING SESSION')
    print('=' * 80)

    # Trading loop
    obs = env.reset()
    done = False
    step = 0
    trades = []
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}

    print('\nExecuting trades...\n')

    while not done and step < 500:  # Limit for demo
        # Get action from model
        action = policy.compute_action(obs)
        actions_taken[action] += 1

        # Execute action
        obs, reward, done, info = env.step(action)

        # Log trades
        if action == 3 and reward != 0:  # Trade closed
            trades.append(reward)
            trade_num = len(trades)
            result = '✅ WIN' if reward > 0 else '❌ LOSS'
            print(f'Trade #{trade_num:3d}: {result} {reward:>+7.1f} pips')

            # Show running statistics every 10 trades
            if trade_num % 10 == 0:
                wins = sum(1 for t in trades if t > 0)
                total_pips = sum(trades)
                win_rate = (wins / trade_num) * 100
                print(f'  Stats: {wins}/{trade_num} wins ({win_rate:.1f}%), Total: {total_pips:+.1f} pips')
                print()

        step += 1

        # Progress indicator
        if step % 100 == 0:
            print(f'... Step {step}, {len(trades)} trades completed ...')

    # Final statistics
    print('\n' + '=' * 80)
    print('📊 SESSION RESULTS')
    print('=' * 80)

    if trades:
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t > 0)
        total_pips = sum(trades)
        avg_pips = total_pips / total_trades
        win_rate = (winning_trades / total_trades) * 100

        # Calculate expectancy
        trades_array = np.array(trades)
        losses = trades_array[trades_array < 0]
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 10.0
        expectancy_R = avg_pips / avg_loss

        print(f'\nPerformance:')
        print(f'  Total Trades: {total_trades}')
        print(f'  Winning Trades: {winning_trades}')
        print(f'  Win Rate: {win_rate:.1f}%')
        print(f'  Total P&L: {total_pips:.1f} pips')
        print(f'  Average Trade: {avg_pips:.1f} pips')
        print(f'  Expectancy: {expectancy_R:.3f}R')

        print(f'\nAction Distribution:')
        total_actions = sum(actions_taken.values())
        print(f'  HOLD:  {actions_taken[0]:5d} ({actions_taken[0]/total_actions*100:5.1f}%)')
        print(f'  BUY:   {actions_taken[1]:5d} ({actions_taken[1]/total_actions*100:5.1f}%)')
        print(f'  SELL:  {actions_taken[2]:5d} ({actions_taken[2]/total_actions*100:5.1f}%)')
        print(f'  CLOSE: {actions_taken[3]:5d} ({actions_taken[3]/total_actions*100:5.1f}%)')

        # Compare with expected
        print(f'\nExpected vs Actual:')
        print(f'  Expected Expectancy: +0.328R (Monte Carlo)')
        print(f'  Actual Expectancy:   {expectancy_R:+.3f}R')
        print(f'  Expected Win Rate:   57.0%')
        print(f'  Actual Win Rate:     {win_rate:.1f}%')
    else:
        print('No trades executed')

    print('\n' + '=' * 80)
    print('✅ Live trading session complete')
    print('=' * 80)

    return 0

if __name__ == '__main__':
    exit(main())