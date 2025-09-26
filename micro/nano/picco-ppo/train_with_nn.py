#!/usr/bin/env python3
"""
Train PPO with Neural Networks for M5/H1 Trading.
Uses the real PPO agent with optimized architecture.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.trading_env import TradingEnv
from ppo_agent import PPOLearningPolicy
from rolling_expectancy import RollingExpectancyTracker
from checkpoint_manager import CheckpointManager


def train_ppo(episodes=20, load_checkpoint=None):
    """
    Train PPO agent with neural networks.

    Args:
        episodes: Number of training episodes
        load_checkpoint: Path to checkpoint to resume from
    """
    print("\n" + "="*60)
    print("🧠 PPO NEURAL NETWORK TRAINING")
    print("="*60)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print("\n📊 DATA CONFIGURATION:")
    print("  Split: Training (60% of data)")
    print("  Episode length: 2880 steps (24 hours)")
    print("  Features: 17 (7 market + 6 position + 4 time)")
    print("  Actions: 4 (hold, buy, sell, close)")

    # Initialize components
    ckpt_manager = CheckpointManager("checkpoints")
    expectancy_tracker = RollingExpectancyTracker(window_sizes=[100, 500, 1000])

    # Load checkpoint if provided
    start_episode = 0
    if load_checkpoint:
        checkpoint = ckpt_manager.load_checkpoint(load_checkpoint)
        if checkpoint:
            start_episode = checkpoint['episode'] + 1
            print(f"\n✅ Resumed from episode {start_episode}")
            print(f"   Best expectancy: {checkpoint.get('expectancy_R', 0):.3f}R")

    # Initialize PPO agent
    policy = PPOLearningPolicy(state_dim=17, load_path=load_checkpoint)

    print("\n🏗️ NEURAL NETWORK ARCHITECTURE:")
    print("  Market Encoder: 7 → 64 → 32")
    print("  Position Encoder: 6 → 32 → 16")
    print("  Time Encoder: 4 → 16")
    print("  Shared Trunk: 64 → 128 → 128")
    print("  Policy Head: 128 → 64 → 4")
    print("  Value Head: 128 → 64 → 1")

    print("\n📈 TRAINING PARAMETERS:")
    print(f"  Learning Rate: 1e-4 (with decay)")
    print(f"  Batch Size: 32")
    print(f"  Update Frequency: 1024 steps")
    print(f"  PPO Epochs: 4")
    print(f"  Clip Ratio: 0.1")
    print(f"  Entropy Coefficient: 0.005")

    # Training metrics
    all_trades = []
    episode_rewards = []

    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)

    for episode in range(start_episode, episodes):
        print(f"\n📊 Episode {episode + 1}/{episodes}")
        print("-" * 40)

        # Create environment for this episode
        # Training data: bars 100,000 to 700,000
        env = TradingEnv(
            start_idx=100000,
            end_idx=700000,
            max_episode_steps=2880
        )

        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_trades = 0
        episode_pips = 0

        # Episode loop
        done = False
        step = 0

        while not done:
            # Get action from PPO policy
            action = policy.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition for learning
            policy.store_transition(state, action, reward, next_state, done)

            # Track metrics
            episode_reward += reward
            if 'trade_result' in info and info['trade_result'] != 0:
                episode_trades += 1
                episode_pips += info['trade_result']
                all_trades.append(info['trade_result'])

                # Update rolling expectancy
                expectancy_tracker.add_trade(info['trade_result'])

            # Update state
            state = next_state
            step += 1

            # Display progress every 500 steps
            if step % 500 == 0:
                print(f"   Step {step}: Reward={episode_reward:.2f}, Trades={episode_trades}, Pips={episode_pips:.1f}")

        # Episode summary
        episode_rewards.append(episode_reward)

        print(f"\n📈 Episode {episode + 1} Summary:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Total Trades: {episode_trades}")
        print(f"   Total Pips: {episode_pips:.1f}")
        print(f"   Avg Pips/Trade: {episode_pips/max(1, episode_trades):.2f}")

        # Get PPO stats
        ppo_stats = policy.get_stats()
        print(f"   PPO Updates: {ppo_stats['updates']}")
        print(f"   Total Steps Trained: {ppo_stats['total_steps']}")

        # Display rolling expectancy
        expectancy_data = expectancy_tracker.calculate_expectancies()
        print(f"\n📊 Rolling Expectancy:")

        for window in [100, 500, 1000]:
            key = f'expectancy_R_{window}'
            if key in expectancy_data and expectancy_data[f'sample_size_{window}'] > 0:
                exp_r = expectancy_data[key]
                sample = expectancy_data[f'sample_size_{window}']
                win_rate = expectancy_data.get(f'win_rate_{window}', 0)

                # Quality indicator
                if exp_r > 0.5:
                    quality = "🏆 EXCELLENT"
                elif exp_r > 0.25:
                    quality = "✅ GOOD"
                elif exp_r > 0:
                    quality = "⚠️ ACCEPTABLE"
                else:
                    quality = "🔴 POOR"

                print(f"   {window:4d}-trade: {exp_r:+.3f}R {quality} (WR: {win_rate:.1f}%, n={sample})")

        # Save checkpoint every 2 episodes
        if (episode + 1) % 2 == 0:
            print(f"\n💾 Saving checkpoint...")

            # Save PPO model
            model_path = f"checkpoints/ppo_nn_ep{episode:06d}.pth"
            policy.save(model_path)

            # Calculate lifetime expectancy
            if len(all_trades) > 0:
                lifetime_exp = expectancy_tracker.calculate_expectancy_r(all_trades)
            else:
                lifetime_exp = 0

            # Save checkpoint
            metrics = {
                'total_trades': len(all_trades),
                'avg_pips': np.mean(all_trades) if all_trades else 0,
                'sessions_completed': episode + 1
            }

            saved_path = ckpt_manager.save_checkpoint(
                episode=episode,
                expectancy_R=lifetime_exp,
                metrics=metrics,
                policy_state=model_path
            )

            if saved_path:
                print(f"   Saved: {saved_path}")

        # Save rolling expectancy
        expectancy_tracker.save_to_file()

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)

    # Final statistics
    if len(all_trades) > 0:
        final_exp = expectancy_tracker.calculate_expectancy_r(all_trades)
        print(f"\n🎯 Final Statistics:")
        print(f"   Total Trades: {len(all_trades)}")
        print(f"   Lifetime Expectancy: {final_exp:+.3f}R")
        print(f"   Average Episode Reward: {np.mean(episode_rewards):.2f}")
        print(f"   Best Episode Reward: {max(episode_rewards):.2f}")

        # Check for improvement
        if final_exp > 0.2:
            print(f"\n🎉 SUCCESS! Achieved positive expectancy with neural network learning!")
        else:
            print(f"\n⚠️ Training needs more episodes for better convergence.")

    return policy, expectancy_tracker


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO with Neural Networks")
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--load', type=str, default=None, help='Checkpoint to load')

    args = parser.parse_args()

    # Run training
    train_ppo(episodes=args.episodes, load_checkpoint=args.load)