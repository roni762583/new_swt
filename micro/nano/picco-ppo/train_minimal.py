#!/usr/bin/env python3
"""
Minimal PPO-style training implementation.
Demonstrates the trading logic without heavy dependencies.
"""

import numpy as np
import pandas as pd
import duckdb
import json
from datetime import datetime
import os
import sys
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from checkpoint_manager import CheckpointManager

class MinimalTradingEnv:
    """Lightweight trading environment"""

    def __init__(self, db_path="/app/data/master.duckdb", start_idx=100000, end_idx=1100000):
        """
        Initialize environment with data splits (1M bars total):
        - Training: bars 100,000 to 700,000 (60%)
        - Validation: bars 700,000 to 1,000,000 (30%)
        - Test: bars 1,000,000 to 1,100,000 (10%)
        """
        self.db_path = db_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.pip_multiplier = 100.0

        # Load data
        self._load_data()

        # State tracking
        self.reset()

        # AMDDP1 tracking
        self.dd_sum = 0.0  # Cumulative drawdown sum for AMDDP1

    def _load_data(self):
        """Load and prepare M5/H1 data"""
        print(f"📂 Loading data from {self.db_path}")
        print(f"   Range: bars {self.start_idx:,} to {self.end_idx:,}")

        conn = duckdb.connect(self.db_path, read_only=True)

        # Load M1 data
        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {self.start_idx} AND {self.end_idx}
        ORDER BY bar_index
        """

        print("   Executing query...")
        cursor = conn.execute(query)
        print("   Fetching data...")
        m1_data = cursor.df()
        print(f"   Loaded {len(m1_data):,} M1 bars")
        conn.close()

        # Aggregate to M5
        def aggregate(data, period):
            data['group'] = data['bar_index'] // period
            return data.groupby('group').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'bar_index': 'first'
            }).reset_index(drop=True)

        self.m5_data = aggregate(m1_data, 5)
        self.h1_data = aggregate(m1_data, 60)

        # Calculate features
        self._calculate_features()

        print(f"Loaded {len(self.m5_data)} M5 bars, {len(self.h1_data)} H1 bars")

    def _calculate_features(self):
        """Calculate optimal features"""
        df = self.m5_data

        # SMAs
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma200'] = df['close'].rolling(200).mean()

        # React ratio
        df['reactive'] = df['close'] - df['sma200']
        df['lessreactive'] = df['sma20'] - df['sma200']
        df['react_ratio'] = (df['reactive'] / (df['lessreactive'] + 0.0001)).clip(-5, 5)

        # Efficiency ratio
        direction = (df['close'] - df['close'].shift(10)).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = direction / (volatility + 0.0001)

        # Bollinger Bands
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = ((df['close'] - df['sma20']) / (bb_std * 2 + 0.0001)).clip(-1, 1)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_extreme'] = (df['rsi'] - 50) / 50

        # H1 features (simplified)
        h1 = self.h1_data
        h1['h1_sma20'] = h1['close'].rolling(20).mean()
        h1['h1_trend'] = 0
        h1.loc[h1['close'] > h1['h1_sma20'], 'h1_trend'] = 1
        h1.loc[h1['close'] < h1['h1_sma20'], 'h1_trend'] = -1
        h1['h1_momentum'] = h1['close'].pct_change(5)

        # Map H1 to M5
        df['h1_trend'] = 0.0
        df['h1_momentum'] = 0.0
        for i in range(len(df)):
            m5_bar = df.iloc[i]['bar_index']
            h1_idx = (h1['bar_index'] <= m5_bar).sum() - 1
            if 0 <= h1_idx < len(h1):
                df.loc[i, 'h1_trend'] = h1.iloc[h1_idx]['h1_trend']
                df.loc[i, 'h1_momentum'] = h1.iloc[h1_idx]['h1_momentum']

        # Regime flag
        df['use_mean_reversion'] = (df['efficiency_ratio'] < 0.3).astype(float)

    def reset(self):
        """Reset environment"""
        self.current_step = 250  # Start after warmup
        self.balance = 0.0  # Track pips only
        self.equity = 0.0  # Track pips only

        # Position tracking
        self.position_side = 0.0
        self.position_pips = 0.0
        self.bars_since_entry = 0.0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.pips_from_peak = 0.0
        self.max_drawdown_pips = 0.0
        self.accumulated_dd = 0.0
        self.peak_pips = 0.0

        self.trades = []
        self.trades_pnl = []  # For R-multiple calculation

        return self.get_state()

    def get_state(self):
        """Get current state vector"""
        current = self.m5_data.iloc[self.current_step]

        # Market features (7)
        market = np.array([
            current['react_ratio'],
            current['h1_trend'],
            current['h1_momentum'],
            current['efficiency_ratio'],
            current['bb_position'],
            current['rsi_extreme'],
            current['use_mean_reversion']
        ], dtype=np.float32)

        # Position features (6)
        position = np.array([
            self.position_side,
            self.position_pips / 100.0,
            self.bars_since_entry / 100.0,
            self.pips_from_peak / 100.0,
            self.max_drawdown_pips / 100.0,
            self.accumulated_dd / 1000.0
        ], dtype=np.float32)

        return np.concatenate([market, position])

    def step(self, action):
        """Execute action and return next state, reward"""
        current_price = self.m5_data.iloc[self.current_step]['close']

        # Execute action
        if action == 1 and self.position_side <= 0:  # Buy
            if self.position_side < 0:
                self._close_position(current_price)
            self._open_position(1, current_price)

        elif action == 2 and self.position_side >= 0:  # Sell
            if self.position_side > 0:
                self._close_position(current_price)
            self._open_position(-1, current_price)

        elif action == 3 and self.position_side != 0:  # Close
            self._close_position(current_price)

        # Update position tracking
        self._update_position(current_price)

        # Calculate reward using AMDDP1 if closing position
        if action == 3 and self.position_side != 0:
            # AMDDP1 reward for closing
            close_pips = self.position_pips - 0.2  # Subtract transaction cost
            reward = self._calculate_amddp1(close_pips, self.dd_sum)
        else:
            # Small penalty for holding or neutral for other actions
            reward = -0.001 if self.position_side != 0 else 0.0

        # Update equity for display
        self.equity = self.balance + self.position_pips

        # Move to next step
        self.current_step += 1

        # Check if done
        done = self.current_step >= len(self.m5_data) - 1

        return self.get_state(), reward, done

    def _open_position(self, side, price):
        """Open new position"""
        self.position_side = side
        self.entry_price = price
        self.entry_bar = self.current_step
        self.bars_since_entry = 0
        self.position_pips = 0
        self.peak_pips = 0
        self.balance -= 0.2  # Transaction cost

    def _close_position(self, price):
        """Close current position"""
        if self.position_side == 0:
            return

        # Calculate P&L
        if self.position_side > 0:
            pips = (price - self.entry_price) * self.pip_multiplier
        else:
            pips = (self.entry_price - price) * self.pip_multiplier

        # Apply AMDDP1 to balance (tracking pips only)
        self.balance += pips - 0.2  # Subtract transaction cost
        self.equity = self.balance  # In pips

        # Track trade
        self.trades.append({
            'entry': self.entry_price,
            'exit': price,
            'pips': pips,
            'bars': self.bars_since_entry,
            'side': 'long' if self.position_side > 0 else 'short'
        })
        self.trades_pnl.append(pips)  # Store for R calculation

        # Update accumulated DD
        if pips < 0:
            self.accumulated_dd += abs(pips)

        # Reset position
        self.position_side = 0
        self.position_pips = 0
        self.bars_since_entry = 0
        self.peak_pips = 0
        self.dd_sum = 0.0  # Reset AMDDP1 tracking

    def _update_position(self, current_price):
        """Update position tracking"""
        if self.position_side != 0:
            # Calculate current P&L
            prev_pips = self.position_pips

            if self.position_side > 0:
                self.position_pips = (current_price - self.entry_price) * self.pip_multiplier
            else:
                self.position_pips = (self.entry_price - current_price) * self.pip_multiplier

            self.bars_since_entry += 1
            self.peak_pips = max(self.peak_pips, self.position_pips)
            self.pips_from_peak = self.peak_pips - self.position_pips

            # Track drawdown for AMDDP1
            if self.pips_from_peak > 0:
                # Accumulate drawdown increases only
                dd_increase = self.pips_from_peak - (self.peak_pips - prev_pips if prev_pips < self.peak_pips else 0)
                if dd_increase > 0:
                    self.dd_sum += dd_increase

            if self.position_pips < 0:
                self.max_drawdown_pips = max(self.max_drawdown_pips, abs(self.position_pips))

    def _calculate_amddp1(self, pnl_pips, dd_sum):
        """
        Calculate AMDDP1 reward (Asymmetric Mean Drawdown Duration Penalty).

        Formula: reward = pnl_pips - 0.01 * cumulative_drawdown_sum

        With profit protection: If profitable trade has negative reward due to DD,
        return small positive reward instead.
        """
        # Base AMDDP1 formula with 1% penalty
        base_reward = pnl_pips - 0.01 * dd_sum

        # Apply profit protection
        if pnl_pips > 0 and base_reward < 0:
            return 0.001  # Small positive reward for profitable trades
        else:
            return base_reward

    def _calculate_market_outcome(self, current_price, next_price):
        """
        Quantize market movement into 3 buckets using 0.33σ threshold.
        Returns: 0=UP, 1=NEUTRAL, 2=DOWN

        This gives ~44% neutral, ~28% up, ~28% down distribution.
        """
        # Calculate rolling standard deviation (20-bar)
        if self.current_step >= 20:
            recent_prices = self.m5_data.iloc[self.current_step-20:self.current_step]['close'].values
            rolling_stdev = np.std(recent_prices)
        else:
            rolling_stdev = 0.001  # Default for insufficient history

        price_change = next_price - current_price
        threshold = 0.33 * rolling_stdev

        if price_change > threshold:
            return 0  # UP
        elif price_change < -threshold:
            return 2  # DOWN
        else:
            return 1  # NEUTRAL


class SimplePolicy:
    """Simple rule-based policy for demonstration"""

    def get_action(self, state):
        """Get action based on state features"""
        # Unpack features
        react_ratio = state[0]
        h1_trend = state[1]
        efficiency = state[3]
        bb_position = state[4]
        use_mr = state[6]
        position_side = state[7]

        # Regime-based strategy
        if use_mr > 0.5:  # Mean reversion mode
            if bb_position < -0.8 and position_side <= 0:
                return 1  # Buy oversold
            elif bb_position > 0.8 and position_side >= 0:
                return 2  # Sell overbought
        else:  # Trend following mode
            if react_ratio > 0.5 and h1_trend > 0 and efficiency > 0.3 and position_side <= 0:
                return 1  # Buy with trend
            elif react_ratio < -0.5 and h1_trend < 0 and efficiency > 0.3 and position_side >= 0:
                return 2  # Sell with trend

        # Close if in position for too long
        if state[9] > 0.2:  # bars_since_entry normalized > 20
            return 3

        return 0  # Hold


def train(load_checkpoint=None, save_freq=2):
    """Run training simulation with checkpoint support.

    Args:
        load_checkpoint: Path to checkpoint to resume from
        save_freq: Save checkpoint every N episodes
    """
    print("\n" + "="*60)
    print("MINIMAL PPO-STYLE TRAINING")
    print("="*60)
    print("\n📊 DATA SPLITS (1M bars total):")
    print("  Training:   bars 100,000 - 700,000 (600k bars, 60%)")
    print("  Validation: bars 700,000 - 1,000,000 (300k bars, 30%)")
    print("  Test:       bars 1,000,000 - 1,100,000 (100k bars, 10%)")

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager("checkpoints")

    # Initialize rolling expectancy tracker
    from rolling_expectancy import RollingExpectancyTracker
    expectancy_tracker = RollingExpectancyTracker(window_sizes=[100, 500, 1000])

    # Load checkpoint if provided
    start_episode = 0
    policy_state = None
    training_history = []

    if load_checkpoint:
        checkpoint = ckpt_manager.load_checkpoint(load_checkpoint)
        if checkpoint:
            start_episode = checkpoint['episode']
            policy_state = checkpoint.get('policy_state', None)
            training_history = checkpoint.get('training_history', [])
            print(f"\n🔄 Resuming from episode {start_episode}")
            print(f"   Previous expectancy: {checkpoint['expectancy_R']:.3f}R")

    # Create environment (using training data)
    env = MinimalTradingEnv(
        db_path="../../../data/master.duckdb" if not os.path.exists("/app/data/master.duckdb") else "/app/data/master.duckdb",
        start_idx=100000,
        end_idx=700000  # Training data only (600k bars)
    )

    # Create policy
    policy = SimplePolicy()

    # Training parameters
    n_episodes = 10  # 10 full passes through 600k bars
    results = training_history.copy()  # Resume from history if loaded
    all_trades_pnl = []  # Track all trades for R calculation

    # Adjust episode range if resuming
    episode_range = range(start_episode, start_episode + n_episodes)

    for episode_num, episode in enumerate(episode_range):
        print(f"\nEpisode {episode + 1} (Pass {episode_num + 1}/{n_episodes} through 600k bars)")
        print("-" * 40)

        state = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            # Get action from policy
            action = policy.get_action(state)

            # Execute action
            next_state, reward, done = env.step(action)

            episode_reward += reward
            steps += 1

            # Log progress
            if steps % 100 == 0:
                print(f"  Step {steps}: P&L={env.equity:.1f} pips, Trades={len(env.trades)}")

            state = next_state

            if done:
                break

        # Calculate expectancy_R (Van Tharp method)
        if env.trades_pnl:
            trades_array = np.array(env.trades_pnl)
            wins = trades_array[trades_array > 0]
            losses = trades_array[trades_array < 0]

            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 10.0  # R value
            expectancy_pips = sum(env.trades_pnl) / len(env.trades_pnl)
            expectancy_R = expectancy_pips / avg_loss

            all_trades_pnl.extend(env.trades_pnl)  # Accumulate for global stats

            # Update rolling expectancy tracker
            for trade_pips in env.trades_pnl:
                expectancy_tracker.add_trade(trade_pips)
        else:
            expectancy_R = 0.0
            avg_loss = 10.0
            expectancy_pips = 0.0

        # Get rolling expectancies
        rolling_stats = expectancy_tracker.calculate_expectancies()

        # Save rolling expectancy to file for monitoring
        expectancy_log = {
            'episode': episode + 1,
            'total_trades': len(expectancy_tracker.all_trades),
            'cumulative_pips': env.equity,
            **rolling_stats
        }

        os.makedirs("results", exist_ok=True)
        with open("results/rolling_expectancy.json", 'w') as f:
            json.dump(expectancy_log, f, indent=2)

        # Episode summary
        total_pips = env.equity  # Already in pips
        results.append({
            'episode': episode + 1,
            'total_pips': total_pips,
            'trades': len(env.trades),
            'final_equity': env.equity,
            'expectancy_pips': expectancy_pips,
            'expectancy_R': expectancy_R,
            'avg_loss_R': avg_loss
        })

        # Save checkpoint periodically
        if (episode + 1) % save_freq == 0 or episode_num == n_episodes - 1:
            # Calculate overall expectancy for checkpoint
            if all_trades_pnl:
                ckpt_trades = np.array(all_trades_pnl)
                ckpt_losses = ckpt_trades[ckpt_trades < 0]
                ckpt_R = abs(np.mean(ckpt_losses)) if len(ckpt_losses) > 0 else 10.0
                ckpt_expectancy_R = np.mean(ckpt_trades) / ckpt_R
            else:
                ckpt_expectancy_R = 0.0

            # Save checkpoint
            checkpoint_state = {
                'policy_state': policy_state,  # Would contain NN weights in full version
                'training_history': results,
                'all_trades_pnl': all_trades_pnl
            }

            checkpoint_metrics = {
                'total_trades': sum(r['trades'] for r in results),
                'avg_pips': np.mean([r['total_pips'] for r in results]),
                'sessions_completed': len(results)
            }

            ckpt_manager.save_checkpoint(
                state=checkpoint_state,
                episode=episode + 1,
                expectancy_R=ckpt_expectancy_R,
                metrics=checkpoint_metrics
            )

        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Final P&L: {env.equity:.1f} pips")
        print(f"  Total Trades: {len(env.trades)}")
        print(f"  Expectancy: {expectancy_pips:.1f} pips = {expectancy_R:.3f}R (R={avg_loss:.1f} pips)")

        # Display rolling expectancy every 10 episodes
        if (episode + 1) % 10 == 0:
            print("\n📊 ROLLING EXPECTANCY UPDATE:")
            for window_size in [100, 500, 1000]:
                exp_key = f'expectancy_R_{window_size}'
                if exp_key in rolling_stats:
                    exp_R = rolling_stats[exp_key]
                    sample = rolling_stats[f'sample_size_{window_size}']
                    quality = "🏆" if exp_R > 0.5 else "✅" if exp_R > 0.25 else "⚠️" if exp_R > 0 else "🔴"
                    print(f"  {window_size:4d}-trade: {exp_R:+.3f}R {quality} (n={sample})")

        if env.trades:
            winning_trades = [t for t in env.trades if t['pips'] > 0]
            win_rate = len(winning_trades) / len(env.trades) * 100
            total_pips = sum(t['pips'] for t in env.trades)
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total Pips: {total_pips:.1f}")

    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    df = pd.DataFrame(results)

    # Calculate overall expectancy_R from all trades
    if all_trades_pnl:
        all_trades = np.array(all_trades_pnl)
        all_losses = all_trades[all_trades < 0]
        global_R = abs(np.mean(all_losses)) if len(all_losses) > 0 else 10.0
        global_expectancy_pips = np.mean(all_trades)
        global_expectancy_R = global_expectancy_pips / global_R
    else:
        global_expectancy_R = 0.0
        global_R = 10.0
        global_expectancy_pips = 0.0

    print(f"\nAverage P&L: {df['total_pips'].mean():.1f} pips")
    print(f"Best P&L: {df['total_pips'].max():.1f} pips")
    print(f"Average Trades: {df['trades'].mean():.1f}")
    print(f"\n🎯 SESSION PERFORMANCE (Van Tharp):")
    print(f"  Overall Expectancy: {global_expectancy_pips:.1f} pips = {global_expectancy_R:.3f}R")
    print(f"  Risk Unit (R): {global_R:.1f} pips")
    print(f"  System Quality: {'EXCELLENT' if global_expectancy_R > 0.5 else 'GOOD' if global_expectancy_R > 0.25 else 'ACCEPTABLE' if global_expectancy_R > 0 else 'NEEDS IMPROVEMENT'}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    results_file = f"results/training_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")

    print("\n✅ Training demonstration complete!")
    print("\nThis simplified version shows:")
    print("- Correct feature calculations (7 market + 6 position)")
    print("- M5/H1 timeframe aggregation")
    print("- Regime-adaptive strategy (trend vs mean reversion)")
    print("- Position tracking and P&L calculation")
    print("\nFor full PPO training with neural networks, install:")
    print("  pip install gymnasium stable-baselines3 torch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal PPO Training")
    parser.add_argument('--load-checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save-freq', type=int, default=2, help='Save checkpoint every N episodes')
    args = parser.parse_args()

    train(load_checkpoint=args.load_checkpoint, save_freq=args.save_freq)