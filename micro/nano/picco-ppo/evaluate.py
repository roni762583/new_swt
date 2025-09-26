#!/usr/bin/env python3
"""
Evaluation and Backtesting Script for Trained PPO Models.

Tests model performance on unseen data and generates detailed reports.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env.trading_env import M5H1TradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained PPO models on test data."""

    def __init__(
        self,
        model_path: str,
        test_start: int = 900000,
        test_end: int = 1000000,
        n_episodes: int = 10
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to saved model
            test_start: Start index for test data
            test_end: End index for test data
            n_episodes: Number of evaluation episodes
        """
        self.model_path = model_path
        self.test_start = test_start
        self.test_end = test_end
        self.n_episodes = n_episodes

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = PPO.load(model_path)

        # Results storage
        self.episode_results = []
        self.all_trades = []

    def evaluate(self) -> Dict:
        """Run evaluation episodes."""

        logger.info(f"Running {self.n_episodes} evaluation episodes...")

        for episode in range(self.n_episodes):
            # Create test environment with random start
            env = M5H1TradingEnv(
                db_path="../../../../data/master.duckdb",
                start_idx=self.test_start,
                end_idx=self.test_end,
                initial_balance=10000.0,
                max_episode_steps=1000
            )

            # Run episode
            obs, info = env.reset()
            done = False
            episode_trades = []
            episode_rewards = []

            while not done:
                # Get action from model (deterministic)
                action, _ = self.model.predict(obs, deterministic=True)

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_rewards.append(reward)

                # Track trades
                if len(env.episode_trades) > len(episode_trades):
                    episode_trades = env.episode_trades.copy()

            # Store results
            final_equity = info['equity']
            total_return = (final_equity - 10000) / 10000 * 100

            episode_result = {
                'episode': episode,
                'final_equity': final_equity,
                'total_return': total_return,
                'n_trades': len(episode_trades),
                'max_drawdown': info['max_drawdown'],
                'total_reward': sum(episode_rewards)
            }

            self.episode_results.append(episode_result)
            self.all_trades.extend(episode_trades)

            logger.info(f"Episode {episode}: Return={total_return:.2f}%, "
                       f"Trades={len(episode_trades)}, Equity=${final_equity:.2f}")

            env.close()

        return self._calculate_statistics()

    def _calculate_statistics(self) -> Dict:
        """Calculate performance statistics."""

        df = pd.DataFrame(self.episode_results)

        # Calculate statistics
        stats = {
            'mean_return': df['total_return'].mean(),
            'std_return': df['total_return'].std(),
            'sharpe_ratio': df['total_return'].mean() / (df['total_return'].std() + 1e-10),
            'mean_trades': df['n_trades'].mean(),
            'mean_equity': df['final_equity'].mean(),
            'win_rate': (df['total_return'] > 0).mean() * 100,
            'max_return': df['total_return'].max(),
            'min_return': df['total_return'].min(),
            'mean_max_dd': df['max_drawdown'].mean()
        }

        # Trade statistics
        if self.all_trades:
            trades_with_pnl = [t for t in self.all_trades if 'pips' in t]
            if trades_with_pnl:
                pips = [t['pips'] for t in trades_with_pnl]
                stats['total_pips'] = sum(pips)
                stats['avg_pips_per_trade'] = np.mean(pips)
                stats['trade_win_rate'] = (np.array(pips) > 0).mean() * 100

        return stats

    def generate_report(self, save_dir: str = "./results"):
        """Generate evaluation report with visualizations."""

        os.makedirs(save_dir, exist_ok=True)

        stats = self._calculate_statistics()

        # Create report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(save_dir, f"evaluation_report_{timestamp}.txt")

        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PPO MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: bars {self.test_start:,} to {self.test_end:,}\n")
            f.write(f"Episodes: {self.n_episodes}\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Return: {stats['mean_return']:.2f}%\n")
            f.write(f"Std Return: {stats['std_return']:.2f}%\n")
            f.write(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}\n")
            f.write(f"Episode Win Rate: {stats['win_rate']:.1f}%\n")
            f.write(f"Mean Final Equity: ${stats['mean_equity']:.2f}\n")
            f.write(f"Mean Max Drawdown: {stats['mean_max_dd']:.1f} pips\n")

            if 'total_pips' in stats:
                f.write(f"\nTRADE STATISTICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Trades: {len(self.all_trades)}\n")
                f.write(f"Mean Trades per Episode: {stats['mean_trades']:.1f}\n")
                f.write(f"Total Pips: {stats['total_pips']:.1f}\n")
                f.write(f"Avg Pips per Trade: {stats['avg_pips_per_trade']:.2f}\n")
                f.write(f"Trade Win Rate: {stats['trade_win_rate']:.1f}%\n")

            f.write("\n" + "="*60 + "\n")

        logger.info(f"Report saved to {report_file}")

        # Generate plots
        self._create_visualizations(save_dir, timestamp)

    def _create_visualizations(self, save_dir: str, timestamp: str):
        """Create evaluation visualizations."""

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Episode returns distribution
        df = pd.DataFrame(self.episode_results)

        ax = axes[0, 0]
        ax.bar(df['episode'], df['total_return'], color=['green' if r > 0 else 'red'
                                                         for r in df['total_return']])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Episode Returns')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)

        # 2. Equity curves
        ax = axes[0, 1]
        for _, row in df.iterrows():
            equity_curve = [10000]  # Start with initial balance
            # Simulate equity growth (simplified)
            final = row['final_equity']
            steps = 100
            growth = (final - 10000) / steps
            for i in range(steps):
                equity_curve.append(equity_curve[-1] + growth)
            ax.plot(equity_curve, alpha=0.5)

        ax.set_title('Equity Curves')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)

        # 3. Trade distribution
        ax = axes[1, 0]
        if self.all_trades:
            trades_with_pnl = [t['pips'] for t in self.all_trades if 'pips' in t]
            if trades_with_pnl:
                ax.hist(trades_with_pnl, bins=30, edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--')
                ax.set_title('Trade P&L Distribution')
                ax.set_xlabel('Pips')
                ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

        # 4. Performance metrics
        ax = axes[1, 1]
        ax.axis('off')

        stats = self._calculate_statistics()
        metrics_text = f"""
Performance Summary:
-------------------
Mean Return: {stats['mean_return']:.2f}%
Sharpe Ratio: {stats['sharpe_ratio']:.3f}
Win Rate: {stats['win_rate']:.1f}%

Best Episode: {stats['max_return']:.2f}%
Worst Episode: {stats['min_return']:.2f}%

Mean Trades: {stats['mean_trades']:.1f}
Mean Equity: ${stats['mean_equity']:.2f}
        """

        ax.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
               verticalalignment='center')

        plt.suptitle('PPO Model Evaluation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(save_dir, f"evaluation_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {plot_file}")


def main():
    """Main evaluation function."""

    parser = argparse.ArgumentParser(description="Evaluate PPO Trading Model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model file")
    parser.add_argument("--test_start", type=int, default=900000,
                       help="Test data start index")
    parser.add_argument("--test_end", type=int, default=1000000,
                       help="Test data end index")
    parser.add_argument("--n_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory for results")

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        test_start=args.test_start,
        test_end=args.test_end,
        n_episodes=args.n_episodes
    )

    # Run evaluation
    stats = evaluator.evaluate()

    # Generate report
    evaluator.generate_report(args.save_dir)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Mean Return: {stats['mean_return']:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    if 'total_pips' in stats:
        print(f"Total Pips: {stats['total_pips']:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()