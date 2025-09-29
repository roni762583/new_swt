"""
Improved validation script with rolling std gating and comprehensive metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.trading_env_4action import TradingEnv4Action
from config_improved import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationMetrics:
    """Calculate comprehensive validation metrics."""

    def __init__(self):
        self.trades = []
        self.equities = []
        self.gate_stats = {
            'total_gates': 0,
            'false_rejects': 0,
            'successful_gates': 0,
            'gate_rates': []
        }
        self.rolling_stds = []
        self.thresholds = []

    def add_episode(self, info: Dict, episode_trades: List[float]):
        """Add episode results."""
        self.trades.extend(episode_trades)
        if 'equity' in info:
            self.equities.append(info['equity'])
        if 'gates_triggered' in info:
            self.gate_stats['total_gates'] += info['gates_triggered']
            self.gate_stats['gate_rates'].append(info.get('gate_rate', 0))
        if 'current_rolling_std' in info:
            self.rolling_stds.append(info['current_rolling_std'])
        if 'current_threshold' in info:
            self.thresholds.append(info['current_threshold'])

    def calculate_metrics(self) -> Dict:
        """Calculate all validation metrics."""
        if not self.trades:
            return {'error': 'No trades to analyze'}

        trades_array = np.array(self.trades)
        profitable = trades_array[trades_array > 0]
        losses = trades_array[trades_array <= 0]

        # Basic metrics
        metrics = {
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable),
            'losing_trades': len(losses),
            'win_rate': len(profitable) / len(self.trades) * 100,
            'avg_trade_pips': np.mean(trades_array),
            'total_pips': np.sum(trades_array),
        }

        # Expectancy metrics
        if len(losses) > 0:
            avg_loss = abs(np.mean(losses))
            metrics['avg_loss'] = avg_loss
            metrics['expectancy_R'] = metrics['avg_trade_pips'] / avg_loss
        else:
            metrics['expectancy_R'] = float('inf') if metrics['avg_trade_pips'] > 0 else 0

        # Van Tharp SQN (System Quality Number)
        if len(trades_array) > 1:
            metrics['sqn'] = np.sqrt(len(trades_array)) * metrics['avg_trade_pips'] / np.std(trades_array)
        else:
            metrics['sqn'] = 0

        # Recovery ratio
        if len(self.equities) > 0:
            equity_curve = np.array(self.equities)
            drawdowns = np.maximum.accumulate(equity_curve) - equity_curve
            max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
            total_profit = equity_curve[-1] - equity_curve[0]
            metrics['recovery_ratio'] = total_profit / max_dd if max_dd > 0 else float('inf')
            metrics['max_drawdown'] = max_dd

        # Profit factor
        if len(losses) > 0:
            gross_profit = np.sum(profitable)
            gross_loss = abs(np.sum(losses))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            metrics['profit_factor'] = float('inf') if len(profitable) > 0 else 1.0

        # Sharpe ratio (assuming daily)
        if len(trades_array) > 1:
            returns = pd.Series(trades_array)
            metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        else:
            metrics['sharpe_ratio'] = 0

        # Rolling expectancy windows
        for window in [100, 500, 1000]:
            if len(trades_array) >= window:
                metrics[f'expectancy_{window}'] = np.mean(trades_array[-window:])

        # Gating metrics
        if self.gate_stats['gate_rates']:
            metrics['avg_gate_rate'] = np.mean(self.gate_stats['gate_rates']) * 100
            metrics['total_gates'] = self.gate_stats['total_gates']

        # Rolling std metrics
        if self.rolling_stds:
            metrics['avg_rolling_std'] = np.mean(self.rolling_stds)
            metrics['avg_threshold'] = np.mean(self.thresholds)

        # Trade distribution
        if len(profitable) > 0:
            metrics['avg_win'] = np.mean(profitable)
            metrics['max_win'] = np.max(profitable)
        if len(losses) > 0:
            metrics['avg_loss'] = np.mean(losses)
            metrics['max_loss'] = np.min(losses)

        return metrics


def validate_model(
    model_path: str,
    n_episodes: int = 100,
    save_results: bool = True
) -> Dict:
    """Validate a trained model."""

    logger.info(f"Validating model: {model_path}")
    logger.info(f"Running {n_episodes} episodes")

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return {}

    # Check database
    if not os.path.exists("precomputed_features.duckdb"):
        logger.error("Database not found! Run precompute_features_to_db.py first.")
        return {}

    # Create environment
    env = TradingEnv4Action(
        db_path="precomputed_features.duckdb",
        episode_length=TRAINING["episode_length"],
        initial_balance=TRAINING["initial_balance"],
        instrument=DEFAULT_INSTRUMENT,
        reward_scaling=1.0,  # No scaling for validation
        seed=42,
        # Use stricter gating for validation
        sigma_window=GATING_CONFIG["sigma_window"],
        k_threshold=GATING_CONFIG["k_threshold_end"],  # Use final strict value
        m_spread=GATING_CONFIG["m_spread"],
        min_threshold_pips=GATING_CONFIG["min_threshold_pips"],
        use_hard_gate=False,  # Soft gate for validation
        gate_penalty=0,  # No penalty in validation
        # No weighted learning in validation
        winner_weight=1.0,
        loser_weight=1.0,
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Check for vec normalize
    vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = PPO.load(model_path, env=env)

    # Run validation
    metrics_tracker = ValidationMetrics()
    episode_results = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_trades = []
        episode_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1

            # Track trades
            if info[0].get('trades', 0) > len(episode_trades):
                # New trade closed
                if 'expectancy' in info[0]:
                    episode_trades.append(info[0]['expectancy'])

        # Store episode results
        episode_info = info[0]
        metrics_tracker.add_episode(episode_info, episode_trades)

        episode_results.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'trades': len(episode_trades),
            'pips': sum(episode_trades) if episode_trades else 0,
            'steps': steps,
            'gate_rate': episode_info.get('gate_rate', 0)
        })

        # Log progress
        if (episode + 1) % 10 == 0:
            recent_pips = sum([r['pips'] for r in episode_results[-10:]])
            recent_trades = sum([r['trades'] for r in episode_results[-10:]])
            logger.info(f"Episode {episode + 1}/{n_episodes}: "
                       f"Recent 10 eps: {recent_pips:.1f} pips, {recent_trades} trades")

    # Calculate final metrics
    final_metrics = metrics_tracker.calculate_metrics()

    # Add summary stats
    final_metrics['episodes'] = n_episodes
    final_metrics['model_path'] = model_path
    final_metrics['timestamp'] = datetime.now().isoformat()

    # Log results
    logger.info("\n" + "="*50)
    logger.info("VALIDATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total Trades: {final_metrics.get('total_trades', 0)}")
    logger.info(f"Win Rate: {final_metrics.get('win_rate', 0):.2f}%")
    logger.info(f"Expectancy: {final_metrics.get('avg_trade_pips', 0):.4f} pips/trade")
    logger.info(f"Expectancy_R: {final_metrics.get('expectancy_R', 0):.4f}")
    logger.info(f"Total Pips: {final_metrics.get('total_pips', 0):.2f}")
    logger.info(f"SQN: {final_metrics.get('sqn', 0):.4f}")
    logger.info(f"Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"Recovery Ratio: {final_metrics.get('recovery_ratio', 0):.2f}")
    logger.info(f"Profit Factor: {final_metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Max Drawdown: {final_metrics.get('max_drawdown', 0):.2f}")
    logger.info(f"Avg Gate Rate: {final_metrics.get('avg_gate_rate', 0):.2f}%")
    logger.info(f"Avg Rolling Std: {final_metrics.get('avg_rolling_std', 0):.4f}")
    logger.info("="*50)

    # Save results
    if save_results:
        results_dir = "validation_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save metrics
        metrics_file = os.path.join(results_dir, f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Results saved to: {metrics_file}")

        # Save episode details
        episodes_file = os.path.join(results_dir, f"episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(episodes_file, 'w') as f:
            json.dump(episode_results, f, indent=2)

    return final_metrics


def compare_models(model_paths: List[str], n_episodes: int = 50):
    """Compare multiple models."""
    results = {}

    for path in model_paths:
        model_name = os.path.basename(path)
        logger.info(f"\nValidating: {model_name}")
        metrics = validate_model(path, n_episodes, save_results=False)
        results[model_name] = metrics

    # Create comparison table
    df = pd.DataFrame(results).T
    print("\nMODEL COMPARISON")
    print("="*80)
    print(df[['win_rate', 'avg_trade_pips', 'expectancy_R', 'sqn', 'sharpe_ratio', 'recovery_ratio']])

    # Save comparison
    comparison_file = f"validation_results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(comparison_file)
    logger.info(f"Comparison saved to: {comparison_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate PPO trading model")
    parser.add_argument("--model", type=str, default="models/best_model.zip",
                       help="Path to model file")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of validation episodes")
    parser.add_argument("--compare", nargs="+",
                       help="Compare multiple models")

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.episodes)
    else:
        validate_model(args.model, args.episodes)