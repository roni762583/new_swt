#!/usr/bin/env python3
"""
PPO Training Script for M5/H1 Trading Strategy.

Uses Stable-Baselines3 with optimal hyperparameters for trading.
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import torch

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import TRAINING, PPO_CONFIG, NETWORK_CONFIG

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from collections import deque

try:
    from env.trading_env_optimized import OptimizedTradingEnv as TradingEnv
    print("Using OptimizedTradingEnv with precomputed features")
except ImportError:
    from env.trading_env import TradingEnv
    print("Warning: Falling back to original TradingEnv")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingCallback(EvalCallback):
    """Custom callback for tracking trading metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf
        self.best_equity = -np.inf
        self.trade_results = deque(maxlen=100)  # Track last 100 trades
        self.all_trade_results = []  # Track all trades
        self.profitable_trades_total = 0
        self.total_trades = 0

    def _on_step(self) -> bool:
        # Log custom metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            # Collect trade results from all environments
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'trade_result' in info:
                    trade_pips = info['trade_result']
                    self.trade_results.append(trade_pips)
                    self.all_trade_results.append(trade_pips)
                    self.total_trades += 1
                    if trade_pips > 0:
                        self.profitable_trades_total += 1

            # Calculate rolling expectancy
            if len(self.trade_results) > 0:
                rolling_expectancy = np.mean(self.trade_results)
                win_rate = sum(1 for t in self.trade_results if t > 0) / len(self.trade_results) * 100

                print(f"\n{'='*60}")
                print(f"   Step {self.n_calls:,}: TRADING METRICS")
                print(f"   Profitable Trades: {self.profitable_trades_total:,} / 1000 target")
                print(f"   Total Trades: {self.total_trades:,}")
                print(f"   Rolling Expectancy (100): {rolling_expectancy:.2f} pips")
                print(f"   Rolling Win Rate: {win_rate:.1f}%")

                # Check learning phase from env
                for env_idx, info in enumerate(infos[:1]):
                    if 'learning_phase' in info:
                        print(f"   Learning Phase: {info['learning_phase'].upper()}")
                        if info['learning_phase'] == 'full_learning':
                            print(f"   ✅ FULL LEARNING ACTIVATED!")
                print(f"{'='*60}\n")

        return True


class LoggingWrapper(Monitor):
    """Wrapper to log trading metrics."""

    def __init__(self, env):
        super().__init__(env)
        self.trade_count = 0
        self.profitable_trades = 0
        self.recent_trades = deque(maxlen=50)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)

        # Log trade results
        if 'trade_result' in info:
            trade_pips = info['trade_result']
            self.trade_count += 1
            self.recent_trades.append(trade_pips)

            if trade_pips > 0:
                self.profitable_trades += 1

            # Log every trade initially, then every 10th trade
            if self.profitable_trades <= 100 or self.trade_count % 10 == 0:
                expectancy = np.mean(self.recent_trades) if self.recent_trades else 0
                win_rate = sum(1 for t in self.recent_trades if t > 0) / len(self.recent_trades) * 100 if self.recent_trades else 0

                status = "✅" if trade_pips > 0 else "❌"
                print(f"     {status} TRADE CLOSED: {trade_pips:.1f} pips (Env {info.get('env_id', 0)})")

                if self.trade_count % 10 == 0:
                    print(f"     [Profitable: {info.get('profitable_trades', 0)}/1000, Expectancy: {expectancy:.1f} pips, Win Rate: {win_rate:.0f}%]")

        return obs, reward, done, truncated, info

def create_env(seed: int = None, env_id: int = 0):
    """Create a single trading environment using precomputed features."""

    # Check if we're in Docker or local environment
    import os
    if os.path.exists("/app/precomputed_features.duckdb"):
        db_path = "/app/precomputed_features.duckdb"
    else:
        db_path = "precomputed_features.duckdb"

    env = TradingEnv(
        db_path=db_path,
        episode_length=TRAINING["episode_length"],
        initial_balance=TRAINING["initial_balance"],
        instrument="GBPJPY",  # Uses config for pip values, spread, trade_size
        reward_scaling=TRAINING["reward_scaling"],
        seed=seed
    )

    # Wrap with logging wrapper
    env = LoggingWrapper(env)

    # Store env_id for tracking
    env.env_id = env_id

    if seed is not None:
        env.reset(seed=seed)

    return env


def train_ppo(
    total_timesteps: int = None,
    n_envs: int = None,
    eval_freq: int = None,
    save_freq: int = None,
    tensorboard_log: str = None,
    model_dir: str = None,
    pretrain_path: str = None
):
    """
    Train PPO agent on trading environment.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
        save_freq: Model checkpoint frequency
        tensorboard_log: TensorBoard log directory
        pretrain_path: Path to pretrained checkpoint (optional)
        model_dir: Model save directory
    """

    # Use config values as defaults
    total_timesteps = total_timesteps or TRAINING["total_timesteps"]
    n_envs = n_envs or TRAINING["n_envs"]
    eval_freq = eval_freq or TRAINING["eval_freq"]
    save_freq = save_freq or TRAINING["save_freq"]
    tensorboard_log = tensorboard_log or TRAINING["tensorboard_log"]
    model_dir = model_dir or TRAINING["model_dir"]

    # Create directories
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_m5h1_{timestamp}"

    logger.info(f"Starting PPO training run: {run_name}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Parallel environments: {n_envs}")

    # Create vectorized training environments
    logger.info("Creating training environments...")

    def make_env(rank: int):
        def _init():
            # Each env gets a different random seed
            return create_env(seed=rank, env_id=rank + 1)
        return _init

    # Use SubprocVecEnv for true parallelization
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(0)])

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: create_env(seed=42, env_id=0)])

    # Check environment
    logger.info("Checking environment compatibility...")
    check_env(create_env(seed=42))

    # PPO hyperparameters from config
    ppo_config = {
        "policy": "MlpPolicy",
        "env": train_env,
        **PPO_CONFIG,  # Unpack all PPO config values
        "policy_kwargs": {
            "net_arch": NETWORK_CONFIG["net_arch"],
            "activation_fn": getattr(torch.nn, NETWORK_CONFIG["activation_fn"]),
            "normalize_images": NETWORK_CONFIG["normalize_images"]
        },
        "tensorboard_log": tensorboard_log,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    logger.info(f"Device: {ppo_config['device']}")
    logger.info("Creating PPO model...")

    model = PPO(**ppo_config)

    # Load pretrained weights if provided
    if pretrain_path:
        logger.info(f"Loading pretrained weights from: {pretrain_path}")
        try:
            checkpoint = torch.load(pretrain_path, map_location=ppo_config['device'])

            # SB3's policy network is accessible via model.policy
            # The pretrained model has action_dim=3, but PPO uses action_dim=4
            # We'll load the shared layers and skip the final policy head
            pretrained_dict = checkpoint['model_state_dict']
            model_dict = model.policy.state_dict()

            # Filter out incompatible keys (policy_head with different output dim)
            pretrained_dict_filtered = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            # Update model with pretrained weights
            model_dict.update(pretrained_dict_filtered)
            model.policy.load_state_dict(model_dict, strict=False)

            loaded_keys = len(pretrained_dict_filtered)
            total_keys = len(model_dict)
            logger.info(f"✅ Loaded {loaded_keys}/{total_keys} pretrained weights")
            logger.info(f"   Skipped: final policy head (3→4 action mismatch)")
            logger.info(f"   Pretrain val_acc: {checkpoint.get('val_acc', 'N/A')}%")
        except Exception as e:
            logger.error(f"❌ Failed to load pretrained weights: {e}")
            logger.info("Continuing with random initialization...")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for multiple envs
        save_path=model_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = TradingCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(model_dir, "eval_logs"),
        eval_freq=eval_freq // n_envs,  # Adjust for multiple envs
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train the model
    logger.info("Starting training...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100,  # Less frequent logging
            tb_log_name=run_name,
            progress_bar=False  # Disable progress bar for cleaner output
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Clean up
    train_env.close()
    eval_env.close()

    logger.info("Training completed!")

    return model


def main():
    """Main training function."""

    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on M5/H1 Trading")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--eval_freq", type=int, default=10_000,
                       help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=10_000,
                       help="Model save frequency")
    parser.add_argument("--tensorboard_dir", type=str, default="./tensorboard/",
                       help="TensorBoard log directory")
    parser.add_argument("--model_dir", type=str, default="./models/",
                       help="Model save directory")
    parser.add_argument("--pretrain", type=str, default=None,
                       help="Path to pretrained checkpoint (optional)")

    args = parser.parse_args()

    # Train model
    model = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        tensorboard_log=args.tensorboard_dir,
        model_dir=args.model_dir,
        pretrain_path=args.pretrain
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()