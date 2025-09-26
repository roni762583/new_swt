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

from env.trading_env import M5H1TradingEnv

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

    def _on_step(self) -> bool:
        # Log custom metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.evaluations_results) > 0:
                mean_reward = np.mean(self.evaluations_results[-1])
                logger.info(f"Step {self.n_calls}: Mean Reward = {mean_reward:.2f}")

                # Get equity from eval env
                eval_env = self.eval_env.envs[0]
                if hasattr(eval_env, 'equity'):
                    logger.info(f"  Current Equity: ${eval_env.equity:.2f}")

        return True


def create_env(start_idx: int, end_idx: int, seed: int = None):
    """Create a single trading environment."""

    env = M5H1TradingEnv(
        db_path="../../../../data/master.duckdb",
        start_idx=start_idx,
        end_idx=end_idx,
        initial_balance=10000.0,
        max_episode_steps=1000,
        reward_scaling=0.01
    )

    # Wrap with Monitor for logging
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


def train_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    tensorboard_log: str = "./tensorboard/",
    model_dir: str = "./models/"
):
    """
    Train PPO agent on trading environment.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
        save_freq: Model checkpoint frequency
        tensorboard_log: TensorBoard log directory
        model_dir: Model save directory
    """

    # Create directories
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_m5h1_{timestamp}"

    logger.info(f"Starting PPO training run: {run_name}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Parallel environments: {n_envs}")

    # Data split for train/eval
    # Using different data ranges to prevent overfitting
    train_start = 100000
    train_end = 800000
    eval_start = 800000
    eval_end = 900000

    # Create vectorized training environments
    logger.info("Creating training environments...")

    def make_env(rank: int):
        def _init():
            # Each env gets a different slice of training data
            env_range = (train_end - train_start) // n_envs
            start = train_start + rank * env_range
            end = start + env_range
            return create_env(start, end, seed=rank)
        return _init

    # Use SubprocVecEnv for true parallelization
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(0)])

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([lambda: create_env(eval_start, eval_end, seed=42)])

    # Check environment
    logger.info("Checking environment compatibility...")
    check_env(create_env(train_start, train_end))

    # PPO hyperparameters optimized for trading
    ppo_config = {
        "policy": "MlpPolicy",
        "env": train_env,
        "learning_rate": 3e-4,  # Adaptive learning rate
        "n_steps": 2048,  # Steps per update
        "batch_size": 64,  # Minibatch size
        "n_epochs": 10,  # PPO epochs
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # GAE lambda
        "clip_range": 0.2,  # PPO clip parameter
        "clip_range_vf": None,  # Value function clip
        "ent_coef": 0.01,  # Entropy coefficient for exploration
        "vf_coef": 0.5,  # Value function coefficient
        "max_grad_norm": 0.5,  # Gradient clipping
        "use_sde": False,  # State dependent exploration
        "sde_sample_freq": -1,
        "policy_kwargs": {
            "net_arch": [256, 256],  # Two hidden layers
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False
        },
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42
    }

    logger.info(f"Device: {ppo_config['device']}")
    logger.info("Creating PPO model...")

    model = PPO(**ppo_config)

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
            log_interval=10,
            tb_log_name=run_name,
            progress_bar=True
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
    parser.add_argument("--save_freq", type=int, default=50_000,
                       help="Model save frequency")
    parser.add_argument("--tensorboard_dir", type=str, default="./tensorboard/",
                       help="TensorBoard log directory")
    parser.add_argument("--model_dir", type=str, default="./models/",
                       help="Model save directory")

    args = parser.parse_args()

    # Train model
    model = train_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        tensorboard_log=args.tensorboard_dir,
        model_dir=args.model_dir
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()