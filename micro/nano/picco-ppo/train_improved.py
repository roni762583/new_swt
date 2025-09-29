"""
Improved PPO training script with rolling std gating, weighted learning, and optimized hyperparameters.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import logging
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.trading_env_improved import ImprovedTradingEnv
from config_improved import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """Custom callback to track trading-specific metrics with gating stats."""

    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_count = 0
        self.trades_history = []
        self.expectancy_history = []
        self.gate_stats = {
            'gates_triggered': [],
            'false_rejects': [],
            'gate_rates': []
        }
        self.rolling_windows = [100, 500, 1000]
        self.best_expectancy = -float('inf')
        self.current_stage = None

    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1

            # Extract info from environment
            info = self.locals["infos"][0]

            # Track trades
            if "trades" in info:
                self.trades_history.append(info["trades"])

            # Track expectancy
            if "expectancy" in info:
                self.expectancy_history.append(info["expectancy"])

                # Calculate rolling expectancies
                for window in self.rolling_windows:
                    if len(self.expectancy_history) >= window:
                        rolling_exp = np.mean(self.expectancy_history[-window:])
                        self.logger.record(f"rollout/expectancy_{window}", rolling_exp)

            # Track gating metrics
            if "gates_triggered" in info:
                self.gate_stats['gates_triggered'].append(info["gates_triggered"])
                self.gate_stats['gate_rates'].append(info.get("gate_rate", 0))

                # Log gating stats
                self.logger.record("gating/gates_triggered", info["gates_triggered"])
                self.logger.record("gating/gate_rate", info.get("gate_rate", 0))
                self.logger.record("gating/current_threshold", info.get("current_threshold", 0))
                self.logger.record("gating/rolling_std", info.get("current_rolling_std", 0))

            # Track win rate
            if "win_rate" in info:
                self.logger.record("rollout/win_rate", info["win_rate"])

            # Track drawdown
            if "accumulated_dd" in info:
                self.logger.record("risk/accumulated_drawdown", info["accumulated_dd"])

            # Log every N episodes
            if self.episode_count % self.log_interval == 0:
                self._log_summary()

        # Update curriculum stage
        timestep = self.num_timesteps
        new_stage = get_current_curriculum_stage(timestep)
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            logger.info(f"Curriculum stage updated at timestep {timestep}: {new_stage}")

            # Update entropy coefficient
            if hasattr(self.model, "ent_coef"):
                self.model.ent_coef = new_stage["ent_coef"]

        return True

    def _log_summary(self):
        """Log summary statistics."""
        if len(self.expectancy_history) > 0:
            current_exp = self.expectancy_history[-1]

            # Check if best model
            if current_exp > self.best_expectancy:
                self.best_expectancy = current_exp
                logger.info(f"New best expectancy: {current_exp:.4f} pips/trade")

                # Save best model
                model_path = os.path.join(TRAINING["model_dir"], "best_model.zip")
                self.model.save(model_path)

                # Save metrics
                metrics = {
                    "expectancy": float(current_exp),
                    "episode": self.episode_count,
                    "timestep": self.num_timesteps,
                    "gate_rate": float(np.mean(self.gate_stats['gate_rates'][-100:])) if self.gate_stats['gate_rates'] else 0,
                    "timestamp": datetime.now().isoformat()
                }
                with open(os.path.join(TRAINING["model_dir"], "best_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

        # Log average gate rate
        if self.gate_stats['gate_rates']:
            avg_gate_rate = np.mean(self.gate_stats['gate_rates'][-100:])
            logger.info(f"Episode {self.episode_count}: Avg gate rate: {avg_gate_rate:.2%}")


# Note: Risk management callbacks removed from training
# Deployment-layer risk controls should be handled in live trading only
# The training environment should learn the natural reward distribution without artificial limits


def make_env(env_id: int, rank: int, seed: int = 0):
    """Create a single environment instance."""
    def _init():
        # Get current timestep for curriculum
        # Note: This is approximate as we don't have access to global timesteps here
        env = ImprovedTradingEnv(
            db_path="precomputed_features.duckdb",
            episode_length=TRAINING["episode_length"],
            initial_balance=TRAINING["initial_balance"],
            instrument=DEFAULT_INSTRUMENT,
            reward_scaling=TRAINING["reward_scaling"],
            seed=seed + rank,
            # Gating parameters
            sigma_window=GATING_CONFIG["sigma_window"],
            k_threshold=GATING_CONFIG["k_threshold_start"],
            m_spread=GATING_CONFIG["m_spread"],
            min_threshold_pips=GATING_CONFIG["min_threshold_pips"],
            use_hard_gate=GATING_CONFIG["use_hard_gate"],
            gate_penalty=GATING_CONFIG["gate_penalty"],
            # Weighted learning parameters
            winner_weight=LEARNING_CONFIG["winner_weight"],
            loser_weight=LEARNING_CONFIG["loser_weight_start"],
            weight_anneal_steps=LEARNING_CONFIG["weight_anneal_steps"],
        )
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


def train():
    """Main training function with improved configuration."""

    # Create directories
    os.makedirs(TRAINING["model_dir"], exist_ok=True)
    os.makedirs(TRAINING["tensorboard_log"], exist_ok=True)

    # Check if database exists
    if not os.path.exists("precomputed_features.duckdb"):
        logger.error("Database not found! Run precompute_features_to_db.py first.")
        return

    # Create vectorized environments
    if TRAINING["n_envs"] > 1:
        env = SubprocVecEnv([
            make_env(i, i, PPO_CONFIG["seed"])
            for i in range(TRAINING["n_envs"])
        ])
    else:
        env = DummyVecEnv([make_env(0, 0, PPO_CONFIG["seed"])])

    # Optionally add reward normalization
    if PPO_CONFIG.get("normalize_advantage", True):
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,  # Don't normalize observations
            norm_reward=True,  # Normalize rewards
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=PPO_CONFIG["gamma"]
        )

    # Create PPO model with improved config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_CONFIG["learning_rate"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        clip_range_vf=PPO_CONFIG["clip_range_vf"],
        ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        use_sde=PPO_CONFIG["use_sde"],
        target_kl=PPO_CONFIG.get("target_kl", None),
        tensorboard_log=TRAINING["tensorboard_log"],
        policy_kwargs={
            "net_arch": NETWORK_CONFIG["net_arch"],
            "activation_fn": getattr(torch.nn, NETWORK_CONFIG["activation_fn"]),
            "ortho_init": NETWORK_CONFIG.get("ortho_init", True),
        },
        verbose=PPO_CONFIG["verbose"],
        seed=PPO_CONFIG["seed"],
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING["save_freq"],
        save_path=TRAINING["model_dir"],
        name_prefix="ppo_improved",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Trading metrics callback
    metrics_callback = TradingMetricsCallback(
        log_interval=MONITORING_CONFIG["log_interval"]
    )
    callbacks.append(metrics_callback)

    # Note: Risk management callback removed - artificial limits don't belong in training
    # These controls are for deployment layer only (trade_live_improved.py)

    # Evaluation callback (optional)
    if TRAINING.get("eval_freq", 0) > 0:
        eval_env = DummyVecEnv([make_env(99, 99, PPO_CONFIG["seed"] + 99)])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(TRAINING["model_dir"], "eval_best"),
            log_path=os.path.join(TRAINING["model_dir"], "eval_logs"),
            eval_freq=TRAINING["eval_freq"],
            n_eval_episodes=MONITORING_CONFIG["eval_episodes"],
            deterministic=True,
        )
        callbacks.append(eval_callback)

    # Log configuration
    logger.info("Starting training with improved configuration:")
    logger.info(f"  - Gating: σ-based with k={GATING_CONFIG['k_threshold_start']} → {GATING_CONFIG['k_threshold_end']}")
    logger.info(f"  - Learning: Weighted sampling (winner={LEARNING_CONFIG['winner_weight']}, loser={LEARNING_CONFIG['loser_weight_start']}→1.0)")
    logger.info(f"  - PPO: lr={PPO_CONFIG['learning_rate']}, entropy={PPO_CONFIG['ent_coef']}→{PPO_CONFIG['ent_coef_end']}")
    logger.info(f"  - Network: {NETWORK_CONFIG['net_arch']}")
    logger.info(f"  - Environments: {TRAINING['n_envs']} parallel")

    # Train
    try:
        model.learn(
            total_timesteps=TRAINING["total_timesteps"],
            callback=callbacks,
            log_interval=1,
            tb_log_name=f"ppo_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reset_num_timesteps=True,
            progress_bar=False,
        )

        # Save final model
        model.save(os.path.join(TRAINING["model_dir"], "final_model"))
        if isinstance(env, VecNormalize):
            env.save(os.path.join(TRAINING["model_dir"], "vec_normalize.pkl"))

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save(os.path.join(TRAINING["model_dir"], "interrupted_model"))

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    finally:
        env.close()


if __name__ == "__main__":
    train()