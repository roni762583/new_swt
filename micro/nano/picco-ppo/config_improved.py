"""
Improved trading and training configuration with gating and optimized hyperparameters.
"""

# Instrument-specific configurations
INSTRUMENTS = {
    "GBPJPY": {
        "pip_value": 0.01,      # 1 pip = 0.01 for JPY pairs
        "pip_multiplier": 100,  # To convert price difference to pips
        "spread": 4.0,          # Spread in pips
        "trade_size": 1000.0,   # Standard trade size (fixed, no compounding)
        "max_exposure": 1,      # Maximum simultaneous positions
        "max_consecutive_losses": 5,  # Circuit breaker
    }
}

# Training configuration
TRAINING = {
    "total_timesteps": 1_000_000,
    "n_envs": 4,                    # Number of parallel environments
    "eval_freq": 10_000,             # Evaluation frequency
    "save_freq": 10_000,             # Checkpoint save frequency
    "tensorboard_log": "./tensorboard/",
    "model_dir": "./models/",
    "episode_length": 2000,          # M5 bars per episode
    "initial_balance": 10000.0,
    "reward_scaling": 0.01,
    # Early stopping
    "early_stop_threshold": -0.3,   # Stop if expectancy drops below this
    "patience": 10,                 # Episodes to wait before stopping
}

# Gating configuration (rolling std-based)
GATING_CONFIG = {
    "sigma_window": 12,             # Rolling window for M5 (12 bars = 1 hour)
    "k_threshold_start": 0.15,      # Initial threshold multiplier (lenient)
    "k_threshold_end": 0.25,        # Final threshold multiplier (strict)
    "k_anneal_steps": 200_000,      # Steps to anneal k from start to end
    "m_spread": 2.0,                # Minimum threshold as spread multiple
    "min_threshold_pips": 2.0,      # Absolute minimum threshold
    "use_hard_gate": True,          # Start with hard gate
    "switch_to_soft_at": 500_000,   # Switch to soft gate after this many steps
    "gate_penalty": -0.01,          # Penalty for gated actions
}

# PPO hyperparameters (optimized)
PPO_CONFIG = {
    "learning_rate": 3e-4,           # Learning rate
    "n_steps": 2048,                 # Steps per update
    "batch_size": 64,                # Minibatch size
    "n_epochs": 10,                  # PPO epochs
    "gamma": 0.99,                   # Discount factor
    "gae_lambda": 0.95,              # GAE lambda
    "clip_range": 0.2,               # PPO clip parameter
    "clip_range_vf": None,           # Value function clip
    "ent_coef": 0.02,                # Start higher for exploration
    "ent_coef_end": 0.005,           # Final entropy coefficient
    "ent_anneal_steps": 500_000,    # Steps to anneal entropy
    "vf_coef": 0.25,                 # Reduced value function coefficient
    "max_grad_norm": 0.5,            # Gradient clipping
    "use_sde": False,                # State dependent exploration
    "target_kl": 0.03,               # KL divergence threshold for early stopping
    "normalize_advantage": True,     # Normalize advantages
    "verbose": 1,
    "seed": 42,
}

# Network architecture (potentially smaller to avoid overfitting)
NETWORK_CONFIG = {
    "net_arch": [128, 128],          # Smaller network to reduce overfitting
    "activation_fn": "ReLU",         # Activation function
    "normalize_images": False,
    "ortho_init": True,              # Orthogonal initialization
    "use_rms_prop": False,           # Use Adam instead of RMSProp
    "weight_decay": 1e-4,            # L2 regularization
}

# Weighted learning configuration (replaces winner-only phase)
LEARNING_CONFIG = {
    "winner_weight": 1.0,            # Weight for profitable trades
    "loser_weight_start": 0.2,       # Initial weight for losses
    "loser_weight_end": 1.0,         # Final weight for losses (equal)
    "weight_anneal_steps": 200_000,  # Steps to anneal weights
    "use_weighted_sampling": True,   # Enable weighted experience replay
}

# Replay buffer configuration (for off-policy methods if used)
BUFFER_CONFIG = {
    "buffer_size": 100_000,          # Total buffer size
    "prioritized_replay": True,      # Use PER
    "per_alpha": 0.6,                # PER alpha
    "per_beta_start": 0.4,           # Initial beta
    "per_beta_end": 1.0,             # Final beta
    "per_anneal_steps": 500_000,     # Steps to anneal beta
    "success_buffer_size": 5_000,    # Size of success memory
    "success_sample_rate": 0.1,      # Fraction of batch from success buffer
    "recency_weight": True,          # Use recency weighting
    "recency_decay": 0.99,           # Exponential decay for recency
    "trade_quota": 0.4,              # Minimum fraction of trade experiences
}

# Environment constraints (defines the physics of trading)
ENV_CONSTRAINTS = {
    "fixed_position_size": 1000,     # Fixed position size (no compounding)
    "max_positions": 1,              # Only 1 position at a time (no pyramiding)
    "use_position_sizing": False,    # No dynamic sizing for now
}

# Deployment-layer risk controls (NOT for training, only for live trading)
# These belong in trade_live_improved.py, not in the training environment
DEPLOYMENT_RISK_LIMITS = {
    "max_daily_loss": 0.02,          # 2% daily stop (deployment only)
    "max_drawdown": 0.05,            # 5% absolute stop (deployment only)
    "max_consecutive_losses": 5,     # Circuit breaker (deployment only)
    "note": "These limits are enforced at deployment, NOT during training"
}

# Monitoring and logging
MONITORING_CONFIG = {
    "log_interval": 100,             # Log every N episodes
    "eval_episodes": 10,             # Episodes for evaluation
    "track_metrics": [
        "expectancy_pips",
        "win_rate",
        "avg_trade_pips",
        "max_drawdown",
        "recovery_ratio",
        "sharpe_ratio",
        "gate_rate",
        "false_reject_rate",
        "rolling_expectancy_100",
        "rolling_expectancy_500",
        "rolling_expectancy_1000",
    ],
    "save_best_metric": "expectancy_pips",  # Metric to determine best model
    "tensorboard_metrics": True,     # Log to tensorboard
}

# Curriculum learning schedule
CURRICULUM_CONFIG = {
    "stages": [
        {
            "timesteps": 0,
            "k_threshold": 0.15,      # Lenient gating
            "ent_coef": 0.02,         # High exploration
            "gate_type": "hard",
        },
        {
            "timesteps": 200_000,
            "k_threshold": 0.20,      # Medium gating
            "ent_coef": 0.01,         # Medium exploration
            "gate_type": "hard",
        },
        {
            "timesteps": 500_000,
            "k_threshold": 0.25,      # Strict gating
            "ent_coef": 0.005,        # Low exploration
            "gate_type": "soft",      # Switch to soft gate
        },
    ]
}

# Stress testing configuration
STRESS_TEST_CONFIG = {
    "add_slippage": True,            # Add random slippage
    "slippage_range": (0.5, 2.0),    # Slippage in pips
    "spread_variance": 0.5,          # Spread can vary by ±50%
    "latency_ms": 50,                # Simulated latency
    "adverse_selection": 0.1,        # Remove 10% best trades in stress test
}

# Default instrument
DEFAULT_INSTRUMENT = "GBPJPY"

# Helper functions
def get_instrument_config(instrument=DEFAULT_INSTRUMENT):
    """Get configuration for specific instrument."""
    return INSTRUMENTS.get(instrument, INSTRUMENTS[DEFAULT_INSTRUMENT])

def get_annealed_value(start, end, current_step, total_steps):
    """Calculate annealed value between start and end."""
    progress = min(current_step / total_steps, 1.0)
    return start + (end - start) * progress

def get_current_curriculum_stage(timestep):
    """Get current curriculum stage based on timestep."""
    for stage in reversed(CURRICULUM_CONFIG["stages"]):
        if timestep >= stage["timesteps"]:
            return stage
    return CURRICULUM_CONFIG["stages"][0]