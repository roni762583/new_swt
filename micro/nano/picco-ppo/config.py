"""
Trading and training configuration.
"""

# Instrument-specific configurations
INSTRUMENTS = {
    "GBPJPY": {
        "pip_value": 0.01,      # 1 pip = 0.01 for JPY pairs
        "pip_multiplier": 100,  # To convert price difference to pips
        "spread": 4.0,          # Spread in pips
        "trade_size": 1000.0,   # Standard trade size
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
}

# PPO hyperparameters
PPO_CONFIG = {
    "learning_rate": 3e-4,           # Adaptive learning rate
    "n_steps": 2048,                 # Steps per update
    "batch_size": 64,                # Minibatch size
    "n_epochs": 10,                  # PPO epochs
    "gamma": 0.99,                   # Discount factor
    "gae_lambda": 0.95,              # GAE lambda
    "clip_range": 0.2,               # PPO clip parameter
    "clip_range_vf": None,           # Value function clip
    "ent_coef": 0.01,                # Entropy coefficient for exploration
    "vf_coef": 0.5,                  # Value function coefficient
    "max_grad_norm": 0.5,            # Gradient clipping
    "use_sde": False,                # State dependent exploration
    "sde_sample_freq": -1,
    "verbose": 1,
    "seed": 42,
}

# Network architecture
NETWORK_CONFIG = {
    "net_arch": [256, 256],          # Two hidden layers
    "activation_fn": "ReLU",         # Activation function
    "normalize_images": False,
}

# Winner-focused learning settings
LEARNING_PHASES = {
    "phase1_threshold": 1000,        # Learn from winners until 1000 profitable trades
    "ignore_losses": True,           # Ignore losses in phase 1
}

# Default instrument
DEFAULT_INSTRUMENT = "GBPJPY"

# Get config for current instrument
def get_instrument_config(instrument=DEFAULT_INSTRUMENT):
    """Get configuration for specific instrument."""
    return INSTRUMENTS.get(instrument, INSTRUMENTS[DEFAULT_INSTRUMENT])