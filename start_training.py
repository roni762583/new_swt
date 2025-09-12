#!/usr/bin/env python3
"""
Start fresh SWT training with proper 137-feature architecture
Configured for Episode 13475 replacement with correct setup
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "experimental_research"))

from experimental_research.swt_training.swt_trainer import SWTTrainer, SWTTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_training_config():
    """Create training configuration with proper 137-feature setup"""
    
    config = {
        "experiment_name": "swt_muzero_137_features",
        "device": "cuda:0",  # Change to "cpu" if no GPU
        "seed": 42,
        
        # Training parameters
        "num_episodes": 50000,
        "train_interval": 10,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "gradient_clip_norm": 0.5,
        "min_buffer_size": 1000,
        "max_buffer_size": 100000,
        
        # Environment configuration
        "environment": {
            "session_hours": 6,  # 6-hour sessions
            "reward_type": "amddp1",  # AMDDP1 reward
            "spread_pips": 1.5,
            "pip_value": 0.01,
            "initial_balance": 10000,
            "leverage": 100,
            "reassign_on_trade_complete": True  # Critical for proper credit assignment
        },
        
        # Market encoder configuration (137 → 128 fusion)
        "market_encoder_config": {
            "market_feature_dim": 128,  # WST features
            "position_feature_dim": 9,   # Position features
            "hidden_dim": 128,           # Final fused dimension
            "dropout_rate": 0.1
        },
        
        # MuZero network configuration
        "muzero_config": {
            "total_input_dim": 137,      # 128 market + 9 position
            "final_input_dim": 128,      # After fusion
            "hidden_dim": 256,
            "num_actions": 4,
            "support_size": 601,
            "representation_blocks": 2,
            "dynamics_blocks": 2,
            "latent_z_dim": 16,
            "dropout_rate": 0.1,
            "layer_norm": True,
            "use_optimized_blocks": True
        },
        
        # MCTS configuration
        "mcts_config": {
            "num_simulations": 50,
            "c_puct": 1.25,
            "discount": 0.997,
            "temperature": 1.0,
            "root_dirichlet_alpha": 0.3,
            "root_exploration_fraction": 0.25,
            "pb_c_base": 19652,
            "pb_c_init": 1.25,
            "value_support_size": 601
        },
        
        # Checkpoint configuration
        "checkpoint_dir": "checkpoints",
        "checkpoint_frequency": 25,  # Save every 25 episodes
        "keep_last_n_checkpoints": 2,
        "save_best": True,
        
        # Data configuration
        "data_path": "data/GBPJPY_M1_3.5years_20250912.csv",  # Updated 3.5-year dataset
        "validation_split": 0.2,
        
        # Multi-processing
        "num_workers": 4,  # Parallel workers for data collection
        "use_multiprocessing": True
    }
    
    return config

def verify_data_exists(data_path):
    """Verify training data exists and has sufficient records"""
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        logger.info("Please ensure you have GBPJPY M1 data in the data/ directory")
        return False
    
    # Quick check for data size
    import pandas as pd
    try:
        df = pd.read_csv(data_path, nrows=1000)
        total_rows = sum(1 for _ in open(data_path)) - 1  # Subtract header
        
        logger.info(f"✅ Found data file with ~{total_rows:,} bars")
        
        # Check for required columns
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(set(df.columns)):
            logger.error(f"❌ Missing required columns. Found: {df.columns.tolist()}")
            return False
            
        # Check for sufficient data (at least 1 year)
        min_bars = 360 * 250  # ~250 trading days
        if total_rows < min_bars:
            logger.warning(f"⚠️ Limited data: {total_rows:,} bars (recommend >{min_bars:,})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading data file: {e}")
        return False

def main():
    """Main training entry point"""
    
    print("=" * 60)
    print("SWT STOCHASTIC MUZERO TRAINING")
    print("137 Features (128 Market + 9 Position)")
    print("=" * 60)
    
    # Create configuration
    config_dict = create_training_config()
    
    # Verify data exists
    if not verify_data_exists(config_dict["data_path"]):
        logger.error("Please download or provide GBPJPY M1 data")
        logger.info("Expected format: CSV with columns: time/timestamp, open, high, low, close, volume")
        return 1
    
    # Create checkpoint directory
    checkpoint_dir = Path(config_dict["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = checkpoint_dir / f"training_config_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"💾 Configuration saved to {config_path}")
    
    # Initialize trainer
    logger.info("🚀 Initializing SWT Trainer...")
    try:
        config = SWTTrainingConfig(**config_dict)
        trainer = SWTTrainer(config)
        
        # Check for existing checkpoint to resume
        latest_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("episode_*.pth"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                logger.info(f"📂 Found checkpoint: {latest_checkpoint}")
                
                response = input("Resume from checkpoint? (y/n): ")
                if response.lower() == 'y':
                    trainer.load_checkpoint(str(latest_checkpoint))
                    logger.info("✅ Resumed from checkpoint")
        
        # Start training
        logger.info("🎯 Starting training...")
        logger.info(f"   Episodes: {config.num_episodes}")
        logger.info(f"   Session length: {config.environment['session_hours']} hours")
        logger.info(f"   Reward: {config.environment['reward_type'].upper()}")
        logger.info(f"   Features: 137 (128 WST + 9 Position)")
        
        trainer.train()
        
        logger.info("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())