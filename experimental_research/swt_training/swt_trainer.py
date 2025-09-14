#!/usr/bin/env python3
"""
SWT Enhanced Stochastic MuZero Trainer
Complete training system integrating WST, Stochastic MuZero, and AMDDP5 rewards
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import json
import logging
import time
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import copy

# SWT imports
from swt_models.swt_market_encoder import SWTMarketStateEncoder
from swt_models.swt_stochastic_networks import create_swt_stochastic_muzero_network, SWTStochasticMuZeroConfig
from swt_core.swt_mcts import create_swt_stochastic_mcts, SWTMCTSConfig
from swt_environments.swt_forex_env import create_swt_forex_environment
from swt_core.swt_checkpoint_manager import create_swt_checkpoint_manager, SWTCheckpointMetadata, SWTCheckpointConfig
from swt_core.swt_curriculum_learning import create_swt_curriculum_learning, CurriculumConfig, CurriculumStage
from swt_core.swt_quality_buffer import SWTQualityExperienceBuffer, SWTExperience
from swt_core.swt_session_manager import create_swt_session_manager, SWTSessionManager, SWTSessionConfig
from swt_core.simple_csv_session_manager import SimpleCsvSessionManager
# Optional visualization module - may not be available in minimal training containers
try:
    from swt_visualizations.checkpoint_trade_visualizer import create_trade_chart
    HAS_VISUALIZATION = True
except ImportError:
    create_trade_chart = None
    HAS_VISUALIZATION = False

logger = logging.getLogger(__name__)


@dataclass
class SWTTrainingConfig:
    """Complete training configuration for SWT system"""
    
    # Model parameters (from config file)
    wst: Dict[str, Any] = None
    fusion: Dict[str, Any] = None
    stochastic_muzero: Dict[str, Any] = None
    training: Dict[str, Any] = None
    reward: Dict[str, Any] = None
    reward_profiles: Dict[str, Any] = None
    environment: Dict[str, Any] = None
    checkpoints: Dict[str, Any] = None
    
    # Legacy model parameters (for backwards compatibility)
    market_encoder_config: Dict[str, Any] = None
    muzero_config: Dict[str, Any] = None
    mcts_config: Dict[str, Any] = None
    
    # WST parameters (extracted from config)
    wst_j: int = 2
    wst_q: int = 6
    wst_backend: str = 'kymatio'
    
    # Network architecture parameters
    hidden_dim: int = 256
    latent_dim: int = 16
    num_actions: int = 4
    value_support_size: int = 601
    
    # Market configuration
    price_series_length: int = 256
    position_features_dim: int = 9
    
    # Training parameters
    num_episodes: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    gradient_clip: float = 10.0
    
    # Experience replay (Quality Buffer)
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    eviction_batch: int = 2000
    min_quality_threshold: float = 0.1
    num_unroll_steps: int = 5
    td_steps: int = 10
    
    # Training schedule
    train_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    temperature_schedule: bool = True
    
    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    
    # Environment
    data_path: str = None
    reward_type: str = 'pure_pips'
    
    # Production Features
    partial_reset: bool = False
    enable_curriculum: bool = False
    enable_checkpoints: bool = True
    save_frequency: int = 25
    trades_per_checkpoint: int = 1000
    
    # Logging
    log_level: str = 'INFO'
    save_dir: str = 'swt_checkpoints'
    experiment_name: str = 'swt_stochastic_muzero'
    
    def __post_init__(self):
        """Set default configurations if not provided"""
        # Handle nested config structure from JSON
        if self.wst is None:
            self.wst = {}
        if self.fusion is None:
            self.fusion = {}
        if self.stochastic_muzero is None:
            self.stochastic_muzero = {}
        if self.training is None:
            self.training = {}
        if self.reward is None:
            self.reward = {}
        if self.environment is None:
            self.environment = {}
        if self.checkpoints is None:
            self.checkpoints = {}
            
        # Legacy backwards compatibility
        if self.market_encoder_config is None:
            self.market_encoder_config = {}
        if self.muzero_config is None:
            self.muzero_config = {}
        if self.mcts_config is None:
            self.mcts_config = {}
            
        # Extract WST parameters from nested config
        if self.wst:
            if 'J' in self.wst:
                self.wst_j = self.wst['J']
            if 'Q' in self.wst:
                self.wst_q = self.wst['Q']
            if 'input_length' in self.wst:
                self.price_series_length = self.wst['input_length']
                
        # Extract network parameters from stochastic_muzero config and populate muzero_config
        if self.stochastic_muzero:
            # Map stochastic_muzero to muzero_config for SWTStochasticMuZeroConfig
            self.muzero_config.update({
                'hidden_dim': self.stochastic_muzero.get('latent_dim', 256),
                'num_actions': self.stochastic_muzero.get('action_dim', 4),
                'latent_z_dim': self.stochastic_muzero.get('z_dim', 16),
                # Remove parameters not in SWTStochasticMuZeroConfig
                # num_simulations, discount, dirichlet_alpha, exploration_fraction are MCTS params, not network params
            })
            
            # Also extract for legacy attributes
            if 'hidden_dim' in self.stochastic_muzero:
                self.hidden_dim = self.stochastic_muzero['hidden_dim']
            elif 'latent_dim' in self.stochastic_muzero:
                self.hidden_dim = self.stochastic_muzero['latent_dim']
            if 'z_dim' in self.stochastic_muzero:
                self.latent_dim = self.stochastic_muzero['z_dim']
            if 'action_dim' in self.stochastic_muzero:
                self.num_actions = self.stochastic_muzero['action_dim']
                
        # Extract market config from environment
        if self.environment:
            if 'price_series_length' in self.environment:
                self.price_series_length = self.environment['price_series_length']
            
        # Override training parameters from nested config if available
        if self.training:
            if 'batch_size' in self.training:
                self.batch_size = self.training['batch_size']
            if 'learning_rate' in self.training:
                self.learning_rate = self.training['learning_rate']
            if 'weight_decay' in self.training:
                self.weight_decay = self.training['weight_decay']
            if 'gradient_clip' in self.training:
                self.gradient_clip = self.training['gradient_clip']
            if 'buffer_size' in self.training:
                self.buffer_size = self.training['buffer_size']
            if 'num_unroll_steps' in self.training:
                self.num_unroll_steps = self.training['num_unroll_steps']
            if 'td_steps' in self.training:
                self.td_steps = self.training['td_steps']
            if 'num_epochs' in self.training:
                self.num_episodes = self.training['num_epochs']
            if 'save_interval' in self.training:
                self.save_interval = self.training['save_interval']
            if 'eval_interval' in self.training:
                self.eval_interval = self.training['eval_interval']


# Using production-grade SWTQualityExperienceBuffer with smart eviction


class SWTStochasticMuZeroTrainer:
    """
    Complete SWT Stochastic MuZero training system
    Integrates WST market encoding, stochastic planning, and AMDDP5 rewards
    """
    
    def __init__(self, config: SWTTrainingConfig):
        self.config = config
        
        # Set up logging
        self._setup_logging()
        
        # Set up save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Track total trades for checkpoint triggering
        self.total_trades_completed = 0
        self.trades_per_checkpoint = getattr(config, 'trades_per_checkpoint', 1000)
        
        # Initialize models
        self._initialize_models()
        
        # Initialize training components
        self._initialize_training()
        
        # Initialize CSV-based random session sampling
        logger.info(f"📊 Using CSV-based random 6-hour sessions with gap detection")
        self.use_session_manager = False  # Use direct CSV sampling
        self.csv_data_path = self.config.data_path
        self._initialize_csv_session_sampling()
        
        # Create reward config from profiles
        if hasattr(self.config, 'reward_profiles') and self.config.reward_profiles and self.config.reward_type in self.config.reward_profiles:
            reward_config = self.config.reward_profiles[self.config.reward_type].copy()
        else:
            reward_config = {
                'type': self.config.reward_type,
                'drawdown_penalty': 0.01,
                'profit_protection': True,
                'min_protected_reward': 0.001,
                'reassign_on_trade_complete': True,
                'use_final_pnl': True
            }
        
        self.env_config_template = {
            'reward': reward_config,
            'environment': {
                'spread_pips': 4.0,
                'pip_value': 0.01
            }
        }
        
        # Initialize production-grade quality experience buffer
        # Use multiprocessing Manager for shared buffer across processes
        self.mp_manager = multiprocessing.Manager()
        self.shared_experience_queue = self.mp_manager.Queue(maxsize=config.buffer_size * 2)
        
        # Initialize shared WST cache for multiprocessing speedup
        from swt_models.swt_wavelet_scatter import initialize_wst_cache
        initialize_wst_cache(self.mp_manager)
        
        self.experience_buffer = SWTQualityExperienceBuffer(
            capacity=config.buffer_size,
            eviction_batch=config.eviction_batch,
            min_quality_threshold=config.min_quality_threshold,
            diversity_sampling=True
        )
        
        # Multiprocessing configuration
        self.num_processes = config.multiprocessing.get('num_workers', 8) if hasattr(config, 'multiprocessing') else 8
        self.gradient_aggregation_queue = self.mp_manager.Queue()
        self.shared_model_state = self.mp_manager.dict()
        self.process_pool = None
        
        # Initialize checkpoint manager
        if config.enable_checkpoints:
            # Extract checkpoint config from JSON if available
            checkpoint_settings = config.checkpoints if hasattr(config, 'checkpoints') and config.checkpoints else {}
            checkpoint_config = SWTCheckpointConfig(
                checkpoint_dir=str(self.save_dir / 'checkpoints'),
                save_frequency=checkpoint_settings.get('save_frequency', config.save_frequency),
                keep_best_count=1,  # Keep only 1 best checkpoint
                keep_recent_count=2,  # Keep only last 2 recent checkpoints
                emergency_backup_frequency=100,
                auto_cleanup=True  # Enable cleanup to maintain only best + last 2
            )
            self.checkpoint_manager = create_swt_checkpoint_manager(asdict(checkpoint_config))
        else:
            self.checkpoint_manager = None
            
        # Initialize curriculum learning
        if config.enable_curriculum:
            curriculum_config = CurriculumConfig(
                enable_curriculum=True,
                stage_transition_episodes=100,
                performance_threshold=0.6,
                reassignment_frequency=50,
                adaptive_transitions=True
            )
            self.curriculum = create_swt_curriculum_learning(curriculum_config.__dict__)
        else:
            self.curriculum = None
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.loss_history = []
        
        logger.info(f"🚀 SWT Stochastic MuZero Trainer initialized")
        logger.info(f"   Target episodes: {config.num_episodes}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Save directory: {self.save_dir}")
        logger.info(f"   Quality buffer: {len(self.experience_buffer)} / {self.experience_buffer.capacity}")
        logger.info(f"   Checkpoints: {'✅' if self.checkpoint_manager else '❌'}")
        logger.info(f"   Curriculum: {'✅' if self.curriculum else '❌'}")
        
        # Handle partial reset if requested (after all components are initialized)
        if hasattr(config, 'partial_reset') and config.partial_reset:
            self._handle_partial_reset()
        
    def _handle_partial_reset(self):
        """Handle partial reset: keep networks, clear experience buffer, switch to test data"""
        logger.info("🔄 Executing PARTIAL RESET...")
        
        # Load best checkpoint (Episode 75)
        best_checkpoint_path = "swt_checkpoints/checkpoints/best_checkpoint.pth"
        
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(best_checkpoint_path)
            
            # Restore networks from best checkpoint
            if 'market_encoder_state' in checkpoint:
                self.market_encoder.load_state_dict(checkpoint['market_encoder_state'])
                logger.info("✅ Restored market encoder from Episode 75")
                
            if 'muzero_network_state' in checkpoint:
                self.muzero_network.load_state_dict(checkpoint['muzero_network_state'])
                logger.info("✅ Restored MuZero networks from Episode 75")
                
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.info("✅ Restored optimizer state from Episode 75")
                
            # Clear experience buffer (fresh start)
            self.experience_buffer.clear()
            logger.info("🗑️ Cleared experience buffer")
            
            # Initialize session manager for session-based training
            # This will be properly initialized in _initialize_session_manager()
            logger.info("📊 Using session-based training mode")
            
            # Reset episode counter for fresh exploration schedule
            self.episode_rewards = []
            self.episode_lengths = []
            self.training_losses = []
            self.loss_history = []
            
            logger.info("🚀 PARTIAL RESET completed - ready for 1000-episode exploration schedule")
            
        except Exception as e:
            logger.error(f"❌ Partial reset failed: {e}")
            raise
        
    def resume_training_from_checkpoint(self, checkpoint_path: str = None) -> int:
        """
        Resume training from a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file, or None for best checkpoint
            
        Returns:
            Episode number to resume from
        """
        
        if not self.checkpoint_manager:
            logger.warning("⚠️ No checkpoint manager - cannot resume training")
            return 0
            
        try:
            # Get resumption info
            resume_info = self.checkpoint_manager.get_resumption_info()
            
            if not resume_info['can_resume']:
                logger.info("ℹ️ No checkpoints found - starting fresh training")
                return 0
                
            logger.info(f"📂 Available checkpoints: {resume_info['available_checkpoints']}")
            logger.info(f"📈 Latest episode: {resume_info['latest_episode']}")
            logger.info(f"🏆 Best reward: {resume_info['best_reward']:.2f}")
            
            # Load checkpoint
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # Restore model states
            if 'market_encoder_state' in checkpoint:
                self.market_encoder.load_state_dict(checkpoint['market_encoder_state'])
                logger.info("✅ Market encoder state restored")
                
            if 'muzero_network_state' in checkpoint:
                self.muzero_network.load_state_dict(checkpoint['muzero_network_state'])
                logger.info("✅ MuZero network state restored")
                
            # Restore training state
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.info("✅ Optimizer state restored")
                
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.info("✅ Scheduler state restored")
                
            # Restore trade counter
            if 'total_trades_completed' in checkpoint:
                self.total_trades_completed = checkpoint['total_trades_completed']
                logger.info(f"✅ Trade counter restored: {self.total_trades_completed} trades")
                
            # Restore experience buffer
            if 'experience_buffer' in checkpoint and checkpoint['experience_buffer']:
                buffer_data = checkpoint['experience_buffer']
                logger.info(f"📦 Restoring {len(buffer_data)} experiences from checkpoint")
                
                # Clear current buffer and restore from checkpoint
                self.experience_buffer.clear()
                
                for exp_data in buffer_data:
                    # Convert data back to tensors if needed
                    obs_history = exp_data['observation_history']
                    if not isinstance(obs_history, torch.Tensor):
                        obs_history = torch.tensor(obs_history)
                    
                    market_prices = exp_data['market_prices'] 
                    if not isinstance(market_prices, torch.Tensor):
                        market_prices = torch.tensor(market_prices)
                        
                    position_features = exp_data['position_features']
                    if not isinstance(position_features, torch.Tensor):
                        position_features = torch.tensor(position_features)
                    
                    experience = SWTExperience(
                        observation_history=obs_history,
                        market_prices=market_prices,
                        position_features=position_features,
                        action=exp_data['action'],
                        reward=exp_data['reward'],
                        value_target=exp_data['value_target'],
                        policy_target=exp_data['policy_target'],
                        next_observation={'market_prices': market_prices, 'position_features': position_features},
                        done=exp_data.get('done', False)
                    )
                    # Set additional fields manually to avoid __init__ issues
                    experience.quality_score = exp_data.get('quality_score', 1.0)
                    self.experience_buffer.add(experience)
                    
                logger.info(f"✅ Experience buffer restored: {len(self.experience_buffer)} experiences")
                
            # Restore training metrics
            if 'training_metrics' in checkpoint:
                metrics = checkpoint['training_metrics']
                
                self.episode_rewards = metrics.get('episode_rewards', [])
                self.episode_lengths = metrics.get('episode_lengths', [])
                self.training_losses = metrics.get('training_losses', [])
                self.loss_history = metrics.get('loss_history', [])
                
                # Restore win rate history
                win_rate_data = metrics.get('win_rate_history', {})
                self.win_rate_history = deque(
                    [win_rate_data.get(str(i), 0.0) for i in range(len(win_rate_data))],
                    maxlen=100
                )
                
                logger.info(f"✅ Training metrics restored:")
                logger.info(f"   Episodes: {len(self.episode_rewards)}")
                logger.info(f"   Avg reward: {np.mean(self.episode_rewards[-10:]):.2f}")
                logger.info(f"   Win rate: {np.mean(list(self.win_rate_history)):.1%}")
                
            # Get episode to resume from
            resume_episode = checkpoint.get('episode', 0) + 1  # Resume from next episode
            
            logger.info(f"🔄 Training resumption successful")
            logger.info(f"   Resuming from episode: {resume_episode}")
            logger.info(f"   Previous best reward: {resume_info['best_reward']:.2f}")
            logger.info(f"   Experience buffer size: {len(self.experience_buffer)}")
            
            return resume_episode
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            logger.info("🆕 Starting fresh training")
            return 0
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _initialize_models(self):
        """Initialize all neural network models"""
        
        # Market encoder (WST + position fusion)
        # Check if precomputed WST should be used
        precomputed_wst_path = getattr(self.config, 'precomputed_wst_path', None)
        if precomputed_wst_path:
            logger.info(f"🗃️ Using precomputed WST features from: {precomputed_wst_path}")
        else:
            logger.info(f"🔄 Using on-the-fly WST computation")
            
        self.market_encoder = SWTMarketStateEncoder(
            config_dict=self.config.market_encoder_config,
            precomputed_wst_path=precomputed_wst_path
        )
        self.market_encoder = self.market_encoder.to(self.device)
        
        # Stochastic MuZero network
        # Map config parameters correctly
        muzero_params = self.config.muzero_config.copy()
        if 'latent_dim' in muzero_params:
            muzero_params['hidden_dim'] = muzero_params.pop('latent_dim')
        if 'action_dim' in muzero_params:
            muzero_params['num_actions'] = muzero_params.pop('action_dim')
        if 'z_dim' in muzero_params:
            muzero_params['latent_z_dim'] = muzero_params.pop('z_dim')
            
        muzero_config = SWTStochasticMuZeroConfig(**muzero_params)
        self.muzero_network = create_swt_stochastic_muzero_network(self.config.muzero_config)
        self.muzero_network = self.muzero_network.to(self.device)
        
        # MCTS
        mcts_config = SWTMCTSConfig(**self.config.mcts_config)
        self.mcts = create_swt_stochastic_mcts(self.muzero_network, self.config.mcts_config)
        
        logger.info("🧠 Models initialized:")
        logger.info("   ✅ WST Market Encoder")
        logger.info("   ✅ Stochastic MuZero Network (5 networks)")
        logger.info("   ✅ Stochastic MCTS")
        
    def _initialize_training(self):
        """Initialize training components"""
        
        # Combine parameters from both models
        all_parameters = (
            list(self.market_encoder.parameters()) +
            list(self.muzero_network.parameters())
        )
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            all_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000, 
            gamma=0.9
        )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.reward_loss_fn = nn.MSELoss()
        
        logger.info("⚙️ Training components initialized")
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = -float('inf')
        self.win_rate_history = deque(maxlen=100)  # Track win rate for curriculum
        
    def _initialize_csv_session_sampling(self):
        """Initialize CSV data for random 6-hour session sampling with gap detection"""
        if self.config.data_path is None:
            raise ValueError("Data path must be provided in config")
        
        # Smart CSV metadata loading - only get file info, not all data
        logger.info(f"📁 Analyzing CSV file: {self.config.data_path}")
        
        # Get total rows without loading entire file
        with open(self.config.data_path, 'r') as f:
            self.total_csv_rows = sum(1 for _ in f) - 1  # Subtract header
        
        # Sample first few rows to understand structure
        sample_data = pd.read_csv(self.config.data_path, nrows=1000)
        
        # Determine datetime column
        self.datetime_column = None
        if 'timestamp' in sample_data.columns:
            self.datetime_column = 'timestamp'
        elif 'datetime' in sample_data.columns:
            self.datetime_column = 'datetime'
        
        logger.info(f"📊 CSV metadata loaded: {self.total_csv_rows:,} total rows")
        logger.info(f"📅 Datetime column: {self.datetime_column or 'artificial'}")
        
        # Store file path for per-session loading
        self.csv_file_path = self.config.data_path
        self.csv_data = None  # Don't store entire dataset
        
        # Calculate session parameters
        session_hours = self.config.environment.get('session_hours', 6)
        max_gap_minutes = self.config.environment.get('max_gap_minutes', 10)
        self.session_bars = int(session_hours * 60)  # 6 hours = 360 bars (1min data)
        self.max_gap_minutes = max_gap_minutes
        
        # Use simple approach: any start index that leaves room for full session
        # Gap validation will happen during session loading (per-session basis)
        max_start_index = self.total_csv_rows - self.session_bars - 1
        
        # Generate potential session starts (every hour to avoid too many options)
        self.valid_session_starts = list(range(0, max_start_index, 60))  # Every hour
        
        # Simplified weekend boundary sessions (every day as potential boundary) 
        self.weekend_boundary_sessions = list(range(0, max_start_index, 1440))  # Every day
        
        # Combine with weekend weighting
        all_sessions = self.valid_session_starts + (self.weekend_boundary_sessions * 3)
        self.weighted_session_starts = all_sessions
        
        logger.info(f"📊 Streaming CSV session sampling initialized:")
        logger.info(f"   Total bars: {self.total_csv_rows:,}")
        logger.info(f"   Session length: {session_hours} hours ({self.session_bars} bars)")
        logger.info(f"   Max gap tolerance: {max_gap_minutes} minutes")
        logger.info(f"   Valid session starts: {len(self.valid_session_starts):,}")
        logger.info(f"   Weekend boundary sessions: {len(self.weekend_boundary_sessions):,}")
        logger.info(f"   Total weighted sessions: {len(self.weighted_session_starts):,}")
        
    def _find_valid_session_starts_DISABLED(self) -> List[int]:
        """Find all valid session start indices that won't have gaps >10min or weekend breaks"""
        valid_starts = []
        
        for start_idx in range(0, len(self.csv_data) - self.session_bars, 60):  # Check every hour
            end_idx = start_idx + self.session_bars
            
            if end_idx >= len(self.csv_data):
                break
                
            # Check for gaps in this session window
            session_data = self.csv_data.iloc[start_idx:end_idx]
            
            if self._has_valid_session_data(session_data):
                valid_starts.append(start_idx)
                
        return valid_starts
        
    def _has_valid_session_data(self, session_data: pd.DataFrame) -> bool:
        """Check if session data has no gaps >10min and no weekend breaks"""
        if len(session_data) < self.session_bars * 0.9:  # Must have at least 90% of expected bars
            return False
            
        # Check for time gaps
        time_diffs = session_data['datetime'].diff().dt.total_seconds() / 60  # Minutes
        max_gap = time_diffs.max()
        
        if max_gap > self.max_gap_minutes:
            return False
            
        # Check for weekend breaks (Friday close to Monday open)
        for i in range(1, len(session_data)):
            current_time = session_data.iloc[i]['datetime']
            prev_time = session_data.iloc[i-1]['datetime']
            
            # If gap spans weekend (Friday evening to Monday morning)
            if (current_time.weekday() == 0 and  # Monday
                prev_time.weekday() == 4 and    # Friday
                (current_time - prev_time).total_seconds() > 3600):  # >1 hour gap
                return False
                
        return True
        
    def _find_weekend_boundary_sessions_DISABLED(self) -> List[int]:
        """Find sessions that capture the first/last 6 hours before/after weekends"""
        boundary_sessions = []
        
        # Find Friday closes and Monday opens
        for i in range(1, len(self.csv_data)):
            current_time = self.csv_data.iloc[i]['datetime']
            prev_time = self.csv_data.iloc[i-1]['datetime']
            
            # Friday close to Monday open (weekend gap)
            if (current_time.weekday() == 0 and  # Monday
                prev_time.weekday() == 4):       # Friday
                
                # Last 6 hours before weekend (Friday close)
                friday_close_idx = i - 1
                friday_session_start = max(0, friday_close_idx - self.session_bars)
                if friday_session_start >= 0 and friday_close_idx < len(self.csv_data):
                    boundary_sessions.append(friday_session_start)
                    
                # First 6 hours after weekend (Monday open)  
                monday_open_idx = i
                monday_session_end = min(len(self.csv_data), monday_open_idx + self.session_bars)
                if monday_session_end <= len(self.csv_data):
                    boundary_sessions.append(monday_open_idx)
                    
        return boundary_sessions
        
    def _sample_random_session_data(self) -> Optional[pd.DataFrame]:
        """Sample a random 6-hour session with streaming CSV loading"""
        if not self.weighted_session_starts:
            logger.warning("No valid session starts available")
            return None
            
        # Pick random start index from weighted list (weekend boundaries have 3x chance)
        start_idx = np.random.choice(self.weighted_session_starts)
        end_idx = start_idx + self.session_bars
        
        # Load only required rows for this session (much faster!)
        session_data = pd.read_csv(
            self.csv_file_path, 
            skiprows=range(1, start_idx + 1),  # Skip to start_idx (keeping header)
            nrows=self.session_bars  # Load only session_bars rows
        )
        
        # Add datetime column if needed
        if self.datetime_column and self.datetime_column in session_data.columns:
            session_data['datetime'] = pd.to_datetime(session_data[self.datetime_column])
        elif self.datetime_column is None:
            # Create artificial datetime starting from start_idx
            session_data['datetime'] = pd.date_range(
                start='2020-01-01', 
                periods=len(session_data), 
                freq='1min'
            )
        
        # Determine session type
        is_weekend_boundary = start_idx in self.weekend_boundary_sessions
        session_type = "weekend_boundary" if is_weekend_boundary else "regular"
        
        # Store session metadata AND actual data for chart generation
        self.current_session_data = {
            'session_id': f"csv_session_{start_idx}_{end_idx}",
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': session_data.iloc[0]['datetime'],
            'end_time': session_data.iloc[-1]['datetime'],
            'total_bars': len(session_data),
            'duration_hours': self.config.environment.get('session_hours', 6),
            'data_split': 'random_session',
            'session_type': session_type,
            'data': session_data  # Store actual CSV data for chart generation
        }
        
        session_emoji = "🏁" if is_weekend_boundary else "📊"
        logger.info(f"{session_emoji} {session_type.title()} session: {start_idx}-{end_idx} ({len(session_data)} bars) [{self.current_session_data['start_time']} - {self.current_session_data['end_time']}]")
        
        return session_data
    
    def _initialize_session_manager(self):
        """Initialize session manager for 6-hour episodes with 1-hour walk-forward"""
        if self.config.data_path is None:
            raise ValueError("Data path must be provided in config")
        
        # Create session manager configuration
        session_config = SWTSessionConfig(
            session_hours=6.0,  # 6-hour sessions (360 M1 bars) - as per config
            walk_forward_hours=1.0,  # 1-hour walk forward (60 M1 bars)
            max_gap_minutes=10,
            min_session_bars=300,  # Minimum 5 hours of data
            train_ratio=0.80,
            validation_ratio=0.10,
            test_ratio=0.10
        )
        
        # Initialize session manager - use SimpleCsvSessionManager for CSV files
        from pathlib import Path
        data_path = Path(self.config.data_path)
        if data_path.suffix == '.csv':
            self.session_manager = SimpleCsvSessionManager(data_path=str(data_path), config=session_config)
        else:
            self.session_manager = SWTSessionManager(db_path=self.config.data_path, config=session_config)
        
        # Store environment config template using reward profiles
        if hasattr(self.config, 'reward_profiles') and self.config.reward_profiles and self.config.reward_type in self.config.reward_profiles:
            reward_profile = self.config.reward_profiles[self.config.reward_type]
            reward_config = reward_profile.copy()
        else:
            # Fallback configuration
            reward_config = {
                'type': self.config.reward_type,
                'drawdown_penalty': 0.01 if self.config.reward_type == 'amddp1' else 0.05,
                'profit_protection': True,
                'min_protected_reward': 0.001
            }
        
        self.env_config_template = {
            'reward': reward_config
        }
        
        logger.info(f"🗓️ Session manager initialized")
        logger.info(f"   6-hour sessions with 1-hour walk-forward")
        
        # Get session statistics
        stats = self.session_manager.get_session_stats()
        train_info = stats.get('train_split', {})
        test_info = stats.get('test_split', {})
        
        logger.info(f"   Training sessions: {train_info.get('sessions_available', 0):,}")
        logger.info(f"   Test sessions: {test_info.get('sessions_available', 0):,}")
        
        # Set to training split initially
        self.session_manager.set_split('train')
        self.current_environment = None
    
    def _create_episode_environment(self, episode: int, split: str = None) -> Optional['SWTForexEnvironment']:
        """Create environment for specific episode using random 6-hour CSV sessions"""
        try:
            # Sample random 6-hour session from CSV
            session_data = self._sample_random_session_data()
            
            if session_data is None:
                logger.warning(f"No valid session data available for episode {episode}")
                return None
            
            # Create environment with session data
            env = create_swt_forex_environment(
                session_data=session_data,
                config_dict=self.env_config_template
            )
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment for episode {episode}: {e}")
            return None
        
    def _preprocess_observation(self, obs: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess environment observation for models"""
        
        # Convert to tensors
        market_prices = torch.tensor(obs['market_prices'], dtype=torch.float32).unsqueeze(0)
        position_features = torch.tensor(obs['position_features'], dtype=torch.float32).unsqueeze(0)
        
        # Move to device
        market_prices = market_prices.to(self.device)
        position_features = position_features.to(self.device)
        
        return market_prices, position_features
        
    def _get_fused_observation(self, market_prices: torch.Tensor, position_features: torch.Tensor) -> torch.Tensor:
        """Get fused observation from market encoder"""
        with torch.no_grad():
            fused_obs = self.market_encoder(market_prices, position_features)
        return fused_obs
        
    def _select_action(self, fused_obs: torch.Tensor, obs_history: Optional[torch.Tensor] = None, 
                      temperature: float = 1.0, episode: int = 0) -> Tuple[int, np.ndarray, float]:
        """Select action using MCTS with epsilon-greedy exploration"""
        
        # 1000-episode epsilon-greedy exploration schedule
        exploration_rate = max(0.1, 1.0 - (episode / 1000))
        
        # Store exploration rate for checkpoint metadata
        self.current_exploration_rate = exploration_rate
        
        # Force random exploration if in exploration phase
        if np.random.random() < exploration_rate:
            # Random action for exploration
            action = np.random.randint(0, self.config.num_actions)
            # Create uniform probability distribution for random action
            action_probs = np.ones(self.config.num_actions) / self.config.num_actions
            root_value = 0.0  # Neutral value for exploration
            
            if episode > 0 and episode % 100 == 0:  # Log exploration rate periodically
                logger.info(f"🎲 Exploration active: {exploration_rate:.1%} chance (Episode {episode})")
            
            return action, action_probs, root_value
        
        # Use MCTS for exploitation
        action_probs, search_stats, root_value = self.mcts.run(
            fused_obs,
            obs_history=obs_history,
            add_exploration_noise=True,
            override_temperature=temperature
        )
        
        # Sample action from probabilities
        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)
            
        return action, action_probs, root_value
        
    def _calculate_temperature(self, episode: int) -> float:
        """Calculate temperature for action selection"""
        if not self.config.temperature_schedule:
            return 1.0
            
        # Decay temperature over training
        if episode < 100:
            return 1.0
        elif episode < 500:
            return 0.5
        else:
            return 0.1
            
    def train_episode(self, episode: int) -> Dict[str, float]:
        """Train a single episode using session-based data"""
        
        # Create environment for this episode
        env = self._create_episode_environment(episode)
        if env is None:
            logger.error(f"Failed to create environment for episode {episode}")
            return {
                'episode_reward': 0.0,
                'episode_length': 0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl_pips': 0.0,
                'temperature': 1.0
            }
        
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        obs_history = deque(maxlen=4)  # For uncertainty encoding
        
        # Current temperature
        temperature = self._calculate_temperature(episode)
        
        while not done:
            # Preprocess observation
            market_prices, position_features = self._preprocess_observation(obs)
            fused_obs = self._get_fused_observation(market_prices, position_features)
            
            # Add to observation history
            obs_history.append(fused_obs.clone())
            
            # Select action
            if len(obs_history) >= 2:
                history_tensor = torch.stack(list(obs_history)).unsqueeze(0)  # (1, T, 128)
            else:
                history_tensor = None
                
            action, action_probs, root_value = self._select_action(
                fused_obs, history_tensor, temperature, episode
            )
            
            # Execute action
            next_obs, reward, done, truncated, env_info = env.step(action)
            
            # Store experience
            if len(obs_history) >= 2:
                experience = SWTExperience(
                    observation_history=history_tensor.squeeze(0).cpu(),
                    market_prices=market_prices.squeeze(0).cpu(),
                    position_features=position_features.squeeze(0).cpu(),
                    action=action,
                    reward=reward,
                    value_target=root_value,  # Could be improved with TD targets
                    policy_target=action_probs,
                    next_observation={
                        'market_prices': torch.tensor(next_obs['market_prices']),
                        'position_features': torch.tensor(next_obs['position_features'])
                    },
                    done=done
                )
                self.experience_buffer.add(experience)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Training step
            if (episode_length % self.config.train_interval == 0 and 
                len(self.experience_buffer) >= self.config.min_buffer_size):
                loss_dict = self._training_step()
                self.training_losses.append(loss_dict['total_loss'])
                
        # CRITICAL FIX: Handle open positions at episode end
        if hasattr(env, 'position') and (env.position.is_long or env.position.is_short):
            logger.info(f"🔧 FORCED CLOSE: Open position at episode end - forcing close for reward reassignment")
            # Force close to get actual trade outcome
            close_obs, close_reward, force_done, _, close_info = env.step(3)  # CLOSE action
            episode_reward += close_reward
            episode_length += 1
            
            # Store the forced close experience
            if len(obs_history) >= 2:
                market_prices, position_features = self._preprocess_observation(obs)
                fused_obs = self._get_fused_observation(market_prices, position_features)
                history_tensor = torch.stack(list(obs_history)).unsqueeze(0)
                
                force_close_experience = SWTExperience(
                    observation_history=history_tensor.squeeze(0).cpu(),
                    market_prices=market_prices.squeeze(0).cpu(),
                    position_features=position_features.squeeze(0).cpu(),
                    action=3,  # CLOSE action
                    reward=close_reward,
                    value_target=0.0,  # End of episode
                    policy_target=np.array([0.0, 0.0, 0.0, 1.0]),  # Force CLOSE
                    next_observation={
                        'market_prices': torch.tensor(close_obs['market_prices']),
                        'position_features': torch.tensor(close_obs['position_features'])
                    },
                    done=True
                )
                self.experience_buffer.add(force_close_experience)
                logger.info(f"   Forced close reward: {close_reward:.3f}, final episode reward: {episode_reward:.3f}")
        
        # Episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Get trading statistics (includes all trades now, including forced closes)
        trade_stats = env.get_trade_statistics()
        
        # Post-session reward reassignment for all completed trades (including forced closes)
        if len(env.completed_trades) > 0:
            self._reassign_session_trade_rewards(env.completed_trades)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'total_trades': trade_stats['total_trades'],
            'win_rate': trade_stats['win_rate'],
            'avg_pnl_pips': trade_stats['avg_pnl_pips'],
            'temperature': temperature,
            'env_trades': env.completed_trades  # Add trade data for checkpoint
        }
        
    def _training_step(self) -> Dict[str, float]:
        """Perform a training step with quality buffer sampling"""
        
        # Sample batch using prioritized sampling
        batch = self.experience_buffer.sample(
            self.config.batch_size, 
            prioritized=True
        )
        if len(batch) == 0:
            return {'total_loss': 0.0}
            
        # Prepare batch data
        batch_market = torch.stack([exp.market_prices for exp in batch]).to(self.device)
        batch_position = torch.stack([exp.position_features for exp in batch]).to(self.device)
        
        # Handle variable length observation histories by padding
        max_seq_len = max(exp.observation_history.shape[0] for exp in batch)
        batch_history_list = []
        for exp in batch:
            hist = exp.observation_history
            if hist.shape[0] < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(max_seq_len - hist.shape[0], hist.shape[1], hist.shape[2])
                hist = torch.cat([hist, padding], dim=0)
            batch_history_list.append(hist)
        batch_history = torch.stack(batch_history_list).to(self.device)
        
        batch_actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        batch_value_targets = torch.tensor([exp.value_target for exp in batch], dtype=torch.float32).to(self.device)
        batch_policy_targets = torch.stack([torch.tensor(exp.policy_target) for exp in batch]).to(self.device)
        
        # Forward pass through market encoder
        batch_fused = self.market_encoder(batch_market.unsqueeze(1), batch_position)
        
        # Forward pass through MuZero
        initial_outputs = self.muzero_network.initial_inference(batch_fused)
        
        # Compute losses
        value_loss = self.value_loss_fn(
            self._softmax_to_scalar(initial_outputs['value_distribution']), 
            batch_value_targets
        )
        
        policy_loss = self.policy_loss_fn(initial_outputs['policy_logits'], batch_actions)
        
        # KL loss for stochastic latent
        if len(batch_history.shape) == 3:  # Check if we have history
            latent_z, mu, logvar = self.muzero_network.encode_uncertainty(batch_history)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl_loss = torch.tensor(0.0, device=self.device)
            
        # Total loss
        total_loss = (
            self.config.value_loss_weight * value_loss +
            self.config.policy_loss_weight * policy_loss +
            self.config.kl_loss_weight * kl_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.market_encoder.parameters()) + list(self.muzero_network.parameters()),
            self.config.gradient_clip
        )
        self.optimizer.step()
        
        # Update TD errors in buffer for quality recalculation
        if hasattr(self.experience_buffer, 'update_td_errors'):
            td_errors = [value_loss.item()] * len(batch)
            self.experience_buffer.update_td_errors(batch, td_errors)
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item()
        }
        
    def _softmax_to_scalar(self, distribution: torch.Tensor) -> torch.Tensor:
        """Convert distributional prediction to scalar value"""
        # Simple implementation - could use proper support conversion
        return torch.mean(distribution, dim=-1)
        
    def train(self, resume_from_checkpoint: bool = True) -> Dict[str, List[float]]:
        """Main training loop with curriculum learning and production features"""
        
        # Check for training resumption
        start_episode = 0
        if resume_from_checkpoint and self.checkpoint_manager:
            start_episode = self.resume_training_from_checkpoint()
            if start_episode > 0:
                logger.info(f"🔄 Resuming training from episode {start_episode}")
            
        logger.info(f"🚀 Starting SWT training for {self.config.num_episodes} episodes")
        if self.curriculum:
            logger.info(f"📚 Curriculum learning enabled - Starting stage: {self.curriculum.current_stage.value}")
        if self.checkpoint_manager:
            logger.info(f"💾 Checkpoint management enabled - Frequency: {self.config.save_frequency}")
            logger.info(f"📂 Checkpoint directory: {self.checkpoint_manager.checkpoint_dir}")
        logger.info(f"🗃️ Quality experience buffer - Capacity: {self.experience_buffer.capacity:,}")
        
        start_time = time.time()
        
        for episode in range(start_episode, self.config.num_episodes):
            episode_start = time.time()
            
            # Check for full reward reassignment (curriculum learning)
            if (self.curriculum and 
                self.curriculum.should_reassign_rewards(episode)):
                logger.info(f"🔄 Full reward reassignment at episode {episode}")
                self._reassign_all_rewards()
            
            # Train episode
            episode_stats = self.train_episode(episode)
            
            episode_time = time.time() - episode_start
            
            # Session summary after each episode completion
            self._print_session_summary(episode, episode_stats, episode_time)
            
            # Enhanced logging with production features
            if episode % 10 == 0:
                buffer_stats = episode_stats.get('buffer_stats', {})
                curriculum_stage = episode_stats.get('curriculum_stage', 'none')
                
                logger.info(f"Episode {episode:4d}: "
                          f"Reward={episode_stats['episode_reward']:7.2f}, "
                          f"Trades={episode_stats['total_trades']:2d}, "
                          f"Win%={episode_stats['win_rate']:5.1f}, "
                          f"AvgPnL={episode_stats['avg_pnl_pips']:6.2f}, "
                          f"Stage={curriculum_stage}, "
                          f"Buffer={episode_stats.get('buffer_size', 0):,}, "
                          f"Time={episode_time:.1f}s")
                          
                # CLOSE LOGIT MONITORING: Track improvement every 25 episodes
                if episode % 25 == 0 and episode > 0:
                    try:
                        # Create temporary environment for monitoring  
                        temp_env = self._create_episode_environment(episode)
                        if temp_env:
                            obs, _ = temp_env.reset()
                            
                            # Get network output for current observation 
                            if isinstance(obs, dict):
                                price_data = obs.get('price_series', obs.get('observation', obs))
                            else:
                                price_data = obs
                                
                            # Ensure proper tensor format
                            if isinstance(price_data, dict):
                                price_tensor = torch.from_numpy(price_data['price_series']).float().to(self.device)
                            else:
                                price_tensor = torch.from_numpy(price_data).float().to(self.device)
                                
                            fused_obs = self.market_encoder(price_tensor)
                            network_output = self.muzero_network.initial_inference(fused_obs.unsqueeze(0))
                            
                            # Extract CLOSE logit (action index 3)
                            close_logit = network_output.policy_logits[0][3].item()
                            
                            # Get softmax distribution
                            action_probs = torch.softmax(network_output.policy_logits[0], dim=0)
                            close_prob = action_probs[3].item()
                            
                            logger.info(f"         🎯 CLOSE Analysis: "
                                      f"Logit={close_logit:+.3f}, "
                                      f"Prob={close_prob:.1%}, "
                                      f"Target>-0.5@ep500")
                                      
                            # Log milestone progress
                            if episode == 100:
                                logger.info(f"         📈 MILESTONE 100: Expected CLOSE ~8%, Logit ~-0.7")
                            elif episode == 500:
                                logger.info(f"         📈 MILESTONE 500: Expected CLOSE ~12%, Logit ~-0.4") 
                                
                    except Exception as e:
                        logger.warning(f"         ⚠️ CLOSE logit monitoring failed: {e}")
                        
                    finally:
                        # Clean up temporary environment
                        try:
                            temp_env.close()
                        except:
                            pass
                
                # Log buffer quality statistics
                if buffer_stats.get('avg_quality'):
                    logger.info(f"         Buffer Quality: "
                              f"Avg={buffer_stats['avg_quality']:.3f}, "
                              f"P90={buffer_stats.get('quality_percentiles', {}).get('p90', 0):.3f}, "
                              f"Evictions={buffer_stats.get('total_evicted', 0)}")
                          
            # Log curriculum transitions
            if episode_stats.get('curriculum_transitioned', False):
                logger.info(f"🎓 Curriculum stage transition completed at episode {episode}")
                
            # Update total trades counter
            episode_trades = episode_stats.get('total_trades', 0)
            self.total_trades_completed += episode_trades
            
            # Save checkpoint frequently: every 10 episodes for first 100, then every 25
            checkpoint_frequency = 10 if episode < 100 else 25
            should_save_checkpoint = (
                self.checkpoint_manager and (
                    (episode % checkpoint_frequency == 0) or  # Frequent checkpointing
                    (episode < 5) or  # Save first 5 episodes
                    (self.total_trades_completed >= self.trades_per_checkpoint and episode_trades > 0)  # Trade threshold
                )
            )
            
            if should_save_checkpoint:
                # Reset trade counter if checkpoint was triggered by trade count
                if self.total_trades_completed >= self.trades_per_checkpoint and episode_trades > 0:
                    logger.info(f"💾 Checkpoint triggered by trade count: {self.total_trades_completed} trades")
                    self.total_trades_completed = 0  # Reset counter
                
                self._save_production_checkpoint(episode, episode_stats)
            
            # Update scheduler
            self.scheduler.step()
            
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time:.1f} seconds")
        
        # Final statistics
        final_buffer_stats = self.experience_buffer.get_statistics()
        final_curriculum_stats = self.curriculum.get_training_stats() if self.curriculum else {}
        
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Episodes trained: {len(self.episode_rewards)}")
        logger.info(f"   Avg episode reward: {np.mean(self.episode_rewards[-100:]):.3f}")
        logger.info(f"   Final win rate: {np.mean(list(self.win_rate_history)):.3f}")
        logger.info(f"   Buffer utilization: {final_buffer_stats.get('utilization', 0):.1%}")
        logger.info(f"   Total evicted: {final_buffer_stats.get('total_evicted', 0):,}")
        
        if self.curriculum:
            logger.info(f"   Final curriculum stage: {final_curriculum_stats.get('current_stage', 'none')}")
            logger.info(f"   Total stage transitions: {final_curriculum_stats.get('total_transitions', 0)}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'buffer_statistics': final_buffer_stats,
            'curriculum_statistics': final_curriculum_stats
        }
        
    def train_multiprocessing(self, resume_from_checkpoint: bool = True) -> Dict[str, List[float]]:
        """Main training loop with Python multiprocessing for parallel episodes"""
        
        # Check for training resumption and preserve weights
        start_episode = 0
        if resume_from_checkpoint and self.checkpoint_manager:
            start_episode = self.resume_training_from_checkpoint()
            if start_episode > 0:
                logger.info(f"🔄 Resuming training from episode {start_episode}")
                logger.info("✅ Network weights preserved from checkpoint")
                
        logger.info(f"🚀 Starting SWT multiprocessing training for {self.config.num_episodes} episodes")
        logger.info(f"   Parallel processes: {self.num_processes}")
        logger.info(f"   Shared experience buffer: {self.experience_buffer.capacity:,} capacity")
        logger.info("🔧 Workers run independently with fresh networks (simple & reliable)")
        
        # Simplified multiprocessing - no shared weights, no deadlocks
        logger.info("🔧 Simplified multiprocessing: workers run independently")
        
        # Start background experience buffer consumer
        buffer_thread = threading.Thread(target=self._consume_shared_experiences, daemon=True)
        buffer_thread.start()
        
        # Start gradient aggregation thread
        gradient_thread = threading.Thread(target=self._aggregate_gradients, daemon=True)
        gradient_thread.start()
        
        start_time = time.time()
        completed_episodes = start_episode
        
        try:
            # Process episodes in batches using process pool
            batch_size = self.num_processes * 2  # Process 2 episodes per worker per batch
            
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                while completed_episodes < self.config.num_episodes:
                    # Calculate batch episodes
                    remaining_episodes = self.config.num_episodes - completed_episodes
                    current_batch_size = min(batch_size, remaining_episodes)
                    
                    episode_batch = list(range(completed_episodes, completed_episodes + current_batch_size))
                    
                    logger.info(f"📦 Processing episode batch: {episode_batch[0]}-{episode_batch[-1]} ({len(episode_batch)} episodes)")
                    
                    # Submit parallel episode tasks
                    future_to_episode = {
                        executor.submit(
                            self._train_episode_worker, 
                            episode, 
                            self._get_serializable_config()
                        ): episode 
                        for episode in episode_batch
                    }
                    
                    # Collect results
                    batch_stats = []
                    for future in as_completed(future_to_episode):
                        episode = future_to_episode[future]
                        try:
                            episode_stats = future.result(timeout=300)  # 5 min timeout per episode
                            batch_stats.append((episode, episode_stats))
                            
                            # Log episode completion
                            if episode % 10 == 0:
                                logger.info(f"Episode {episode:4d}: "
                                          f"Reward={episode_stats['episode_reward']:7.2f}, "
                                          f"Trades={episode_stats['total_trades']:2d}, "
                                          f"Win%={episode_stats['win_rate']:5.1f}, "
                                          f"AvgPnL={episode_stats['avg_pnl_pips']:6.2f}")
                                          
                        except Exception as e:
                            logger.error(f"❌ Episode {episode} failed: {e}")
                            # Create dummy stats for failed episode
                            batch_stats.append((episode, {
                                'episode_reward': 0.0,
                                'episode_length': 0,
                                'total_trades': 0,
                                'win_rate': 0.0,
                                'avg_pnl_pips': 0.0
                            }))
                    
                    # Process batch results
                    for episode, episode_stats in sorted(batch_stats):
                        self.episode_rewards.append(episode_stats['episode_reward'])
                        self.episode_lengths.append(episode_stats['episode_length'])
                        
                        # Save checkpoint frequently to preserve weights and experience buffer
                        # More frequent saves: every 10 episodes for first 100, then every 25
                        checkpoint_frequency = 10 if episode < 100 else 25
                        if (episode % checkpoint_frequency == 0 or episode < 5) and self.checkpoint_manager:
                            logger.info(f"💾 Saving checkpoint at episode {episode} (weight preservation + experience buffer)")
                            self._save_production_checkpoint(episode, episode_stats)
                            
                    completed_episodes += current_batch_size
                    
                    # Aggregate gradients and update models
                    self._update_models_from_gradients()
                    
                    logger.info(f"✅ Batch complete: {completed_episodes}/{self.config.num_episodes} episodes")
                    
        except Exception as e:
            logger.error(f"❌ Multiprocessing training failed: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'mp_manager'):
                self.mp_manager.shutdown()
                
        total_time = time.time() - start_time
        logger.info(f"✅ Multiprocessing training completed in {total_time:.1f} seconds")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'buffer_statistics': self.experience_buffer.get_statistics()
        }
        
    def _save_shared_model_weights(self) -> None:
        """Save current model weights to shared memory for multiprocessing access"""
        try:
            # Preserve network weights in shared state
            market_encoder_state = {k: v.cpu().clone() for k, v in self.market_encoder.state_dict().items()}
            muzero_network_state = {k: v.cpu().clone() for k, v in self.muzero_network.state_dict().items()}
            
            self.shared_model_state['market_encoder'] = pickle.dumps(market_encoder_state)
            self.shared_model_state['muzero_network'] = pickle.dumps(muzero_network_state)
            self.shared_model_state['last_updated'] = time.time()
            
            logger.info("🔒 Network weights saved to shared memory")
            
        except Exception as e:
            logger.error(f"❌ Failed to save shared model weights: {e}")
            raise
    
    def _get_shared_model_weights(self) -> Dict[str, bytes]:
        """Get current shared model weights"""
        return {
            'market_encoder': self.shared_model_state.get('market_encoder'),
            'muzero_network': self.shared_model_state.get('muzero_network'),
            'last_updated': self.shared_model_state.get('last_updated', 0)
        }
    
    def _get_serializable_config(self) -> Dict[str, Any]:
        """Get serializable configuration for worker processes"""
        return {
            'wst': self.config.wst,
            'fusion': self.config.fusion,
            'stochastic_muzero': self.config.stochastic_muzero,
            'training': self.config.training,
            'reward': self.config.reward,
            'reward_profiles': self.config.reward_profiles,
            'environment': self.config.environment,
            'data_path': self.config.data_path,
            'reward_type': self.config.reward_type,
            'device': str(self.device),
            'num_actions': self.config.num_actions,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'gradient_clip': self.config.gradient_clip
        }
    
    def _consume_shared_experiences(self) -> None:
        """Background thread to consume experiences from shared queue"""
        while True:
            try:
                experience_data = self.shared_experience_queue.get(timeout=1.0)
                if experience_data is None:  # Shutdown signal
                    break
                    
                # Deserialize experience
                experience = pickle.loads(experience_data)
                self.experience_buffer.add(experience)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error consuming shared experience: {e}")
                
    def _aggregate_gradients(self) -> None:
        """Background thread to aggregate gradients from worker processes"""
        while True:
            try:
                gradient_data = self.gradient_aggregation_queue.get(timeout=1.0)
                if gradient_data is None:  # Shutdown signal
                    break
                    
                # Deserialize and apply gradients
                gradients = pickle.loads(gradient_data)
                self._apply_aggregated_gradients(gradients)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error aggregating gradients: {e}")
                
    def _apply_aggregated_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Apply aggregated gradients to main models"""
        try:
            # Update market encoder
            if 'market_encoder' in gradients:
                for name, param in self.market_encoder.named_parameters():
                    if name in gradients['market_encoder'] and param.grad is not None:
                        param.grad.data.add_(gradients['market_encoder'][name])
                        
            # Update muzero network
            if 'muzero_network' in gradients:
                for name, param in self.muzero_network.named_parameters():
                    if name in gradients['muzero_network'] and param.grad is not None:
                        param.grad.data.add_(gradients['muzero_network'][name])
                        
            # Apply optimizer step
            self.optimizer.step()
            
            # Update shared weights with new state
            self._save_shared_model_weights()
            
        except Exception as e:
            logger.error(f"❌ Failed to apply aggregated gradients: {e}")
    
    def _update_models_from_gradients(self) -> None:
        """Update models by processing accumulated gradients"""
        # This is called after each batch to ensure models stay synchronized
        pass  # Already handled by _apply_aggregated_gradients
    
    @staticmethod
    def _train_episode_worker(episode: int, config_dict: Dict[str, Any]) -> Dict[str, float]:
        """Worker function to train single episode in separate process"""
        try:
            # Set up process-local models
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use simplified models without WST to avoid multiprocessing deadlock
            # Create basic MuZero network without market encoder (WST causes deadlock)
            muzero_network = create_swt_stochastic_muzero_network(config_dict.get('muzero_config', {}))
            muzero_network = muzero_network.to(device)
            
            # Skip market encoder initialization to prevent WST deadlock
            market_encoder = None
            
            # Create MCTS
            mcts = create_swt_stochastic_mcts(muzero_network, config_dict.get('mcts_config', {}))
            
            # Load only required CSV rows for this worker (streaming approach)
            session_data = SWTStochasticMuZeroTrainer._load_worker_session_streaming(config_dict)
            if session_data is None:
                return {'episode_reward': 0.0, 'episode_length': 0, 'total_trades': 0, 'win_rate': 0.0, 'avg_pnl_pips': 0.0, 'env_trades': [], 'session_data': {}}
            
            # Create environment
            env = create_swt_forex_environment(
                session_data=session_data,
                config_dict={'reward': config_dict['reward'], 'environment': config_dict['environment']}
            )
            
            # Run episode
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            experiences = []
            
            while not done:
                # Simple action selection for workers (no MCTS to save compute)
                action = np.random.randint(0, config_dict['num_actions'])
                
                next_obs, reward, done, truncated, env_info = env.step(action)
                
                # Store experience for sharing
                experience = SWTExperience(
                    observation_history=torch.zeros(1, 128),  # Simplified for workers
                    market_prices=torch.tensor(obs['market_prices']),
                    position_features=torch.tensor(obs['position_features']),
                    action=action,
                    reward=reward,
                    value_target=0.0,  # Will be calculated later
                    policy_target=np.ones(config_dict['num_actions']) / config_dict['num_actions'],
                    next_observation={'market_prices': torch.tensor(next_obs['market_prices']), 'position_features': torch.tensor(next_obs['position_features'])},
                    done=done
                )
                experiences.append(experience)
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # Get trade statistics
            trade_stats = env.get_trade_statistics()
            
            # Create session metadata for chart generation
            session_metadata = {
                'session_id': f'worker_session_{episode}',
                'start_time': session_data.iloc[0]['datetime'] if 'datetime' in session_data.columns else '',
                'end_time': session_data.iloc[-1]['datetime'] if 'datetime' in session_data.columns else '',
                'total_bars': len(session_data),
                'duration_hours': config_dict['environment'].get('session_hours', 12),
                'data': session_data[['datetime', 'close']].copy() if 'datetime' in session_data.columns and 'close' in session_data.columns else pd.DataFrame()
            }
            
            return {
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'total_trades': trade_stats['total_trades'],
                'win_rate': trade_stats['win_rate'],
                'avg_pnl_pips': trade_stats['avg_pnl_pips'],
                'env_trades': env.completed_trades,  # Add trade data for chart generation
                'session_data': session_metadata,  # Add session data for price charts
                'num_experiences': len(experiences)  # Just send count, not full experiences
            }
            
        except Exception as e:
            logger.error(f"❌ Worker episode {episode} failed: {e}")
            return {'episode_reward': 0.0, 'episode_length': 0, 'total_trades': 0, 'win_rate': 0.0, 'avg_pnl_pips': 0.0, 'env_trades': [], 'session_data': {}}
    
    @staticmethod
    def _sample_worker_session(csv_data: pd.DataFrame, config_dict: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Sample random session for worker process"""
        try:
            session_hours = config_dict['environment'].get('session_hours', 6)
            session_bars = int(session_hours * 60)  # 6 hours = 360 bars
            
            if len(csv_data) < session_bars:
                return None
                
            max_start = len(csv_data) - session_bars
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + session_bars
            
            return csv_data.iloc[start_idx:end_idx].copy()
            
        except Exception as e:
            logger.error(f"❌ Worker session sampling failed: {e}")
            return None
    
    @staticmethod
    def _load_worker_session_streaming(config_dict: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load only required CSV rows for worker session using streaming approach"""
        try:
            data_path = config_dict['data_path']
            session_hours = config_dict['environment'].get('session_hours', 6)
            session_bars = int(session_hours * 60)  # 6 hours = 360 bars
            
            # Get total rows without loading entire file
            with open(data_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            
            if total_rows < session_bars:
                logger.error(f"❌ Not enough data: {total_rows} < {session_bars}")
                return None
                
            # Random start index
            max_start = total_rows - session_bars
            start_idx = np.random.randint(0, max_start)
            
            # Load only required rows for this session (much faster!)
            session_data = pd.read_csv(
                data_path, 
                skiprows=range(1, start_idx + 1),  # Skip to start_idx (keeping header)
                nrows=session_bars  # Load only session_bars rows
            )
            
            # Validate data completeness
            if len(session_data) < session_bars:
                logger.warning(f"⚠️ Session truncated: {len(session_data)} < {session_bars}")
                
            return session_data
            
        except Exception as e:
            logger.error(f"❌ Worker session streaming failed: {e}")
            return None
    
    def _reassign_all_rewards(self):
        """Reassign all rewards in experience buffer (curriculum learning)"""
        
        if not hasattr(self.experience_buffer, 'buffer') or len(self.experience_buffer.buffer) == 0:
            return
            
        logger.info(f"🔄 Reassigning rewards for {len(self.experience_buffer.buffer)} experiences")
        
        # This is a placeholder - in a full implementation, we would:
        # 1. Re-evaluate all experiences with current policy/value function
        # 2. Update value targets and policy targets
        # 3. Recalculate quality scores
        
        # For now, just recalculate quality scores
        for experience in self.experience_buffer.buffer:
            experience.quality_score = experience._calculate_quality_score()
            
        # Rebuild quality heap
        self.experience_buffer.quality_heap = []
        for i, exp in enumerate(self.experience_buffer.buffer):
            import heapq
            heapq.heappush(self.experience_buffer.quality_heap, (exp.quality_score, i))
            
        logger.info("✅ Reward reassignment completed")
    
    def _reassign_session_trade_rewards(self, completed_trades: List) -> None:
        """
        FIXED: Reassign rewards for all actions associated with completed trades
        
        CRITICAL FIX: All trade-associated actions get final AMDDP1 reward as "equal partners"
        This solves the problem where networks never learned CLOSE action value.
        """
        if not hasattr(self.experience_buffer, 'buffer') or len(self.experience_buffer.buffer) == 0:
            return
            
        total_reassigned = 0
        trades_processed = 0
        
        # Get recent experiences from this episode (last 1000 to cover full episode)
        recent_experiences = list(enumerate(self.experience_buffer.buffer))[-1000:]
        
        for trade in completed_trades:
            trades_processed += 1
            
            # Calculate final AMDDP1 reward for this trade
            if self.config.reward['type'] == 'amddp1':
                final_trade_reward = trade.reward_amddp1
            elif self.config.reward['type'] == 'amddp5':
                final_trade_reward = trade.reward_amddp5
            else:
                final_trade_reward = trade.pnl_pips
            
            # CRITICAL FIX: Find ALL experiences in this episode that contributed to trade
            # Since we don't have precise trade timestamps in experience buffer yet,
            # reassign reward to ALL recent experiences from this episode
            # This ensures EVERY action (BUY, SELL, HOLD, CLOSE) learns the final outcome
            
            trade_experiences = []
            
            # Method 1: Use all experiences from current episode (most recent)
            episode_experience_count = min(len(recent_experiences), 500)  # Typical episode length
            
            for exp_idx, experience in recent_experiences[-episode_experience_count:]:
                trade_experiences.append((exp_idx, experience))
                
            logger.debug(f"🔄 Trade {trades_processed}: Reassigning {len(trade_experiences)} experiences with reward {final_trade_reward:.4f}")
            
            # REWARD REASSIGNMENT: ALL trade-associated actions become "equal partners"
            # Every action that contributed to the trade gets the same final AMDDP1 reward
            for exp_idx, experience in trade_experiences:
                # Store original reward for debugging
                original_reward = experience.reward
                
                # EQUAL PARTNERS: All actions get final trade AMDDP1 outcome
                experience.reward = final_trade_reward
                
                # Recalculate quality score with new reward
                if hasattr(experience, '_calculate_quality_score'):
                    experience.quality_score = experience._calculate_quality_score()
                else:
                    # Fallback quality calculation
                    experience.quality_score = abs(final_trade_reward) + 0.1
                    
                total_reassigned += 1
                
                if abs(original_reward - final_trade_reward) > 0.01:
                    logger.debug(f"   Experience {exp_idx}: Reward {original_reward:.3f} → {final_trade_reward:.3f}")
        
        if total_reassigned > 0:
            # Rebuild quality heap after reward updates
            self.experience_buffer.quality_heap = []
            for i, exp in enumerate(self.experience_buffer.buffer):
                import heapq
                heapq.heappush(self.experience_buffer.quality_heap, (exp.quality_score, i))
                
            logger.info(f"✅ REWARD REASSIGNMENT COMPLETED: {total_reassigned} experiences reassigned for {trades_processed} trades")
        else:
            logger.warning(f"⚠️ No reward reassignment performed for {len(completed_trades)} completed trades")
    
    def monitor_action_distribution(self, env, num_samples: int = 100) -> Dict[str, float]:
        """
        MONITORING: Track action distribution and CLOSE logits to verify fix effectiveness
        
        Args:
            env: SWT environment
            num_samples: Number of random observations to sample for analysis
            
        Returns:
            Dict with action distribution and CLOSE logit statistics
        """
        logger.info(f"📊 MONITORING: Analyzing action distribution and CLOSE logits ({num_samples} samples)...")
        
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0, 'CLOSE': 0}
        close_logits = []
        
        # Sample random observations from current session
        for _ in range(num_samples):
            # Reset to get random observation
            obs, _ = env.reset()
            market_prices, position_features = self._preprocess_observation(obs)
            fused_obs = self._get_fused_observation(market_prices, position_features)
            
            # Get policy logits (no MCTS for monitoring speed)
            with torch.no_grad():
                initial_outputs = self.muzero_network.initial_inference(fused_obs.unsqueeze(0))
                policy_logits = initial_outputs.policy_logits.squeeze(0)
                
                # Record CLOSE logit specifically
                close_logits.append(policy_logits[3].item())  # CLOSE = action 3
                
                # Get action probabilities with temperature
                action_probs = torch.softmax(policy_logits / 0.1, dim=0)  # Low temp for analysis
                selected_action = torch.argmax(action_probs).item()
                
                # Count action selection
                action_names = ['HOLD', 'BUY', 'SELL', 'CLOSE']
                action_counts[action_names[selected_action]] += 1
        
        # Calculate statistics
        action_distribution = {action: count/num_samples for action, count in action_counts.items()}
        
        close_logit_stats = {
            'mean': np.mean(close_logits),
            'std': np.std(close_logits),
            'min': np.min(close_logits),
            'max': np.max(close_logits),
            'median': np.median(close_logits)
        }
        
        # Log results
        logger.info(f"🎯 ACTION DISTRIBUTION:")
        for action, prob in action_distribution.items():
            logger.info(f"   {action}: {prob:.1%}")
            
        logger.info(f"📈 CLOSE LOGITS: μ={close_logit_stats['mean']:.3f}, σ={close_logit_stats['std']:.3f}, "
                   f"range=[{close_logit_stats['min']:.3f}, {close_logit_stats['max']:.3f}]")
        
        return {
            'action_distribution': action_distribution,
            'close_logit_stats': close_logit_stats
        }
    
    def clear_experience_buffer_for_fresh_training(self) -> None:
        """
        FRESH START: Clear contaminated experience buffer for corrected reward training
        
        CRITICAL: Must be called before starting training with fixed reward system
        to ensure networks learn proper CLOSE action values from scratch.
        """
        buffer_size_before = len(self.experience_buffer)
        
        # Clear all contaminated experiences
        self.experience_buffer.clear()
        
        # Reset training metrics for fresh start
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.loss_history = []
        
        logger.info(f"🗑️ FRESH TRAINING START: Cleared {buffer_size_before:,} contaminated experiences")
        logger.info(f"   Experience buffer: {len(self.experience_buffer)} / {self.experience_buffer.capacity}")
        logger.info(f"   Training metrics reset for corrected reward structure")
        logger.info(f"   Ready for reward reassignment fix validation")
    
    def _print_session_summary(self, episode: int, episode_stats: Dict[str, Any], episode_time: float):
        """Print comprehensive session summary to console for easy monitoring"""
        
        # Get trading performance metrics
        total_trades = episode_stats.get('total_trades', 0)
        win_rate = episode_stats.get('win_rate', 0.0)
        avg_pnl = episode_stats.get('avg_pnl_pips', 0.0)
        total_pnl = episode_stats.get('total_pnl_pips', 0.0)
        episode_reward = episode_stats.get('episode_reward', 0.0)
        
        # Calculate wins/losses
        if total_trades > 0:
            wins = int(total_trades * (win_rate / 100.0))
            losses = total_trades - wins
        else:
            wins = losses = 0
        
        # Session info
        session_id = episode_stats.get('session_id', f'ep{episode}')
        bars_count = episode_stats.get('bars_count', 0)
        
        # Build summary line
        print(f" {'='*100}")
        print(f"📊 SESSION {episode:4d} COMPLETE: {session_id}")
        print(f"{'='*100}")
        
        # Trading Performance
        print(f"🏦 TRADING PERFORMANCE:")
        print(f"   📈 Trades: {total_trades:2d} | Wins: {wins:2d} | Losses: {losses:2d} | Win Rate: {win_rate:5.1f}%")
        
        if total_trades > 0:
            print(f"   💰 P&L: Total={total_pnl:+6.2f} pips | Average={avg_pnl:+6.2f} pips/trade")
            if avg_pnl > 0:
                print(f"   ✅ PROFITABLE SESSION (+{avg_pnl:.2f} pips avg)")
            else:
                print(f"   ❌ LOSING SESSION ({avg_pnl:.2f} pips avg)")
        else:
            print(f"   ⏸️  NO TRADES EXECUTED")
        
        # Reward and Performance
        total_session_pnl = episode_stats.get('total_pnl_pips', 0.0)
        print(f"🎯 REWARD & PERFORMANCE:")
        print(f"   🔢 Episode Reward (pre-redistribution): {episode_reward:+8.2f}")
        print(f"   💰 Total Session P&L: {total_session_pnl:+8.2f} pips")
        print(f"   ⏱️  Duration: {episode_time:6.1f}s | Bars Processed: {bars_count:3d}")
        
        # Buffer status
        buffer_size = episode_stats.get('buffer_size', len(getattr(self, 'experience_buffer', [])))
        buffer_capacity = getattr(self.experience_buffer, 'capacity', 0) if hasattr(self, 'experience_buffer') else 0
        buffer_utilization = (buffer_size / buffer_capacity * 100) if buffer_capacity > 0 else 0
        
        print(f"📦 EXPERIENCE BUFFER:")
        print(f"   Size: {buffer_size:,} / {buffer_capacity:,} ({buffer_utilization:5.1f}% full)")
        
        # Performance indicators
        if total_trades > 0:
            if win_rate >= 60 and avg_pnl > 1.0:
                print(f"🌟 EXCELLENT PERFORMANCE: High win rate + strong profits!")
            elif win_rate >= 50 and avg_pnl > 0:
                print(f"✨ GOOD PERFORMANCE: Profitable with decent win rate")
            elif avg_pnl > 0:
                print(f"💚 NET POSITIVE: Profitable despite mixed results")
            else:
                print(f"🔴 NEEDS IMPROVEMENT: Focus on risk management")
        
        print(f"{'='*100} ")
    
    def _save_production_checkpoint(self, episode: int, episode_stats: Dict[str, Any]):
        """Save production-ready checkpoint with complete inference state"""
        
        try:
            # Prepare complete checkpoint state for production inference
            checkpoint_state = {
                # Core neural networks (required for inference)
                'market_encoder_state': self.market_encoder.state_dict(),
                'muzero_network_state': self.muzero_network.state_dict(),
                
                # Training state (required for resumption)
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'total_trades_completed': self.total_trades_completed,
                
                # Experience buffer (for continued learning)
                'experience_buffer': [
                    {
                        'observation_history': exp.observation_history.cpu() if hasattr(exp.observation_history, 'cpu') else exp.observation_history,
                        'market_prices': exp.market_prices.cpu() if hasattr(exp.market_prices, 'cpu') else exp.market_prices,
                        'position_features': exp.position_features.cpu() if hasattr(exp.position_features, 'cpu') else exp.position_features,
                        'action': exp.action,
                        'reward': exp.reward,
                        'value_target': exp.value_target,
                        'policy_target': exp.policy_target,
                        'quality_score': exp.quality_score,
                        # 'episode_id': getattr(exp, 'episode_id', episode),  # Removed for compatibility
                        'step_id': getattr(exp, 'step_id', 0)
                    } for exp in self.experience_buffer.buffer
                ],
                
                # Training metrics and history
                'training_metrics': {
                    'episode_rewards': self.episode_rewards,
                    'episode_lengths': self.episode_lengths,
                    'training_losses': self.training_losses,
                    'win_rate_history': dict(self.win_rate_history),
                    'loss_history': self.loss_history
                },
                
                # Configuration (critical for inference setup)
                'config': {
                    # WST configuration
                    'wst_j': self.config.wst_j,
                    'wst_q': self.config.wst_q,
                    'wst_backend': self.config.wst_backend,
                    
                    # Network architecture
                    'hidden_dim': self.config.hidden_dim,
                    'latent_dim': self.config.latent_dim,
                    'num_actions': self.config.num_actions,
                    'value_support_size': self.config.value_support_size,
                    
                    # Market configuration
                    'price_series_length': self.config.price_series_length,
                    'position_features_dim': self.config.position_features_dim,
                    
                    # Training parameters
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'buffer_size': self.config.buffer_size,
                    
                    # Complete config dict
                    'full_config': asdict(self.config)
                }
            }
            
            # Calculate enhanced analytics - use session data from worker if available
            current_session_data = episode_stats.get('session_data', getattr(self, 'current_session_data', {}))
            
            # Get complete trade data from environment for chart recreation
            env_trades = episode_stats.get('env_trades', [])
            session_price_data = current_session_data.get('data', pd.DataFrame())
            
            # Enhanced trade data with chart recreation info
            enhanced_trades = []
            for trade in env_trades:
                # Convert trade record to serializable format with chart data
                trade_dict = {
                    'direction': trade.direction,
                    'entry_price': float(trade.entry_price),
                    'exit_price': float(trade.exit_price),
                    'entry_time': str(trade.entry_time),
                    'exit_time': str(trade.exit_time),
                    'pnl_pips': float(trade.pnl_pips),
                    'duration_bars': trade.duration_bars,
                    'exit_reason': trade.exit_reason,
                    'reward_amddp5': float(trade.reward_amddp5),
                    'reward_amddp1': float(trade.reward_amddp1),
                    'max_drawdown_pips': float(trade.max_drawdown_pips),
                    'accumulated_drawdown_pips': float(trade.accumulated_drawdown_pips),
                    
                    # Chart recreation data
                    'entry_timestamp': str(trade.entry_time),
                    'exit_timestamp': str(trade.exit_time)
                }
                
                enhanced_trades.append(trade_dict)
            
            # Use enhanced trades for analytics
            trades_data = enhanced_trades
            
            # Trading analytics
            total_pnl = sum(trade.get('pnl_pips', 0) for trade in trades_data)
            profitable_trades = [t for t in trades_data if t.get('pnl_pips', 0) > 0]
            losing_trades = [t for t in trades_data if t.get('pnl_pips', 0) < 0]
            
            gross_profit = sum(t.get('pnl_pips', 0) for t in profitable_trades)
            gross_loss = abs(sum(t.get('pnl_pips', 0) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Streak analysis
            consecutive_wins = consecutive_losses = max_wins = max_losses = 0
            current_streak_wins = current_streak_losses = 0
            
            for trade in trades_data:
                if trade.get('pnl_pips', 0) > 0:
                    current_streak_wins += 1
                    current_streak_losses = 0
                    max_wins = max(max_wins, current_streak_wins)
                elif trade.get('pnl_pips', 0) < 0:
                    current_streak_losses += 1
                    current_streak_wins = 0
                    max_losses = max(max_losses, current_streak_losses)
            
            # Drawdown calculation
            running_pnl = 0
            peak_pnl = 0
            max_drawdown = 0
            for trade in trades_data:
                running_pnl += trade.get('pnl_pips', 0)
                peak_pnl = max(peak_pnl, running_pnl)
                drawdown = peak_pnl - running_pnl
                max_drawdown = max(max_drawdown, drawdown)
            
            # Sharpe ratio calculation (simplified)
            if len(trades_data) > 1:
                trade_returns = [t.get('pnl_pips', 0) for t in trades_data]
                mean_return = np.mean(trade_returns)
                std_return = np.std(trade_returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Market context
            market_timeframe = f"{current_session_data.get('start_time', 'Unknown')} - {current_session_data.get('end_time', 'Unknown')}"
            price_data = current_session_data.get('data', pd.DataFrame())
            price_volatility = price_data['close'].std() if 'close' in price_data.columns and len(price_data) > 1 else 0.0
            
            # Trend detection (simple)
            if 'close' in price_data.columns and len(price_data) > 10:
                price_change = price_data['close'].iloc[-1] - price_data['close'].iloc[0]
                if price_change > 0.001:
                    market_trend = "bullish"
                elif price_change < -0.001:
                    market_trend = "bearish" 
                else:
                    market_trend = "sideways"
            else:
                market_trend = "unknown"
            
            # Calculate expectancy for enhanced quality scoring
            expectancy, avg_loss_pips = self.checkpoint_manager._calculate_expectancy(trades_data)
            
            # Calculate average AMDDP rewards for comparison tracking
            if trades_data:
                avg_amddp5_reward = np.mean([trade.get('reward_amddp5', 0) for trade in trades_data])
                avg_amddp1_reward = np.mean([trade.get('reward_amddp1', 0) for trade in trades_data])
            else:
                avg_amddp5_reward = 0.0
                avg_amddp1_reward = 0.0
            
            # Add average reward calculations to episode_stats
            episode_stats['avg_amddp5_reward'] = avg_amddp5_reward
            episode_stats['avg_amddp1_reward'] = avg_amddp1_reward
            
            # Create enhanced checkpoint metadata
            metadata = SWTCheckpointMetadata(
                episode=episode,
                timestamp=datetime.now().isoformat(),
                total_reward=episode_stats.get('episode_reward', 0.0),
                total_trades=episode_stats.get('total_trades', 0),
                win_rate=episode_stats.get('win_rate', 0.0),
                avg_pnl_pips=episode_stats.get('avg_pnl_pips', 0.0),
                avg_amddp5_reward=episode_stats.get('avg_amddp5_reward', 0.0),
                avg_amddp1_reward=episode_stats.get('avg_amddp1_reward', 0.0),
                model_loss=np.mean(self.training_losses[-10:]) if self.training_losses else 0.0,
                wst_backend=self.config.wst_backend,
                session_id=episode,
                curriculum_stage=episode_stats.get('curriculum_stage', 'full_reward'),
                expectancy=expectancy,
                avg_loss_pips=avg_loss_pips,
                
                # Enhanced session data
                session_data={
                    'session_start_time': str(current_session_data.get('start_time', '')),
                    'session_end_time': str(current_session_data.get('end_time', '')),
                    'session_duration_hours': current_session_data.get('duration_hours', 0),
                    'total_bars': current_session_data.get('total_bars', 0),
                    'max_gap_minutes': current_session_data.get('max_gap_minutes', 0),
                    'data_split': current_session_data.get('split', 'unknown')
                },
                
                trading_summary={
                    'total_pnl_pips': total_pnl,
                    'gross_profit_pips': gross_profit,
                    'gross_loss_pips': gross_loss,
                    'profitable_trades': len(profitable_trades),
                    'losing_trades': len(losing_trades),
                    'largest_win_pips': max((t.get('pnl_pips', 0) for t in profitable_trades), default=0),
                    'largest_loss_pips': min((t.get('pnl_pips', 0) for t in losing_trades), default=0),
                    'trades_per_hour': len(trades_data) / max(current_session_data.get('duration_hours', 1), 0.1),
                    'all_trades': trades_data  # Complete trade history
                },
                
                exploration_rate=getattr(self, 'current_exploration_rate', None),
                session_duration=episode_stats.get('duration', 0.0),
                bars_processed=current_session_data.get('total_bars', 0),
                quality_score=self.checkpoint_manager._calculate_trading_quality_score(
                    type('temp', (), {
                        'avg_pnl_pips': episode_stats.get('avg_pnl_pips', 0.0),
                        'total_trades': episode_stats.get('total_trades', 0),
                        'win_rate': episode_stats.get('win_rate', 0.0),
                        'expectancy': expectancy
                    })()
                ),
                
                # Market condition context
                market_session_id=current_session_data.get('session_id', f"ep{episode}"),
                market_timeframe=market_timeframe,
                price_volatility=price_volatility,
                market_trend=market_trend,
                
                # Performance analytics
                max_drawdown_pips=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                consecutive_wins=max_wins,
                consecutive_losses=max_losses
            )
            
            # Determine if this is the best checkpoint
            is_best_reward = len(self.episode_rewards) == 0 or episode_stats.get('episode_reward', -float('inf')) > max(self.episode_rewards)
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                state=checkpoint_state,
                metadata=metadata,
                is_best=is_best_reward
            )
            
            # Verify checkpoint completeness for production inference
            self._verify_production_checkpoint(checkpoint_path, episode)
            
            # Generate trade chart with same name as checkpoint
            self._generate_checkpoint_chart(checkpoint_path, episode_stats)
            
        except Exception as e:
            logger.error(f"❌ Failed to save production checkpoint at episode {episode}: {e}")
            # Continue training even if checkpoint fails
            
    def _verify_production_checkpoint(self, checkpoint_path: Path, episode: int):
        """Verify that checkpoint contains all components needed for production inference"""
        
        try:
            # Load and verify checkpoint structure
            # Fix PyTorch 2.6 weights_only issue with numpy serialization
            import torch.serialization
            try:
                numpy_scalar = getattr(np.core.multiarray, 'scalar', None)
                if numpy_scalar:
                    torch.serialization.add_safe_globals([numpy_scalar])
            except AttributeError:
                pass
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Required keys for production inference
            required_inference_keys = [
                'market_encoder_state',
                'muzero_network_state', 
                'config'
            ]
            
            # Required config keys for inference setup
            required_config_keys = [
                'wst_j', 'wst_q', 'wst_backend',
                'hidden_dim', 'latent_dim', 'num_actions',
                'price_series_length', 'position_features_dim'
            ]
            
            # Verify main structure
            missing_keys = [key for key in required_inference_keys if key not in checkpoint]
            if missing_keys:
                logger.error(f"❌ Checkpoint missing required keys: {missing_keys}")
                return False
                
            # Verify config completeness
            config = checkpoint.get('config', {})
            missing_config = [key for key in required_config_keys if key not in config]
            if missing_config:
                logger.error(f"❌ Checkpoint config missing keys: {missing_config}")
                return False
            
            # Verify model state dict structure
            market_encoder_state = checkpoint.get('market_encoder_state', {})
            muzero_state = checkpoint.get('muzero_network_state', {})
            
            if not market_encoder_state or not muzero_state:
                logger.error(f"❌ Checkpoint has empty model states")
                return False
            
            # Verify tensor shapes are reasonable
            for name, tensor in market_encoder_state.items():
                if not isinstance(tensor, torch.Tensor):
                    logger.error(f"❌ Market encoder {name} is not a tensor")
                    return False
                    
            for name, tensor in muzero_state.items():
                if not isinstance(tensor, torch.Tensor):
                    logger.error(f"❌ MuZero network {name} is not a tensor")
                    return False
            
            # Check file size (should be reasonable for the model)
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 1.0:  # Too small - likely incomplete
                logger.warning(f"⚠️ Checkpoint file size suspiciously small: {file_size_mb:.1f}MB")
            elif file_size_mb > 500.0:  # Too large - potentially problematic
                logger.warning(f"⚠️ Checkpoint file size very large: {file_size_mb:.1f}MB")
            
            logger.info(f"✅ Production checkpoint verified: Episode {episode}")
            logger.info(f"   Size: {file_size_mb:.1f}MB")
            logger.info(f"   Market encoder params: {len(market_encoder_state)}")
            logger.info(f"   MuZero network params: {len(muzero_state)}")
            logger.info(f"   Ready for production inference")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint verification failed: {e}")
            return False
    
    def _generate_checkpoint_chart(self, checkpoint_path: Path, episode_stats: Dict[str, Any]) -> None:
        """Generate trade chart for checkpoint with same filename"""
        try:
            # Check if trades exist
            env_trades = episode_stats.get('env_trades', [])
            if not env_trades:
                logger.info(f"📊 No trades to chart for episode checkpoint")
                return
                
            # Load checkpoint data for chart generation
            checkpoint_data = {
                'trades': [
                    {
                        'direction': trade.direction,
                        'entry_price': float(trade.entry_price),
                        'exit_price': float(trade.exit_price),
                        'entry_timestamp': str(trade.entry_time),
                        'exit_timestamp': str(trade.exit_time),
                        'pnl_pips': float(trade.pnl_pips),
                        'duration_bars': trade.duration_bars,
                        'exit_reason': trade.exit_reason
                    } for trade in env_trades
                ],
                'session_data': current_session_data,
                'episode': checkpoint_path.stem.split('_')[-1] if '_' in checkpoint_path.stem else '0',
                'trading_summary': episode_stats.get('trading_summary', {}),
                '_checkpoint_path': str(checkpoint_path)
            }
            
            # Generate price data from current session (now from worker)
            price_data = pd.DataFrame()
            session_data = current_session_data.get('data', pd.DataFrame())
            
            # Debug: Check what data we have
            logger.info(f"📊 Session data columns: {list(session_data.columns) if not session_data.empty else 'empty'}")
            logger.info(f"📊 Session data length: {len(session_data)}")
            
            if not session_data.empty and 'datetime' in session_data.columns and 'close' in session_data.columns:
                price_data = session_data[['datetime', 'close']].copy()
                price_data.rename(columns={'datetime': 'timestamp'}, inplace=True)
                price_data['volume'] = 100  # Synthetic volume
                logger.info(f"📊 Price data prepared: {len(price_data)} rows")
            else:
                logger.warning(f"📊 Missing price data - columns: {list(session_data.columns) if not session_data.empty else 'empty'}")
            
            # Update checkpoint data session_data to match chart visualizer expectations
            checkpoint_data['session_data'].update({
                'session_start_time': current_session_data.get('start_time', ''),
                'session_end_time': current_session_data.get('end_time', '')
            })
            
            # Create chart with same name as checkpoint
            chart_path = checkpoint_path.parent / f"{checkpoint_path.stem}.png"
            
            # Generate the chart
            create_trade_chart(checkpoint_data, price_data, str(chart_path))
            
            logger.info(f"📈 Trade chart generated: {chart_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate checkpoint chart: {e}")
            # Don't fail checkpoint saving if chart generation fails
    
    def _save_checkpoint(self, episode: int):
        """Legacy checkpoint save (deprecated - using checkpoint manager now)"""
        
        # This method is deprecated - checkpointing is now handled by _save_production_checkpoint
        # which includes full production inference verification
        pass


def create_swt_trainer(config_dict: dict = None, data_path: str = None) -> SWTStochasticMuZeroTrainer:
    """Factory function to create SWT trainer"""
    
    if config_dict is None:
        config_dict = {}
        
    if data_path is not None:
        config_dict['data_path'] = data_path
    
    # Remove parameters that aren't part of SWTTrainingConfig and map structure
    clean_config = {k: v for k, v in config_dict.items() if k != 'partial_reset'}
    
    # Map config structure for compatibility 
    if 'model_config' in clean_config:
        model_config = clean_config.pop('model_config')
        clean_config.update(model_config)
    if 'training_config' in clean_config:
        training_config = clean_config.pop('training_config')
        clean_config['training'] = training_config
    if 'environment_config' in clean_config:
        environment_config = clean_config.pop('environment_config')
        clean_config['environment'] = environment_config
        
    config = SWTTrainingConfig(**clean_config)
    return SWTStochasticMuZeroTrainer(config)


def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description='SWT Stochastic MuZero Training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to forex data')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='swt_checkpoints', help='Save directory')
    parser.add_argument('--reward_type', type=str, default='pure_pips', help='Reward type')
    parser.add_argument('--partial_reset', action='store_true', help='Partial reset: keep networks, clear experience buffer, switch to test data')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = {}
        
    # Override with command line args
    config_dict.update({
        'data_path': args.data_path,
        'num_episodes': args.episodes,
        'save_dir': args.save_dir,
        'reward_type': args.reward_type
    })
    
    # Apply reward profile if specified (amddp1 or amddp5)
    if args.reward_type in ['amddp1', 'amddp5'] and 'reward_profiles' in config_dict:
        reward_profile = config_dict['reward_profiles'].get(args.reward_type)
        if reward_profile:
            # Override reward config with profile settings
            config_dict['reward'].update(reward_profile)
            logger.info(f"📊 Applied {args.reward_type.upper()} reward profile:")
            logger.info(f"   Drawdown penalty: {reward_profile['drawdown_penalty']}")
            logger.info(f"   Profit protection: {reward_profile['profit_protection']}")
    
    # Handle partial_reset separately (not part of SWTTrainingConfig)
    partial_reset = args.partial_reset
    
    # Create and run trainer
    trainer = create_swt_trainer(config_dict)
    
    # Apply partial reset if requested
    if partial_reset:
        trainer._handle_partial_reset()
        
    # Always use multiprocessing for 10,000+ episode training
    # Switch to multiprocessing for large-scale training
    if args.episodes >= 10000:
        logger.info("🔧 Using multiprocessing training for large-scale parallel episode execution")
        results = trainer.train_multiprocessing()
    else:
        logger.info("🔧 Using single-threaded training for current episode count")
        results = trainer.train()
    
    print(f"🎯 Training completed!")
    print(f"   Episodes: {len(results['episode_rewards'])}")
    print(f"   Average reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"   Final checkpoint saved in: {trainer.save_dir}")


if __name__ == "__main__":
    main()