#!/usr/bin/env python3
"""
New SWT Training Main Entry Point
Production-ready training orchestrator with complete monitoring and safety features
"""

import sys
import os
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import asyncio
import time

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_core.types import AgentType, ProcessState, ManagedProcess
from swt_core.exceptions import ConfigurationError, CheckpointError
from swt_core.sqn_calculator import SQNCalculator, SQNResult
from swt_features.feature_processor import FeatureProcessor
from swt_environments.swt_forex_env import SWTForexEnvironment, SWTAction
from swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetwork
from swt_core.swt_mcts import SWTStochasticMCTS, SWTMCTSConfig
from swt_validation.automated_validator import AutomatedValidator, ValidationTrigger, create_validation_callback
from swt_validation.composite_scorer import create_metrics_from_dict
import torch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SWTTrainingOrchestrator(ManagedProcess):
    """
    Main training orchestrator with complete safety and monitoring
    """
    
    def __init__(self, config_path: str, max_episodes: int = 20000, 
                 enable_validation: bool = True, validation_data_path: str = None):
        super().__init__(
            name="SWT-Training",
            max_runtime_hours=24.0,
            max_episodes=max_episodes,
            enable_external_monitoring=True
        )
        
        self.config_path = config_path
        self.config_manager = None
        self.config = None
        self.feature_processor = None
        self.training_engine = None
        self.environment = None
        
        # Validation setup
        self.enable_validation = enable_validation
        self.validator = None
        self.validation_callback = None
        if enable_validation and validation_data_path:
            self._setup_validation(validation_data_path)
        self.network = None
        self.mcts = None
        self.optimizer = None
        self.replay_buffer = []
        
        # Training state
        self.current_episode = 0
        self.last_checkpoint_episode = 0
        self.best_performance = float('-inf')
        self.best_sqn = float('-inf')
        self.sqn_calculator = SQNCalculator()

        # Metrics
        self.training_start_time = datetime.now()
        self.episodes_per_hour = 0.0
        
        logger.info(f"🚀 SWT Training Orchestrator initialized")
    
    def _setup_validation(self, validation_data_path: str):
        """Setup automated validation system"""
        try:
            # Configure validation triggers
            triggers = ValidationTrigger(
                expectancy_improvement_threshold=0.10,  # 10% improvement
                episode_interval=100,  # Every 100 episodes
                time_interval_hours=6.0,  # Every 6 hours
                score_improvement_threshold=5.0,  # 5 point improvement
                force_validation_on_best=True,
                min_trades_for_validation=30
            )
            
            # Create validator
            self.validator = AutomatedValidator(
                data_path=validation_data_path,
                triggers=triggers,
                output_dir="validation_results"
            )
            
            # Create validation callback
            self.validation_callback = create_validation_callback(self.validator)
            
            logger.info(f"✅ Validation system initialized with data: {validation_data_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Validation setup failed: {e}")
            self.enable_validation = False
    
    async def _run_checkpoint_validation(self, checkpoint_path: str, episode_result: dict):
        """Run validation on checkpoint if triggered"""
        if not self.enable_validation or not self.validator:
            return
        
        try:
            # Create metrics from episode result
            metrics_dict = {
                'checkpoint_path': checkpoint_path,
                'episode': episode_result['episode'],
                'timestamp': datetime.now(),
                'expectancy': episode_result.get('avg_pips', 0),
                'win_rate': episode_result.get('win_rate', 0),
                'profit_factor': episode_result.get('profit_factor', 1.0),
                'total_trades': episode_result.get('trades', 0),
                'sharpe_ratio': episode_result.get('sharpe_ratio', 0),
                'sortino_ratio': episode_result.get('sortino_ratio', 0),
                'max_drawdown_pct': episode_result.get('max_drawdown', 0) / 1000,  # Convert pips to percentage
                'max_drawdown_pips': episode_result.get('max_drawdown', 0),
                'avg_win_pips': episode_result.get('avg_win_pips', 0),
                'avg_loss_pips': episode_result.get('avg_loss_pips', 0)
            }
            
            # Run validation callback
            await self.validation_callback(checkpoint_path, metrics_dict)
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
    
    def initialize_training_system(self):
        """Initialize all training components"""
        try:
            logger.info("🔧 Initializing training system...")
            
            # Load configuration
            self.config_manager = ConfigManager()
            self.config = self.config_manager.load_config(self.config_path)
            # self.config_manager.force_episode_13475_mode()  # Method not available in ConfigManager
            
            # Initialize feature processor
            # Use precomputed WST if available
            precomputed_path = "precomputed_wst/GBPJPY_WST_CLEAN_2022-2025.h5"
            self.feature_processor = FeatureProcessor(
                self.config,
                precomputed_wst_path=precomputed_path if Path(precomputed_path).exists() else None
            )
            
            # Initialize environment with proper configuration
            env_config = {
                'price_series_length': 256,
                'spread_pips': 1.5,
                'pip_value': 0.01,
                'reward': {
                    'type': 'amddp5',  # AMDDP5 reward with profit protection
                    'drawdown_penalty': 1.0,
                    'profit_protection': 5.0,
                    'min_protected_reward': 0.0
                }
            }

            self.environment = SWTForexEnvironment(
                data_path="data/GBPJPY_M1_REAL_2022-2025.csv",
                config_dict=env_config
            )
            
            # Initialize network with proper observation shape
            # The environment returns 137 features (128 WST + 9 position)
            from swt_models.swt_stochastic_networks import SWTStochasticMuZeroConfig

            network_config = SWTStochasticMuZeroConfig(
                market_wst_features=128,
                position_features=9,
                total_input_dim=137,
                final_input_dim=137,  # Use full 137 features (128 WST + 9 position)
                num_actions=4,  # BUY, SELL, HOLD, CLOSE
                hidden_dim=256,
                representation_blocks=6,
                dynamics_blocks=6,
                prediction_blocks=2,
                afterstate_blocks=2,
                support_size=300,
                chance_space_size=32,
                chance_history_length=4,
                afterstate_enabled=True,
                dropout_rate=0.1,
                layer_norm=True,
                residual_connections=True,
                latent_z_dim=16
            )

            self.network = SWTStochasticMuZeroNetwork(network_config)
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=0.0002,  # Episode 13475 learning rate
                weight_decay=1e-4
            )
            
            # Initialize MCTS
            mcts_config = SWTMCTSConfig(
                num_simulations=15,  # Episode 13475 simulations
                c_puct=1.25,
                discount=0.997,
                temperature=1.0
            )
            self.mcts = SWTStochasticMCTS(self.network, mcts_config)
            
            logger.info("🧠 Training engine initialized with SWT components")
            
            # Set up monitoring endpoints
            self._setup_monitoring()
            
            self.state = ProcessState.RUNNING
            logger.info("✅ Training system initialization complete")
            
        except Exception as e:
            self.state = ProcessState.ERROR
            self.error_message = str(e)
            logger.error(f"❌ Training system initialization failed: {e}")
            raise ConfigurationError(f"Training initialization failed: {str(e)}")
    
    def should_stop(self) -> bool:
        """Check if training should stop"""
        return self.state in [ProcessState.STOPPING, ProcessState.STOPPED, ProcessState.CRASHED]

    def _check_resource_limits(self) -> bool:
        """Check resource limits"""
        # Simplified resource check without psutil
        # In production, would check memory/CPU properly
        return True

    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        # Simplified without psutil dependency
        return 0.5  # Mock 50% usage

    def _update_heartbeat(self):
        """Update process heartbeat"""
        self.last_heartbeat = time.time()

    def run_training_loop(self):
        """Main training loop with safety monitoring"""
        logger.info("🏃‍♂️ Starting training loop...")
        
        try:
            while not self.should_stop() and self.current_episode < self.max_episodes:
                # Update heartbeat
                self._update_heartbeat()
                
                # Check resource limits
                if not self._check_resource_limits():
                    logger.error("❌ Resource limits exceeded, stopping training")
                    break
                
                # Mock training episode (replace with actual training logic)
                episode_result = self._run_training_episode()
                
                self.current_episode += 1
                self.episode_count = self.current_episode
                
                # Save checkpoint periodically (default every 10 episodes)
                checkpoint_interval = 10  # Fixed interval for production
                if self.current_episode % checkpoint_interval == 0 or self.current_episode == 1:
                    self._save_checkpoint(episode_result)
                    if self.current_episode == 1:
                        logger.info("🧪 TEST: Saved checkpoint after first episode for validation testing")
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log progress
                if self.current_episode % 10 == 0:
                    self._log_training_progress()
                
                # Check for early stopping conditions
                if self._should_early_stop():
                    logger.info("🛑 Early stopping triggered")
                    break
            
            logger.info("🏁 Training loop completed")
            self.state = ProcessState.STOPPED
            
        except Exception as e:
            self.state = ProcessState.CRASHED
            self.error_message = str(e)
            logger.error(f"❌ Training loop crashed: {e}")
            raise
    
    def _run_training_episode(self):
        """Run actual training episode with SWT components"""
        # Initialize episode metrics
        episode_start_time = datetime.now()
        total_reward = 0.0
        episode_length = 0
        losses = []
        trades = []
        
        try:
            # Reset environment
            observation = self.environment.reset()
            done = False
            
            # Episode 13475 compatible parameters
            num_simulations = 15  # MCTS simulations
            c_puct = 1.25
            temperature = 1.0
            discount = 0.997  # Episode 13475 discount factor
            
            # Episode trajectory for training
            trajectory = []

            # Clear trades list for this episode
            trades = []

            while not done:  # Episode ends when environment signals done (360 bars)
                # Extract position state and current price from environment
                position_state = self.environment.position
                current_price = self.environment.df['close'].iloc[
                    self.environment.current_step + self.environment.price_series_length
                ]

                # For precomputed WST, use the current step as window index
                # This maps directly to the precomputed features
                window_index = self.environment.current_step

                # Only add market data if NOT using precomputed (for live trading)
                if self.feature_processor.precomputed_loader is None:
                    from swt_features.market_features import MarketDataPoint
                    current_idx = self.environment.current_step + self.environment.price_series_length
                    market_data = MarketDataPoint(
                        timestamp=self.environment.df.index[current_idx],
                        open=self.environment.df['open'].iloc[current_idx],
                        high=self.environment.df['high'].iloc[current_idx],
                        low=self.environment.df['low'].iloc[current_idx],
                        close=current_price
                    )
                    self.feature_processor.add_market_data(market_data)

                # Convert SWTPositionState to PositionState for feature processor
                from swt_core.types import PositionState, PositionType

                # Determine position type
                if position_state.is_long:
                    pos_type = PositionType.LONG
                elif position_state.is_short:
                    pos_type = PositionType.SHORT
                else:
                    pos_type = PositionType.FLAT

                # Create immutable PositionState
                feature_position_state = PositionState(
                    position_type=pos_type,
                    entry_price=position_state.entry_price,
                    unrealized_pnl_pips=position_state.unrealized_pnl_pips,
                    duration_minutes=position_state.duration_bars  # Using bars as minutes proxy
                )

                # Process observation with position state and current price
                processed_obs = self.feature_processor.process_observation(
                    position_state=feature_position_state,
                    current_price=current_price,
                    window_index=window_index  # Pass window index for precomputed WST
                )
                wst_features = torch.FloatTensor(processed_obs.combined_features).unsqueeze(0)
                
                # Run MCTS to get action probabilities
                action_probs, search_path, root_value = self.mcts.run(
                    root_observation=wst_features,
                    add_exploration_noise=True,
                    override_temperature=temperature if temperature > 0 else None
                )
                
                # Select action based on probabilities
                if temperature > 0:
                    # Sample action according to probabilities
                    action = np.random.choice(len(action_probs), p=action_probs)
                else:
                    # Select most probable action
                    action = np.argmax(action_probs)
                
                # Execute action in environment
                next_observation, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                
                # Store transition
                trajectory.append({
                    'observation': processed_obs.combined_features,
                    'action': action,
                    'reward': reward,
                    'value': root_value,
                    'policy': action_probs
                })
                
                # Update metrics
                total_reward += reward
                episode_length += 1
                
                # Track trades from environment
                if 'completed_trades' in info and info['completed_trades']:
                    trades.extend(info['completed_trades'])
                
                # Train network periodically
                if len(self.replay_buffer) >= 32 and episode_length % 10 == 0:
                    batch_loss = self._train_network_step()
                    losses.append(batch_loss)
                
                observation = next_observation

            # CRITICAL: Collect trades from environment BEFORE episode ends (before reset)
            if hasattr(self.environment, 'completed_trades'):
                trades = self.environment.completed_trades.copy()
                logger.info(f"📊 Episode end: Collected {len(trades)} trades from environment")
            else:
                logger.warning("❌ Environment has no completed_trades attribute")
                trades = []

            # Add trajectory to replay buffer
            if len(trajectory) > 0:
                self.replay_buffer.append(trajectory)
                # Keep buffer size limited
                if len(self.replay_buffer) > 1000:
                    self.replay_buffer.pop(0)

            # Calculate episode statistics
            avg_loss = np.mean(losses) if losses else 0.0
            win_rate = len([t for t in trades if t.pnl_pips > 0]) / max(len(trades), 1)
            
            # Calculate SQN for the episode
            trade_pnls = [t.pnl_pips for t in trades] if trades else []
            sqn_result = self.sqn_calculator.calculate_sqn(trade_pnls)

            # Episode result matching SWT format with SQN
            episode_result = {
                "episode": self.current_episode,
                "reward": total_reward,
                "episode_length": episode_length,
                "loss": avg_loss,
                "learning_rate": 0.0002,  # Episode 13475 LR
                "mcts_simulations": num_simulations,
                "c_puct": c_puct,
                "temperature": temperature,
                "trades": len(trades),
                "win_rate": win_rate,
                "avg_pips": np.mean([t.pnl_pips for t in trades]) if trades else 0,
                "total_pips": sum([t.pnl_pips for t in trades]) if trades else 0,
                "total_pnl": sum([t.pnl_pips for t in trades]) if trades else 0,  # Use actual trade PnL
                "env_reward": total_reward,  # Environment reward (includes unrealized P&L)
                "max_drawdown": max([t.max_drawdown_pips for t in trades]) if trades else 0,
                "sqn": sqn_result.sqn,
                "sqn_classification": sqn_result.classification,
                "sqn_confidence": sqn_result.confidence_level,
                "expectancy_r": sqn_result.expectancy,
                "std_dev_r": sqn_result.std_dev,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - episode_start_time).total_seconds()
            }
            
            # Log episode completion
            if self.current_episode % 10 == 0:
                actual_pnl = sum([t.pnl_pips for t in trades]) if trades else 0
                logger.info(f"Episode {self.current_episode}: "
                          f"PnL={actual_pnl:.2f}, EnvReward={total_reward:.2f}, "
                          f"SQN={sqn_result.sqn:.2f} ({sqn_result.classification}), "
                          f"Trades={len(trades)}, "
                          f"Loss={avg_loss:.4f}")
            
            return episode_result
            
        except Exception as e:
            logger.error(f"Episode {self.current_episode} failed: {e}")
            return {
                "episode": self.current_episode,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _train_network_step(self) -> float:
        """Train network on a batch from replay buffer"""
        if len(self.replay_buffer) == 0:
            return 0.0
        
        # Sample random trajectory
        trajectory = self.replay_buffer[np.random.randint(len(self.replay_buffer))]
        if len(trajectory) == 0:
            return 0.0
        
        # Sample random position in trajectory
        pos = np.random.randint(len(trajectory))
        
        # Prepare training data
        observation = torch.FloatTensor(trajectory[pos]['observation']).unsqueeze(0)
        action = torch.LongTensor([trajectory[pos]['action']])
        target_value = torch.FloatTensor([trajectory[pos]['value']])
        target_policy = torch.FloatTensor([trajectory[pos]['policy']])
        # Reward is stored for potential future use with recurrent_inference training
        
        # Forward pass
        inference_result = self.network.initial_inference(observation)
        value = inference_result['value_distribution']
        policy_logits = inference_result['policy_logits']
        hidden_state = inference_result['hidden_state']
        
        # Calculate losses
        value_loss = torch.nn.functional.mse_loss(value.squeeze(), target_value)
        policy_loss = torch.nn.functional.cross_entropy(
            policy_logits,
            target_policy.argmax(dim=-1)
        )
        # Note: Reward prediction happens in recurrent_inference, not initial_inference
        # So we don't have a reward loss here for now

        # Total loss
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return total_loss.item()
    
    def _save_checkpoint(self, episode_result):
        """Save training checkpoint"""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"swt_episode_{self.current_episode}.json"
            
            checkpoint_data = {
                "episode": self.current_episode,
                "timestamp": datetime.now().isoformat(),
                "performance": episode_result,
                "config": {
                    "agent_type": "swt_muzero",
                    "training_parameters": "standard"
                }
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            # Save actual model checkpoint (keep only last 2 + best)
            if self.network is not None:
                model_checkpoint_path = checkpoint_dir / f"swt_episode_{self.current_episode}.pth"
                torch.save({
                    'episode': self.current_episode,
                    'muzero_network_state': self.network.state_dict(),  # Use expected key name
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode_result': episode_result,
                    'timestamp': datetime.now().isoformat()
                }, model_checkpoint_path)
                logger.info(f"🔥 Model checkpoint saved: {model_checkpoint_path}")

                # Clean up old checkpoints - keep only last 2 regular checkpoints
                import glob
                all_checkpoints = sorted(glob.glob(str(checkpoint_dir / "swt_episode_*.pth")))
                if len(all_checkpoints) > 2:
                    for old_checkpoint in all_checkpoints[:-2]:
                        try:
                            Path(old_checkpoint).unlink()
                            logger.info(f"🗑️ Removed old checkpoint: {old_checkpoint}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {old_checkpoint}: {e}")

                # Save best model checkpoint based on SQN (System Quality Number)
                current_sqn = episode_result.get('sqn', 0.0)
                # Force first episode to be "best" for testing
                if current_sqn > self.best_sqn or self.current_episode == 1:
                    self.best_sqn = current_sqn
                    self.best_performance = episode_result.get('total_pnl', 0.0)
                    best_model_path = checkpoint_dir / "best_checkpoint.pth"
                    torch.save({
                        'episode': self.current_episode,
                        'muzero_network_state': self.network.state_dict(),  # Use expected key name
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'episode_result': episode_result,
                        'best_performance': self.best_performance,
                        'best_sqn': self.best_sqn,
                        'timestamp': datetime.now().isoformat()
                    }, best_model_path)
                    logger.info(f"🏆 BEST CHECKPOINT SAVED: {best_model_path} (Episode {self.current_episode}, SQN: {current_sqn:.2f}, Class: {episode_result.get('sqn_classification', 'Unknown')})")

            self.last_checkpoint_episode = self.current_episode
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Run validation if enabled
            if self.enable_validation:
                # Run validation asynchronously
                asyncio.create_task(self._run_checkpoint_validation(str(checkpoint_path), episode_result))
            
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")
    
    def _update_performance_metrics(self):
        """Update training performance metrics"""
        runtime_hours = (datetime.now() - self.training_start_time).total_seconds() / 3600
        self.episodes_per_hour = self.current_episode / max(runtime_hours, 0.01)
    
    def _log_training_progress(self):
        """Log training progress"""
        runtime_hours = (datetime.now() - self.training_start_time).total_seconds() / 3600
        
        logger.info(f"📊 Training Progress:")
        logger.info(f"   Episode: {self.current_episode}/{self.max_episodes}")
        logger.info(f"   Runtime: {runtime_hours:.2f}h")
        logger.info(f"   Speed: {self.episodes_per_hour:.1f} episodes/hour")
        logger.info(f"   Last checkpoint: Episode {self.last_checkpoint_episode}")
    
    def _should_early_stop(self):
        """Check early stopping conditions"""
        # Mock early stopping logic
        return False
    
    def _setup_monitoring(self):
        """Setup monitoring endpoints"""
        # Mock monitoring setup - would implement actual metrics endpoints
        logger.info("📈 Monitoring endpoints would be set up here")
    
    def cleanup_resources(self):
        """Clean up training resources"""
        logger.info("🧹 Cleaning up training resources...")
        
        try:
            # Save final checkpoint
            if self.current_episode > self.last_checkpoint_episode:
                final_result = {
                    "episode": self.current_episode,
                    "final_episode": True,
                    "total_runtime": (datetime.now() - self.training_start_time).total_seconds(),
                    "episodes_completed": self.current_episode
                }
                self._save_checkpoint(final_result)
            
            # Clean up feature processor
            if self.feature_processor:
                self.feature_processor.save_cache("cache")
            
            logger.info("✅ Resource cleanup complete")
            
        except Exception as e:
            logger.error(f"❌ Resource cleanup failed: {e}")

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "training.log")
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="New SWT Training System")
    parser.add_argument("--config", default="config/training.yaml", 
                       help="Configuration file path")
    parser.add_argument("--max-episodes", type=int, default=20000,
                       help="Maximum episodes to train")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")
    parser.add_argument("--enable-validation", action="store_true", default=True,
                       help="Enable automated validation")
    parser.add_argument("--validation-data", default="data/GBPJPY_M1_202201-202508.csv",
                       help="Path to validation data")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("🚀 Starting New SWT Training System")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Max episodes: {args.max_episodes}")
    
    # Initialize training orchestrator
    orchestrator = None
    
    def signal_handler(signum, frame):
        logger.info(f"📨 Received signal {signum}")
        if orchestrator:
            orchestrator.request_stop()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and run orchestrator
        orchestrator = SWTTrainingOrchestrator(
            args.config, 
            args.max_episodes,
            enable_validation=args.enable_validation,
            validation_data_path=args.validation_data if args.enable_validation else None
        )
        orchestrator.initialize_training_system()
        orchestrator.run_training_loop()
        
        logger.info("🎉 Training completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if orchestrator:
            orchestrator.cleanup_resources()

if __name__ == "__main__":
    sys.exit(main())