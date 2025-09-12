#!/usr/bin/env python3
"""
EfficientZero-Enhanced SWT System - Main Training Entry Point
Production-grade training orchestration with comprehensive validation

Integrates EfficientZero's three key enhancements into SWT-Enhanced Stochastic MuZero
Optimized for forex trading with AMDDP1 reward system and multiprocessing support

Author: SWT Research Team
Date: September 2025
Adherence: CLAUDE.md professional code standards
"""

import sys
import os
import argparse
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np

# Add experimental research to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging with production standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/efficientzero_main.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class EfficientZeroSystemValidator:
    """
    Comprehensive system validation for EfficientZero integration
    Validates all components before training execution
    """
    
    def __init__(self) -> None:
        """Initialize system validator"""
        self.validation_results: Dict[str, Any] = {}
        
    def validate_environment(self) -> bool:
        """Validate Python environment and dependencies"""
        
        logger.info("🔬 Validating EfficientZero environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                raise RuntimeError(f"Python 3.8+ required, got {python_version}")
            
            # Validate critical imports
            import torch
            import numpy as np
            import pandas as pd
            import gymnasium
            import einops
            import transformers
            
            # Test PyTorch functionality
            test_tensor = torch.randn(4, 4)
            torch.matmul(test_tensor, test_tensor.T)
            
            # Test Transformer
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(d_model=32, nhead=4)
            transformer = TransformerEncoder(encoder_layer, num_layers=1)
            test_input = torch.randn(2, 10, 32)
            transformer(test_input)
            
            self.validation_results['environment'] = {
                'status': 'PASSED',
                'python_version': str(python_version),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            logger.info("✅ Environment validation passed")
            return True
            
        except Exception as e:
            self.validation_results['environment'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"❌ Environment validation failed: {e}")
            return False
    
    def validate_swt_components(self) -> bool:
        """Validate SWT base system components"""
        
        logger.info("🔬 Validating SWT base components...")
        
        try:
            # Test SWT core imports
            from swt_models.swt_stochastic_networks import create_swt_stochastic_muzero_network
            from swt_core.swt_mcts import create_swt_stochastic_mcts
            from swt_environments.swt_forex_env import SWTForexEnvironment
            from swt_training.swt_trainer import SWTStochasticMuZeroTrainer
            
            # Test basic network creation
            networks = create_swt_stochastic_muzero_network(
                market_input_dim=128,
                position_input_dim=9,
                hidden_dim=128,
                action_space_size=4
            )
            
            # Test basic MCTS creation  
            mcts_config = {
                'num_simulations': 5,
                'c_puct': 1.0,
                'discount_factor': 0.997
            }
            mcts = create_swt_stochastic_mcts(mcts_config)
            
            self.validation_results['swt_components'] = {
                'status': 'PASSED',
                'networks_created': True,
                'mcts_created': True
            }
            
            logger.info("✅ SWT components validation passed")
            return True
            
        except Exception as e:
            self.validation_results['swt_components'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"❌ SWT components validation failed: {e}")
            return False
    
    def validate_efficientzero_components(self) -> bool:
        """Validate EfficientZero enhancement components"""
        
        logger.info("🔬 Validating EfficientZero components...")
        
        try:
            # Test EfficientZero module imports
            from value_prefix_network import SWTValuePrefixNetwork
            from consistency_loss import SWTConsistencyLoss
            from off_policy_correction import SWTOffPolicyCorrection
            
            # Test value-prefix network
            prefix_net = SWTValuePrefixNetwork(
                input_dim=32,
                architecture='transformer'
            )
            
            test_rewards = torch.randn(8, 5, 1)
            test_values = torch.randn(8, 5, 1)
            predicted_return, aux_outputs = prefix_net(test_rewards, test_values)
            
            if predicted_return.shape != (8, 1):
                raise RuntimeError(f"Invalid prefix network output shape: {predicted_return.shape}")
            
            # Test consistency loss
            consistency_loss = SWTConsistencyLoss()
            
            pred_latent = torch.randn(16, 256)
            true_latent = torch.randn(16, 256)
            loss_val, metrics = consistency_loss.compute_loss(pred_latent, true_latent)
            
            if not torch.is_tensor(loss_val) or loss_val.numel() != 1:
                raise RuntimeError(f"Invalid consistency loss output: {loss_val}")
            
            # Test off-policy correction
            corrector = SWTOffPolicyCorrection()
            
            horizon = corrector.compute_adaptive_horizon(
                data_age_steps=5,
                trajectory_length=20,
                market_volatility=0.3
            )
            
            if not isinstance(horizon, int) or horizon < 1:
                raise RuntimeError(f"Invalid horizon computation: {horizon}")
            
            self.validation_results['efficientzero_components'] = {
                'status': 'PASSED',
                'value_prefix_network': True,
                'consistency_loss': True,
                'off_policy_correction': True,
                'prefix_output_shape': list(predicted_return.shape),
                'consistency_loss_value': loss_val.item(),
                'adaptive_horizon': horizon
            }
            
            logger.info("✅ EfficientZero components validation passed")
            return True
            
        except Exception as e:
            self.validation_results['efficientzero_components'] = {
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            logger.error(f"❌ EfficientZero components validation failed: {e}")
            return False
    
    def validate_system_resources(self) -> bool:
        """Validate system resources for training"""
        
        logger.info("🔬 Validating system resources...")
        
        try:
            import psutil
            
            # Memory validation
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb < 4:
                logger.warning(f"Low memory detected: {memory_gb:.1f}GB (recommended: 8GB+)")
            
            # CPU validation
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_count < 2:
                logger.warning(f"Low CPU count: {cpu_count} (recommended: 4+)")
            
            # Disk space validation
            disk = psutil.disk_usage('/')
            disk_gb = disk.free / (1024**3)
            
            if disk_gb < 10:
                logger.warning(f"Low disk space: {disk_gb:.1f}GB (recommended: 50GB+)")
            
            self.validation_results['system_resources'] = {
                'status': 'PASSED',
                'memory_gb': memory_gb,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_count': cpu_count,
                'cpu_usage_percent': cpu_percent,
                'disk_free_gb': disk_gb
            }
            
            logger.info(f"✅ System resources validated: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
            return True
            
        except ImportError:
            logger.warning("psutil not available, skipping detailed resource validation")
            self.validation_results['system_resources'] = {
                'status': 'SKIPPED',
                'reason': 'psutil not available'
            }
            return True
            
        except Exception as e:
            self.validation_results['system_resources'] = {
                'status': 'FAILED', 
                'error': str(e)
            }
            logger.error(f"❌ System resource validation failed: {e}")
            return False
    
    def run_full_validation(self) -> bool:
        """Run complete system validation"""
        
        logger.info("🚀 Starting EfficientZero System Validation...")
        
        validation_steps = [
            ('environment', self.validate_environment),
            ('swt_components', self.validate_swt_components),
            ('efficientzero_components', self.validate_efficientzero_components),
            ('system_resources', self.validate_system_resources)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            logger.info(f"Running {step_name} validation...")
            
            if not validation_func():
                all_passed = False
                logger.error(f"❌ {step_name} validation FAILED")
            else:
                logger.info(f"✅ {step_name} validation PASSED")
        
        # Save validation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/validation_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        if all_passed:
            logger.info("🎉 ALL VALIDATIONS PASSED - EfficientZero system ready!")
        else:
            logger.error("💥 VALIDATION FAILURES DETECTED - Review results before proceeding")
        
        logger.info(f"Validation results saved: {results_path}")
        
        return all_passed


class EfficientZeroExperimentManager:
    """
    Manages EfficientZero experimental training runs
    Provides configuration, monitoring, and result management
    """
    
    def __init__(self, experiment_name: str) -> None:
        """
        Initialize experiment manager
        
        Args:
            experiment_name: Name for this experimental run
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.experiment_id = f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directories
        self.experiment_dir = Path(f"results/{self.experiment_id}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for experiment
        exp_log_handler = logging.FileHandler(
            self.experiment_dir / "experiment.log"
        )
        exp_log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(exp_log_handler)
        
        logger.info(f"🔬 Initialized experiment: {self.experiment_id}")
    
    def load_experiment_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load experiment configuration"""
        
        if config_path is None:
            config_path = "configs/efficientzero_experiment_config.json"
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded experiment config from: {config_path}")
            else:
                # Default configuration
                config = {
                    "training": {
                        "num_episodes": 100000,
                        "batch_size": 64,
                        "learning_rate": 0.0002,
                        "multiprocessing_workers": 4
                    },
                    "efficientzero": {
                        "consistency_loss": {
                            "enabled": True,
                            "weight": 1.0,
                            "temperature": 0.1
                        },
                        "value_prefix": {
                            "enabled": True,
                            "architecture": "transformer",
                            "weight": 0.5,
                            "hidden_dim": 64
                        },
                        "off_policy_correction": {
                            "enabled": True,
                            "min_horizon": 3,
                            "max_horizon": 8
                        }
                    },
                    "system": {
                        "reward_system": "amddp1",
                        "checkpoint_frequency": 1000,
                        "validation_frequency": 5000
                    }
                }
                logger.info("Using default experiment configuration")
        
            # Save config to experiment directory
            config_save_path = self.experiment_dir / "config.json"
            with open(config_save_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load experiment config: {e}")
            raise
    
    def run_training_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run training experiment with EfficientZero enhancements
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment results
        """
        logger.info(f"🚀 Starting training experiment: {self.experiment_id}")
        
        try:
            # Import training components (lazy import to pass validation)
            logger.info("Importing EfficientZero trainer...")
            
            # Note: This import will fail until efficientzero_trainer.py is implemented
            # For now, we'll create a placeholder that logs the intended training
            logger.info("⚠️  EfficientZero trainer not yet implemented")
            logger.info("This is expected during initial development phase")
            
            # Simulate training configuration
            training_config = config.get('training', {})
            efficientzero_config = config.get('efficientzero', {})
            
            logger.info("Training Configuration:")
            logger.info(f"  Episodes: {training_config.get('num_episodes', 100000)}")
            logger.info(f"  Batch Size: {training_config.get('batch_size', 64)}")
            logger.info(f"  Learning Rate: {training_config.get('learning_rate', 0.0002)}")
            
            logger.info("EfficientZero Configuration:")
            consistency = efficientzero_config.get('consistency_loss', {})
            logger.info(f"  Consistency Loss: {consistency.get('enabled', True)} "
                       f"(weight={consistency.get('weight', 1.0)})")
            
            prefix = efficientzero_config.get('value_prefix', {})
            logger.info(f"  Value Prefix: {prefix.get('enabled', True)} "
                       f"(architecture={prefix.get('architecture', 'transformer')})")
            
            correction = efficientzero_config.get('off_policy_correction', {})
            logger.info(f"  Off-Policy Correction: {correction.get('enabled', True)} "
                       f"(horizon={correction.get('min_horizon', 3)}-{correction.get('max_horizon', 8)})")
            
            # Placeholder for actual training
            # TODO: Implement EfficientZeroSWTTrainer integration
            
            # Simulate training results
            results = {
                'experiment_id': self.experiment_id,
                'status': 'DEVELOPMENT_PLACEHOLDER',
                'message': 'EfficientZero trainer implementation in progress',
                'config_validated': True,
                'components_ready': True,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
            
            # Save results
            results_path = self.experiment_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Experiment results saved: {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"Training experiment failed: {e}")
            logger.error(traceback.format_exc())
            
            # Save error results
            error_results = {
                'experiment_id': self.experiment_id,
                'status': 'FAILED',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
            
            results_path = self.experiment_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(error_results, f, indent=2, default=str)
            
            raise


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="EfficientZero-Enhanced SWT System - Main Training Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation modes
    parser.add_argument(
        '--validate-system', 
        action='store_true',
        help='Run comprehensive system validation only'
    )
    
    parser.add_argument(
        '--run-experiment',
        action='store_true', 
        help='Run EfficientZero training experiment'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='efficientzero_research',
        help='Name for experimental training run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to experiment configuration file'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=100000,
        help='Number of training episodes'
    )
    
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['transformer', 'tcn', 'lstm', 'conv1d'],
        default='transformer',
        help='Value-prefix network architecture'
    )
    
    parser.add_argument(
        '--consistency-weight',
        type=float,
        default=1.0,
        help='Weight for consistency loss'
    )
    
    parser.add_argument(
        '--prefix-weight', 
        type=float,
        default=0.5,
        help='Weight for value-prefix loss'
    )
    
    # System configuration
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of multiprocessing workers'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Computing device for training'
    )
    
    # Logging and debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser


def setup_logging(args: argparse.Namespace) -> None:
    """Setup logging configuration based on arguments"""
    
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level.upper())
    
    # Update root logger level
    logging.getLogger().setLevel(log_level)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)


def main() -> int:
    """Main entry point for EfficientZero system"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args)
    
    logger.info("🔬 EfficientZero-Enhanced SWT System Starting...")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Always run system validation first
        validator = EfficientZeroSystemValidator()
        validation_passed = validator.run_full_validation()
        
        if not validation_passed:
            logger.error("❌ System validation failed - cannot proceed with training")
            return 1
        
        if args.validate_system:
            logger.info("✅ System validation completed successfully")
            return 0
        
        # Run training experiment if requested or no specific mode specified
        if args.run_experiment or not any([args.validate_system]):
            
            # Create experiment manager
            experiment_manager = EfficientZeroExperimentManager(args.experiment_name)
            
            # Load configuration
            config = experiment_manager.load_experiment_config(args.config)
            
            # Override config with command line arguments
            if args.episodes:
                config.setdefault('training', {})['num_episodes'] = args.episodes
            if args.workers:
                config.setdefault('training', {})['multiprocessing_workers'] = args.workers
            if args.architecture:
                config.setdefault('efficientzero', {}).setdefault('value_prefix', {})['architecture'] = args.architecture
            if args.consistency_weight:
                config.setdefault('efficientzero', {}).setdefault('consistency_loss', {})['weight'] = args.consistency_weight
            if args.prefix_weight:
                config.setdefault('efficientzero', {}).setdefault('value_prefix', {})['weight'] = args.prefix_weight
            
            # Run experiment
            results = experiment_manager.run_training_experiment(config)
            
            logger.info(f"🎉 Experiment completed: {results['status']}")
            return 0 if results['status'] != 'FAILED' else 1
        
        logger.info("No operation mode specified, run with --help for options")
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        return 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)