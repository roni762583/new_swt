#!/usr/bin/env python3
"""
Robust Checkpoint Loader for Large Files
Handles checkpoints of any size without memory issues
"""

import gc
import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustCheckpointLoader:
    """
    Memory-efficient checkpoint loader that handles large files
    """
    
    def __init__(self, memory_limit_gb: float = 4.0):
        """
        Initialize robust loader with memory limits

        Args:
            memory_limit_gb: Maximum memory to use in GB
        """
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics (simplified without psutil)"""
        # Simple memory check using /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                mem_total = int(lines[0].split()[1]) / (1024 * 1024)  # Convert KB to GB
                mem_available = int(lines[2].split()[1]) / (1024 * 1024)
                return {
                    'total_gb': mem_total,
                    'available_gb': mem_available,
                    'used_gb': mem_total - mem_available,
                    'percent': ((mem_total - mem_available) / mem_total) * 100
                }
        except:
            # Fallback if /proc/meminfo not available
            return {
                'total_gb': 0,
                'available_gb': self.memory_limit_bytes / (1024**3),
                'used_gb': 0,
                'percent': 0
            }

    def check_memory_available(self, required_gb: float) -> bool:
        """Check if enough memory is available"""
        mem_stats = self.get_memory_usage()
        available = mem_stats['available_gb']

        if available < required_gb:
            logger.warning(f"⚠️ Low memory: {available:.1f}GB available, {required_gb:.1f}GB required")
            return False
        return True
    
    def estimate_checkpoint_memory(self, checkpoint_path: Path) -> float:
        """Estimate memory required for checkpoint"""
        file_size_gb = checkpoint_path.stat().st_size / (1024**3)
        # Estimate 2.5x file size for loading overhead
        estimated_memory_gb = file_size_gb * 2.5
        return estimated_memory_gb
    
    def clean_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove unnecessary data from checkpoint to save memory
        
        Args:
            checkpoint: Raw checkpoint data
            
        Returns:
            Cleaned checkpoint with only essential data
        """
        logger.info("🧹 Cleaning checkpoint to reduce memory usage...")
        
        # Keys to keep for validation
        essential_keys = {
            'muzero_network_state',
            'config',
            'episode',
            'training_step',
            'muzero_weights'  # Alternative key name
        }
        
        # Keys to definitely remove (memory hogs)
        remove_keys = {
            'replay_buffer',
            'priority_replay_buffer',
            'buffer',
            'optimizer_state',
            'scheduler_state',
            'training_history',
            'game_history',
            'self_play_buffer',
            'memory_buffer',
            'trajectories',
            'episode_buffer',
            'experience_replay',
            'agent_state',
            'training_state',
            'stats',
            'metrics'
        }
        
        # Calculate original size
        original_keys = set(checkpoint.keys())
        
        # Remove unnecessary keys
        cleaned = {}
        for key in checkpoint:
            if key in essential_keys:
                cleaned[key] = checkpoint[key]
            elif key not in remove_keys:
                # Check size of unknown keys
                try:
                    size_mb = sys.getsizeof(checkpoint[key]) / (1024**2)
                    if size_mb < 100:  # Keep small unknown keys
                        cleaned[key] = checkpoint[key]
                    else:
                        logger.info(f"  Removing large key '{key}': {size_mb:.1f}MB")
                except:
                    cleaned[key] = checkpoint[key]
        
        # Log cleaning results
        removed_keys = original_keys - set(cleaned.keys())
        if removed_keys:
            logger.info(f"  Removed keys: {removed_keys}")
        
        # Force garbage collection
        del checkpoint
        gc.collect()
        
        return cleaned
    
    def load_checkpoint_safely(self, checkpoint_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Safely load checkpoint with memory management
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (networks, config)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Log initial memory state
        logger.info(f"📊 Initial memory: {self.get_memory_usage()}")
        
        # Estimate memory requirement
        file_size_gb = checkpoint_path.stat().st_size / (1024**3)
        estimated_memory_gb = self.estimate_checkpoint_memory(checkpoint_path)
        
        logger.info(f"📦 Loading checkpoint: {checkpoint_path.name}")
        logger.info(f"  File size: {file_size_gb:.2f}GB")
        logger.info(f"  Estimated memory: {estimated_memory_gb:.2f}GB")
        
        # Check memory availability
        if not self.check_memory_available(estimated_memory_gb):
            logger.warning("⚠️ Attempting to free memory...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if not self.check_memory_available(estimated_memory_gb * 0.7):
                raise MemoryError(f"Insufficient memory for {file_size_gb:.2f}GB checkpoint")
        
        try:
            # Load with map_location to avoid GPU memory issues
            logger.info("⏳ Loading checkpoint (this may take a while for large files)...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location='cpu',
                    weights_only=False
                )
            
            logger.info(f"✅ Checkpoint loaded, keys: {list(checkpoint.keys())}")
            
            # Clean checkpoint to save memory
            checkpoint = self.clean_checkpoint(checkpoint)
            
            # Extract config
            config = self._extract_config(checkpoint)
            
            # Load networks efficiently
            networks = self._load_networks_efficiently(checkpoint, config)
            
            # Final cleanup
            del checkpoint
            gc.collect()
            
            # Log final memory state
            logger.info(f"📊 Final memory: {self.get_memory_usage()}")
            
            return networks, config
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            gc.collect()
            raise
    
    def _extract_config(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from checkpoint
        """
        config = {}
        
        # Try to extract from full config first
        if 'config' in checkpoint:
            if isinstance(checkpoint['config'], dict):
                if 'full_config' in checkpoint['config']:
                    full_config = checkpoint['config']['full_config']
                    if 'muzero_config' in full_config:
                        muzero_cfg = full_config['muzero_config']
                        config['hidden_dim'] = muzero_cfg.get('hidden_dim', 256)
                        config['support_size'] = muzero_cfg.get('support_size', 601)
                        config['action_space_size'] = muzero_cfg.get('action_space_size', 4)
                        config['num_blocks'] = muzero_cfg.get('num_blocks', 2)
                        config['latent_dim'] = muzero_cfg.get('latent_dim', 16)
                else:
                    # Fallback to direct config
                    config = checkpoint['config']
        
        # Ensure we have minimum required config
        config.setdefault('hidden_dim', 256)
        config.setdefault('support_size', 601)
        config.setdefault('action_space_size', 4)
        
        logger.info(f"📋 Extracted config: hidden_dim={config['hidden_dim']}, support_size={config['support_size']}")
        
        return config
    
    def _load_networks_efficiently(self, checkpoint: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """
        Load networks efficiently with proper architecture
        """
        # Import network architecture and config
        sys.path.insert(0, '/workspace')
        from experimental_research.swt_models.swt_stochastic_networks import (
            SWTStochasticMuZeroNetwork, SWTStochasticMuZeroConfig
        )

        # Create config object with correct parameters
        network_config = SWTStochasticMuZeroConfig(
            final_input_dim=137,  # 128 WST + 9 position features
            num_actions=config['action_space_size'],
            support_size=config['support_size'],
            hidden_dim=config['hidden_dim'],
            representation_blocks=config.get('num_blocks', 2),
            dynamics_blocks=config.get('num_blocks', 2),
            prediction_blocks=2,
            latent_z_dim=config.get('latent_dim', 16),
            dropout_rate=0.1,
            layer_norm=True,
            use_optimized_blocks=True
        )

        # Create network with config
        networks = SWTStochasticMuZeroNetwork(network_config)
        
        # Load weights
        state_dict_key = None
        if 'muzero_network_state' in checkpoint:
            state_dict_key = 'muzero_network_state'
        elif 'muzero_weights' in checkpoint:
            state_dict_key = 'muzero_weights'
        elif 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'
        
        if state_dict_key:
            try:
                networks.load_state_dict(checkpoint[state_dict_key])
                logger.info("✅ Network weights loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load weights directly: {e}")
                # Try partial loading
                self._load_weights_partially(networks, checkpoint[state_dict_key])
        else:
            logger.warning("⚠️ No network weights found in checkpoint")
        
        return networks
    
    def _load_weights_partially(self, networks: Any, state_dict: Dict[str, Any]):
        """
        Load weights partially, skipping mismatched layers
        """
        model_dict = networks.state_dict()
        filtered_dict = {}
        
        for key, value in state_dict.items():
            if key in model_dict:
                if model_dict[key].shape == value.shape:
                    filtered_dict[key] = value
                else:
                    logger.warning(f"  Skipping {key}: shape mismatch")
        
        model_dict.update(filtered_dict)
        networks.load_state_dict(model_dict)
        logger.info(f"✅ Partially loaded {len(filtered_dict)}/{len(state_dict)} weights")

    def aggressive_optimize(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply aggressive optimization for very large checkpoints

        Args:
            checkpoint: Checkpoint data

        Returns:
            Aggressively optimized checkpoint
        """
        logger.info("🔥 Applying AGGRESSIVE optimization...")

        # For inference, we only need the network weights
        optimized = {}

        # Keep only the absolute minimum for inference
        network_keys = {
            'muzero_network_state',
            'muzero_weights',
            'model_state_dict',
            'network_state_dict',
            'state_dict'
        }

        # Find and keep only network weights
        found_network = False
        for key in network_keys:
            if key in checkpoint:
                optimized[key] = checkpoint[key]
                found_network = True
                logger.info(f"  Keeping network weights: {key}")
                break

        if not found_network:
            # If no standard key, look for anything with 'network' or 'model'
            for key in checkpoint:
                if 'network' in key.lower() or 'model' in key.lower():
                    if not any(skip in key.lower() for skip in ['optimizer', 'scheduler', 'buffer', 'history']):
                        optimized[key] = checkpoint[key]
                        logger.info(f"  Keeping potential network: {key}")

        # Keep minimal config
        if 'config' in checkpoint:
            if isinstance(checkpoint['config'], dict):
                # Extract only essential config
                minimal_config = {}
                if 'full_config' in checkpoint['config']:
                    fc = checkpoint['config']['full_config']
                    if 'muzero_config' in fc:
                        mc = fc['muzero_config']
                        minimal_config = {
                            'hidden_dim': mc.get('hidden_dim', 256),
                            'support_size': mc.get('support_size', 601),
                            'action_space_size': mc.get('action_space_size', 4),
                            'num_blocks': mc.get('num_blocks', 2),
                            'latent_dim': mc.get('latent_dim', 16)
                        }
                else:
                    # Direct config
                    for key in ['hidden_dim', 'support_size', 'action_space_size', 'num_blocks', 'latent_dim']:
                        if key in checkpoint['config']:
                            minimal_config[key] = checkpoint['config'][key]

                if minimal_config:
                    optimized['config'] = minimal_config
                    logger.info(f"  Keeping minimal config: {list(minimal_config.keys())}")

        # Keep episode number if present
        for key in ['episode', 'training_step', 'epoch']:
            if key in checkpoint:
                optimized[key] = checkpoint[key]
                logger.info(f"  Keeping metadata: {key}={checkpoint[key]}")

        # Calculate size reduction
        import sys
        original_size = sum(sys.getsizeof(v) for v in checkpoint.values() if v is not None) / (1024**2)
        optimized_size = sum(sys.getsizeof(v) for v in optimized.values() if v is not None) / (1024**2)
        reduction = (1 - optimized_size/original_size) * 100 if original_size > 0 else 0

        logger.info(f"  Memory reduced by ~{reduction:.1f}% (in-memory estimate)")
        logger.info(f"  Keys: {len(checkpoint)} → {len(optimized)}")

        # Force cleanup
        del checkpoint
        gc.collect()

        return optimized


def optimize_checkpoint_for_validation(input_path: str, output_path: str, aggressive: bool = False):
    """
    Optimize checkpoint by removing unnecessary data

    Args:
        input_path: Path to original checkpoint
        output_path: Path to save optimized checkpoint
        aggressive: If True, apply more aggressive optimization
    """
    logger.info(f"🔧 Optimizing checkpoint: {input_path}")
    if aggressive:
        logger.info("  Using AGGRESSIVE optimization mode")

    loader = RobustCheckpointLoader()

    # Load checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # First pass: standard cleaning
    cleaned = loader.clean_checkpoint(checkpoint)

    if aggressive:
        # Second pass: aggressive optimization
        cleaned = loader.aggressive_optimize(cleaned)

    # Save optimized version with compression
    torch.save(cleaned, output_path, pickle_protocol=4, _use_new_zipfile_serialization=True)

    # Report size reduction
    original_size = Path(input_path).stat().st_size / (1024**3)
    optimized_size = Path(output_path).stat().st_size / (1024**3)
    reduction = (1 - optimized_size/original_size) * 100

    logger.info(f"✅ Optimization complete:")
    logger.info(f"  Original: {original_size:.2f}GB")
    logger.info(f"  Optimized: {optimized_size:.2f}GB")
    logger.info(f"  Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust checkpoint loader')
    parser.add_argument('--load', help='Load checkpoint safely')
    parser.add_argument('--optimize', help='Optimize checkpoint for validation')
    parser.add_argument('--output', help='Output path for optimized checkpoint')
    parser.add_argument('--memory-limit', type=float, default=4.0, help='Memory limit in GB')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive optimization (minimal inference-only)')

    args = parser.parse_args()

    if args.optimize and args.output:
        optimize_checkpoint_for_validation(args.optimize, args.output, aggressive=args.aggressive)
    elif args.load:
        loader = RobustCheckpointLoader(memory_limit_gb=args.memory_limit)
        networks, config = loader.load_checkpoint_safely(args.load)
        print(f"Successfully loaded checkpoint with config: {config}")