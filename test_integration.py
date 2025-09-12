#!/usr/bin/env python3
"""
SWT Integration Test Suite
Comprehensive testing of the complete SWT system end-to-end
"""

import sys
import os
import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import json
from typing import Dict, Any

# Add to Python path
sys.path.append(str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_core.types import AgentType, ProcessState
from swt_core.exceptions import ConfigurationError, CheckpointError, InferenceError
from swt_features.feature_processor import FeatureProcessor
from swt_features.position_features import PositionFeatures
from swt_features.market_features import MarketFeatures
from swt_features.wst_transform import WSTTransform
from swt_inference.checkpoint_loader import CheckpointLoader
from swt_inference.mcts_engine import MCTSEngine
from swt_inference.inference_engine import SWTInferenceEngine
from swt_inference.agent_factory import AgentFactory

# Test constants
EPISODE_13475_PARAMS = {
    'mcts_simulations': 15,
    'c_puct': 1.25,
    'wst_J': 2,
    'wst_Q': 6,
    'position_features_dim': 9,
    'market_features_dim': 128,
    'observation_dim': 137
}

class TestSWTIntegration:
    """Comprehensive integration tests for SWT system"""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="swt_test_"))
        self.config_dir = self.test_dir / "config"
        self.checkpoint_dir = self.test_dir / "checkpoints"
        self.cache_dir = self.test_dir / "cache"
        
        # Create directories
        for dir_path in [self.config_dir, self.checkpoint_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        self.create_test_config()
        
        # Create mock checkpoint
        self.create_mock_checkpoint()
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_config(self):
        """Create test configuration file"""
        config_data = {
            'agent_system': 'stochastic_muzero',
            'mcts_simulations': EPISODE_13475_PARAMS['mcts_simulations'],
            'c_puct': EPISODE_13475_PARAMS['c_puct'],
            'wst_J': EPISODE_13475_PARAMS['wst_J'],
            'wst_Q': EPISODE_13475_PARAMS['wst_Q'],
            'position_features_dim': EPISODE_13475_PARAMS['position_features_dim'],
            'market_features_dim': EPISODE_13475_PARAMS['market_features_dim'],
            'observation_dim': EPISODE_13475_PARAMS['observation_dim'],
            'episode_13475_mode': True,
            'normalization_params': {
                'duration_scale': 720.0,
                'pnl_scale': 100.0,
                'volume_scale': 1000.0,
                'price_scale': 0.0001,
                'spread_scale': 0.0001
            }
        }
        
        config_path = self.config_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        self.config_path = str(config_path)
    
    def create_mock_checkpoint(self):
        """Create mock checkpoint for testing"""
        import torch
        
        # Create mock networks
        networks = {
            'representation_network': torch.nn.Linear(137, 64),
            'dynamics_network': torch.nn.Linear(64, 64),
            'prediction_network': torch.nn.Linear(64, 32),
            'reward_network': torch.nn.Linear(64, 1),
            'value_network': torch.nn.Linear(64, 1)
        }
        
        checkpoint_data = {
            'networks': {name: net.state_dict() for name, net in networks.items()},
            'episode': 13475,
            'config': EPISODE_13475_PARAMS,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / "episode_13475.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        self.checkpoint_path = str(checkpoint_path)

    def test_config_manager_integration(self):
        """Test configuration manager integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        # Verify Episode 13475 compatibility
        config_manager.force_episode_13475_mode()
        
        assert config.agent_system == AgentType.STOCHASTIC_MUZERO
        assert config.mcts_simulations == EPISODE_13475_PARAMS['mcts_simulations']
        assert config.c_puct == EPISODE_13475_PARAMS['c_puct']
        assert config.episode_13475_mode == True
    
    def test_position_features_integration(self):
        """Test position features processing"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        position_features = PositionFeatures(config)
        
        # Mock position data
        position_data = {
            'position_size': 0.05,
            'unrealized_pnl': 25.0,
            'entry_price': 1.2000,
            'holding_time': 120,  # minutes
            'daily_pnl': 150.0
        }
        
        features = position_features.extract_features(position_data)
        
        assert features.shape == (EPISODE_13475_PARAMS['position_features_dim'],)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
        # Verify normalization
        assert np.all(features >= -5.0) and np.all(features <= 5.0)
    
    def test_market_features_integration(self):
        """Test market features processing with WST"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        market_features = MarketFeatures(config)
        
        # Generate mock price data
        np.random.seed(42)
        prices = 1.2000 + np.cumsum(np.random.randn(300) * 0.0001)
        volumes = 100 + np.random.exponential(50, 300)
        
        market_data = []
        for i in range(300):
            market_data.append({
                'timestamp': datetime.now(),
                'price': prices[i],
                'volume': volumes[i],
                'bid': prices[i] - 0.0002,
                'ask': prices[i] + 0.0002
            })
        
        # Process features
        features = market_features.extract_features(market_data)
        
        assert features.shape == (EPISODE_13475_PARAMS['market_features_dim'],)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_feature_processor_integration(self):
        """Test complete feature processor integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        feature_processor = FeatureProcessor(config)
        
        # Mock market data
        market_data = {
            'price': 1.2000,
            'volume': 150.0,
            'timestamp': datetime.now(),
            'bid': 1.1998,
            'ask': 1.2002
        }
        
        # Mock position data
        position_info = {
            'position_size': 0.03,
            'unrealized_pnl': 15.0,
            'entry_price': 1.1995,
            'holding_time': 60,
            'daily_pnl': 75.0
        }
        
        observation = feature_processor.process_observation(
            market_data=market_data,
            position_info=position_info
        )
        
        assert observation.shape == (EPISODE_13475_PARAMS['observation_dim'],)
        assert not np.any(np.isnan(observation))
        assert not np.any(np.isinf(observation))
    
    def test_checkpoint_loader_integration(self):
        """Test checkpoint loading integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        assert 'networks' in checkpoint_data
        assert 'episode' in checkpoint_data
        assert checkpoint_data['episode'] == 13475
        
        # Verify network structures
        networks = checkpoint_data['networks']
        expected_networks = [
            'representation_network',
            'dynamics_network', 
            'prediction_network',
            'reward_network',
            'value_network'
        ]
        
        for net_name in expected_networks:
            assert net_name in networks
    
    def test_mcts_engine_integration(self):
        """Test MCTS engine integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        # Mock networks
        networks = {
            'representation_network': lambda x: np.random.randn(64),
            'dynamics_network': lambda x: (np.random.randn(64), np.random.randn(1)),
            'prediction_network': lambda x: np.random.randn(3),  # 3 actions
            'value_network': lambda x: np.random.randn(1),
            'reward_network': lambda x: np.random.randn(1)
        }
        
        mcts_engine = MCTSEngine(networks, config)
        
        # Test observation
        observation = np.random.randn(EPISODE_13475_PARAMS['observation_dim'])
        
        result = mcts_engine.search(observation)
        
        assert 'action' in result
        assert 'action_probs' in result
        assert 'value' in result
        assert 'confidence' in result
        
        assert result['action'] in [0, 1, 2]  # Valid actions
        assert len(result['action_probs']) == 3
        assert np.isclose(np.sum(result['action_probs']), 1.0)
    
    def test_inference_engine_integration(self):
        """Test complete inference engine integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        # Load checkpoint
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        # Create inference engine
        inference_engine = SWTInferenceEngine(
            networks=checkpoint_data['networks'],
            config=config
        )
        
        # Test observation
        observation = np.random.randn(EPISODE_13475_PARAMS['observation_dim'])
        
        result = inference_engine.run_inference(observation)
        
        assert 'action' in result
        assert 'confidence' in result
        assert 'diagnostics' in result
        
        diagnostics = result['diagnostics']
        assert 'inference_time_ms' in diagnostics
        assert 'mcts_visits' in diagnostics
        assert 'episode_13475_compatible' in diagnostics
        
        assert diagnostics['episode_13475_compatible'] == True
    
    def test_agent_factory_integration(self):
        """Test agent factory integration"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        agent_factory = AgentFactory(config)
        
        # Test agent creation
        agent = agent_factory.create_agent(AgentType.STOCHASTIC_MUZERO)
        
        assert agent is not None
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'train')
    
    def test_end_to_end_trading_cycle(self):
        """Test complete end-to-end trading cycle"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        # Initialize all components
        feature_processor = FeatureProcessor(config)
        
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        inference_engine = SWTInferenceEngine(
            networks=checkpoint_data['networks'],
            config=config
        )
        
        # Simulate trading cycle
        market_data = {
            'price': 1.2000,
            'volume': 200.0,
            'timestamp': datetime.now(),
            'bid': 1.1998,
            'ask': 1.2002
        }
        
        position_info = {
            'position_size': 0.0,
            'unrealized_pnl': 0.0,
            'entry_price': 0.0,
            'holding_time': 0,
            'daily_pnl': 0.0
        }
        
        # Process observation
        start_time = time.time()
        observation = feature_processor.process_observation(
            market_data=market_data,
            position_info=position_info
        )
        feature_time = time.time() - start_time
        
        # Run inference
        start_time = time.time()
        result = inference_engine.run_inference(observation)
        inference_time = time.time() - start_time
        
        # Verify performance requirements
        assert feature_time < 0.01  # 10ms feature processing
        assert inference_time < 0.2  # 200ms inference time
        
        # Verify result structure
        assert 'action' in result
        assert 'confidence' in result
        
        # Verify Episode 13475 compatibility
        diagnostics = result['diagnostics']
        assert diagnostics['episode_13475_compatible'] == True
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        feature_processor = FeatureProcessor(config)
        
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        inference_engine = SWTInferenceEngine(
            networks=checkpoint_data['networks'],
            config=config
        )
        
        # Benchmark feature processing
        market_data = {
            'price': 1.2000,
            'volume': 150.0,
            'timestamp': datetime.now(),
            'bid': 1.1998,
            'ask': 1.2002
        }
        
        position_info = {
            'position_size': 0.02,
            'unrealized_pnl': 10.0,
            'entry_price': 1.1990,
            'holding_time': 30,
            'daily_pnl': 50.0
        }
        
        # Run multiple cycles for benchmarking
        feature_times = []
        inference_times = []
        
        for _ in range(10):
            # Feature processing
            start_time = time.time()
            observation = feature_processor.process_observation(
                market_data=market_data,
                position_info=position_info
            )
            feature_times.append(time.time() - start_time)
            
            # Inference
            start_time = time.time()
            result = inference_engine.run_inference(observation)
            inference_times.append(time.time() - start_time)
        
        # Performance assertions
        avg_feature_time = np.mean(feature_times)
        avg_inference_time = np.mean(inference_times)
        
        assert avg_feature_time < 0.01  # < 10ms average
        assert avg_inference_time < 0.2   # < 200ms average
        
        p95_feature_time = np.percentile(feature_times, 95)
        p95_inference_time = np.percentile(inference_times, 95)
        
        assert p95_feature_time < 0.02   # < 20ms P95
        assert p95_inference_time < 0.5  # < 500ms P95
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        # Test invalid checkpoint
        with pytest.raises(CheckpointError):
            checkpoint_loader = CheckpointLoader(config)
            checkpoint_loader.load_checkpoint("nonexistent_checkpoint.pt")
        
        # Test invalid observation
        feature_processor = FeatureProcessor(config)
        
        with pytest.raises(ValueError):
            feature_processor.process_observation(
                market_data=None,  # Invalid market data
                position_info={}
            )
        
        # Test inference with invalid observation
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        inference_engine = SWTInferenceEngine(
            networks=checkpoint_data['networks'],
            config=config
        )
        
        with pytest.raises(InferenceError):
            invalid_observation = np.full(10, np.nan)  # Wrong shape and NaN values
            inference_engine.run_inference(invalid_observation)
    
    @pytest.mark.asyncio
    async def test_async_integration(self):
        """Test asynchronous integration for live trading"""
        config_manager = ConfigManager()
        config = config_manager.load_config(self.config_path)
        
        feature_processor = FeatureProcessor(config)
        
        checkpoint_loader = CheckpointLoader(config)
        checkpoint_data = checkpoint_loader.load_checkpoint(self.checkpoint_path)
        
        inference_engine = SWTInferenceEngine(
            networks=checkpoint_data['networks'],
            config=config
        )
        
        async def async_trading_cycle():
            market_data = {
                'price': 1.2000,
                'volume': 180.0,
                'timestamp': datetime.now(),
                'bid': 1.1998,
                'ask': 1.2002
            }
            
            position_info = {
                'position_size': 0.01,
                'unrealized_pnl': 5.0,
                'entry_price': 1.1995,
                'holding_time': 15,
                'daily_pnl': 25.0
            }
            
            # Async feature processing
            observation = await asyncio.to_thread(
                feature_processor.process_observation,
                market_data=market_data,
                position_info=position_info
            )
            
            # Async inference
            result = await asyncio.to_thread(
                inference_engine.run_inference,
                observation
            )
            
            return result
        
        # Run async trading cycle
        result = await async_trading_cycle()
        
        assert 'action' in result
        assert 'confidence' in result
        assert result['diagnostics']['episode_13475_compatible'] == True

def test_episode_13475_parameter_compatibility():
    """Dedicated test for Episode 13475 parameter compatibility"""
    
    # Test all critical parameters match exactly
    config_path = Path(__file__).parent / "config" / "episode_13475_reference.yaml"
    
    if config_path.exists():
        config_manager = ConfigManager()
        config = config_manager.load_config(str(config_path))
        
        assert config.mcts_simulations == 15
        assert config.c_puct == 1.25
        assert config.wst_J == 2
        assert config.wst_Q == 6
        assert config.position_features_dim == 9
        assert config.market_features_dim == 128
        assert config.observation_dim == 137
        
        # Verify normalization parameters
        norm_params = config.normalization_params
        assert norm_params['duration_scale'] == 720.0
        assert norm_params['pnl_scale'] == 100.0

if __name__ == "__main__":
    # Run integration tests
    import subprocess
    import sys
    
    print("🧪 Running SWT Integration Tests...")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--color=yes"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
        sys.exit(result.returncode)