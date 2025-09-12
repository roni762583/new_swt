#!/usr/bin/env python3
"""
Phase 2 Validation Script
Tests the shared feature processing implementation to ensure EXACT compatibility
between training and live systems.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add the new_swt directory to Python path
sys.path.append(str(Path(__file__).parent))

from swt_core.types import PositionState, PositionType, MarketState
from swt_core.config_manager import ConfigManager
from swt_features.wst_transform import WSTProcessor, WSTConfig
from swt_features.market_features import MarketFeatureExtractor, MarketDataPoint
from swt_features.position_features import PositionFeatureExtractor
from swt_features.feature_processor import FeatureProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_wst_transform():
    """Test WST transform with Episode 13475 parameters"""
    logger.info("🧪 Testing WST Transform...")
    
    # Episode 13475 WST configuration
    wst_config = WSTConfig(
        J=2,
        Q=6, 
        backend="manual",  # Use manual backend for testing
        output_dim=128,
        max_order=2,
        normalize=True,
        cache_enabled=True
    )
    
    processor = WSTProcessor(wst_config)
    
    # Test with synthetic price data
    price_series = np.sin(np.linspace(0, 4*np.pi, 256)) + 0.1 * np.random.randn(256)
    price_series = price_series.astype(np.float32)
    
    # Transform
    wst_features = processor.transform(price_series, cache_key="test_prices")
    
    # Validate
    assert wst_features.shape == (128,), f"Expected (128,), got {wst_features.shape}"
    assert wst_features.dtype == np.float32, f"Expected float32, got {wst_features.dtype}"
    assert not np.isnan(wst_features).any(), "WST features contain NaN"
    assert not np.isinf(wst_features).any(), "WST features contain Inf"
    
    logger.info(f"✅ WST Transform test passed - output shape: {wst_features.shape}")
    return wst_features

def test_position_features():
    """Test position feature extraction with exact training parameters"""
    logger.info("🧪 Testing Position Features...")
    
    # Episode 13475 normalization parameters
    normalization_config = {
        "duration_scale": 720.0,           # Normalize by 12-hour session
        "pnl_scale": 100.0,                # Normalize PnL by 100 pips
        "price_change_scale": 50.0,        # Normalize price changes by 50 pips
        "drawdown_scale": 50.0,            # Normalize drawdowns by 50 pips
        "accumulated_dd_scale": 100.0,     # Normalize accumulated DD by 100 pips
        "bars_since_dd_scale": 60.0        # Normalize by 1-hour period
    }
    
    extractor = PositionFeatureExtractor(normalization_config)
    
    # Test flat position (should return all zeros)
    flat_position = PositionState()
    current_price = 199.45
    
    features = extractor.extract_features(flat_position, current_price)
    
    # Validate flat position
    assert features.shape == (9,), f"Expected (9,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    assert np.allclose(features, 0.0), f"Flat position should be all zeros, got {features}"
    
    # Test active position
    active_position = PositionState(
        position_type=PositionType.LONG,
        entry_price=199.20,
        bars_since_entry=50,
        unrealized_pnl_pips=25.0,
        max_adverse_pips=15.0,
        accumulated_drawdown=10.0,
        bars_since_max_drawdown=20
    )
    
    active_features = extractor.extract_features(active_position, current_price)
    
    # Validate active position
    assert active_features.shape == (9,), f"Expected (9,), got {active_features.shape}"
    assert active_features.dtype == np.float32, f"Expected float32, got {active_features.dtype}"
    assert not np.isnan(active_features).any(), "Position features contain NaN"
    assert not np.isinf(active_features).any(), "Position features contain Inf"
    
    # Validate specific features
    assert active_features[0] == 1.0, f"Long position should have position_side=1.0, got {active_features[0]}"
    assert 0 < active_features[1] < 1, f"Duration should be normalized, got {active_features[1]}"
    
    logger.info(f"✅ Position Features test passed - flat: {features[:3]}, active: {active_features[:3]}")
    return active_features

def test_market_features():
    """Test market feature extraction with WST"""
    logger.info("🧪 Testing Market Features...")
    
    # Create WST config
    wst_config = WSTConfig(J=2, Q=6, backend="manual", output_dim=128)
    
    # Initialize market extractor
    market_extractor = MarketFeatureExtractor(
        wst_config=wst_config,
        window_size=256,
        normalization_method="zscore"
    )
    
    # Add synthetic market data (256 points needed)
    base_time = datetime.now()
    base_price = 199.45
    
    for i in range(256):
        # Generate synthetic OHLC data
        price = base_price + 0.01 * np.sin(i * 0.1) + 0.005 * np.random.randn()
        
        data_point = MarketDataPoint(
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.001,
            high=price + 0.002,
            low=price - 0.002,
            close=price,
            volume=100.0
        )
        
        market_extractor.add_market_data(data_point)
    
    # Extract features
    assert market_extractor.is_ready(), "Market extractor should be ready after 256 data points"
    
    market_features = market_extractor.extract_features(cache_key="test_market")
    
    # Validate
    assert market_features.shape == (128,), f"Expected (128,), got {market_features.shape}"
    assert market_features.dtype == np.float32, f"Expected float32, got {market_features.dtype}"
    assert not np.isnan(market_features).any(), "Market features contain NaN"
    assert not np.isinf(market_features).any(), "Market features contain Inf"
    
    # Test market state
    market_state = market_extractor.get_market_state(cache_key="test_market_state")
    assert market_state.wst_features.shape == (128,), f"Market state WST shape incorrect"
    assert len(market_state.ohlc) == 4, f"OHLC should have 4 values"
    
    logger.info(f"✅ Market Features test passed - shape: {market_features.shape}")
    return market_features, market_state

def test_unified_feature_processor():
    """Test the unified feature processor - the critical component"""
    logger.info("🧪 Testing Unified Feature Processor...")
    
    # Load Episode 13475 compatible configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Force Episode 13475 mode
    config_manager.force_episode_13475_mode()
    
    # Initialize feature processor
    processor = FeatureProcessor(config)
    
    # Add market data
    base_time = datetime.now()
    base_price = 199.45
    
    for i in range(256):
        price = base_price + 0.01 * np.sin(i * 0.1) + 0.005 * np.random.randn()
        
        data_point = MarketDataPoint(
            timestamp=base_time + timedelta(minutes=i),
            open=price - 0.001,
            high=price + 0.002,
            low=price - 0.002,
            close=price,
            volume=100.0
        )
        
        processor.add_market_data(data_point)
    
    # Test with flat position
    flat_position = PositionState()
    current_price = 199.45
    
    assert processor.is_ready(), "Processor should be ready after adding 256 data points"
    
    observation = processor.process_observation(
        position_state=flat_position,
        current_price=current_price,
        market_cache_key="unified_test"
    )
    
    # Validate observation
    assert observation.market_features.shape == (128,), f"Market features wrong shape: {observation.market_features.shape}"
    assert observation.position_features.shape == (9,), f"Position features wrong shape: {observation.position_features.shape}"
    assert observation.combined_features.shape == (137,), f"Combined features wrong shape: {observation.combined_features.shape}"
    
    # Validate observation space
    assert observation.validate(config.observation_space), "Observation failed validation"
    
    # Test active position
    active_position = PositionState(
        position_type=PositionType.LONG,
        entry_price=199.20,
        bars_since_entry=100,
        unrealized_pnl_pips=30.0,
        max_adverse_pips=20.0
    )
    
    active_observation = processor.process_observation(
        position_state=active_position,
        current_price=current_price,
        market_cache_key="unified_test_active"
    )
    
    # Validate active observation
    assert active_observation.position_features[0] == 1.0, "Long position should have position_side=1.0"
    assert active_observation.position_features[2] > 0, "Should have positive PnL"
    
    # Get system status
    status = processor.get_system_status()
    assert status["is_ready"], "System should be ready"
    assert status["processing_stats"]["total_observations_processed"] == 2, "Should have processed 2 observations"
    
    logger.info(f"✅ Unified Feature Processor test passed - combined shape: {observation.combined_features.shape}")
    return observation, active_observation

def test_episode_13475_compatibility():
    """Test Episode 13475 exact parameter compatibility"""
    logger.info("🧪 Testing Episode 13475 Compatibility...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Verify Episode 13475 parameters
    compatibility_check = config_manager.verify_episode_13475_compatibility()
    
    assert compatibility_check["is_compatible"], f"Episode 13475 compatibility failed: {compatibility_check}"
    
    # Test specific parameters
    assert config.trading_config.mcts_config.num_simulations == 15, "MCTS simulations should be 15"
    assert config.trading_config.mcts_config.c_puct == 1.25, "C_PUCT should be 1.25"
    assert config.feature_config.wst_J == 2, "WST J should be 2"
    assert config.feature_config.wst_Q == 6, "WST Q should be 6"
    assert config.observation_space.position_state_dim == 9, "Position features should be 9D"
    
    logger.info("✅ Episode 13475 compatibility verified")
    return compatibility_check

def main():
    """Run all Phase 2 validation tests"""
    logger.info("🚀 Starting Phase 2 Validation Tests...")
    
    try:
        # Test individual components
        wst_features = test_wst_transform()
        position_features = test_position_features()
        market_features, market_state = test_market_features()
        
        # Test unified processor (critical component)
        flat_obs, active_obs = test_unified_feature_processor()
        
        # Test Episode 13475 compatibility
        compatibility = test_episode_13475_compatibility()
        
        # Final validation
        logger.info("🎯 Final Validation Summary:")
        logger.info(f"  ✅ WST Transform: {wst_features.shape} features")
        logger.info(f"  ✅ Position Features: {position_features.shape} features")
        logger.info(f"  ✅ Market Features: {market_features.shape} features")
        logger.info(f"  ✅ Unified Processor: {flat_obs.combined_features.shape} combined features")
        logger.info(f"  ✅ Episode 13475 Compatible: {compatibility['is_compatible']}")
        
        logger.info("🎉 Phase 2 Implementation VALIDATED - All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2 validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)