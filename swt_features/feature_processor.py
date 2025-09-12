"""
Feature Processor - Main Interface
Unified feature processing interface ensuring consistency between training and live systems

This is the CRITICAL component that eliminates feature mismatches by providing
a single source of truth for all feature processing operations.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path

from swt_core.types import MarketState, PositionState, ObservationSpace
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import FeatureProcessingError, DataValidationError

from .wst_transform import WSTProcessor, WSTConfig
from .market_features import MarketFeatureExtractor, MarketDataPoint
from .position_features import PositionFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProcessedObservation:
    """Complete processed observation ready for neural networks"""
    market_features: np.ndarray      # Shape: (128,) - WST features
    position_features: np.ndarray    # Shape: (9,) - Position features  
    combined_features: np.ndarray    # Shape: (137,) - Concatenated features
    metadata: Dict[str, Any]         # Processing metadata
    
    def validate(self, observation_space: ObservationSpace) -> bool:
        """Validate observation against expected space"""
        return observation_space.validate_observation(
            self.market_features, 
            self.position_features
        )


class FeatureProcessor:
    """
    Unified feature processor for SWT trading system
    
    CRITICAL COMPONENT: Provides single source of truth for feature processing
    to eliminate training/live system mismatches that caused trading failures.
    """
    
    def __init__(self, config: SWTConfig):
        """
        Initialize unified feature processor
        
        Args:
            config: SWT system configuration
        """
        self.config = config
        
        # Create WST config - let MarketFeatureExtractor handle backend fallback
        wst_config = WSTConfig(
            J=config.feature_config.wst_config.J,
            Q=config.feature_config.wst_config.Q,
            backend="manual",  # Use manual implementation directly for testing
            output_dim=128,
            normalize=True,
            cache_enabled=False
        )
        
        # Initialize market feature extractor
        self.market_extractor = MarketFeatureExtractor(
            wst_config=wst_config,
            window_size=256,  # From config/features.yaml market_data.price_window_size
            normalization_method="zscore"  # From config/features.yaml market_data.price_normalization
        )
        
        # Initialize position feature extractor with EXACT training parameters
        # Convert NormalizationParams to dictionary format expected by PositionFeatureExtractor
        norm_params = config.feature_config.position_features.normalization_params
        normalization_dict = {
            "duration_scale": norm_params.duration_max_bars,
            "pnl_scale": norm_params.pnl_scale_factor,
            "price_change_scale": norm_params.price_change_scale_factor,
            "drawdown_scale": norm_params.drawdown_scale_factor,
            "accumulated_dd_scale": norm_params.accumulated_dd_scale,
            "bars_since_dd_scale": norm_params.bars_since_dd_scale
        }
        self.position_extractor = PositionFeatureExtractor(
            normalization_config=normalization_dict
        )
        
        # Set up observation space validation
        self.observation_space = ObservationSpace(
            market_state_dim=128,  # WST feature dimension
            position_state_dim=9   # Position feature dimension
        )
        
        # Feature processing statistics
        self._processing_stats = {
            "total_observations_processed": 0,
            "market_feature_errors": 0,
            "position_feature_errors": 0,
            "validation_failures": 0
        }
        
        logger.info("🔧 FeatureProcessor initialized for Episode 13475 checkpoint testing")
    
    def add_market_data(self, data_point: MarketDataPoint) -> None:
        """
        Add market data point to processing pipeline
        
        Args:
            data_point: New market data to process
        """
        try:
            self.market_extractor.add_market_data(data_point)
            
        except Exception as e:
            self._processing_stats["market_feature_errors"] += 1
            raise FeatureProcessingError(
                f"Failed to add market data: {str(e)}",
                context={"timestamp": data_point.timestamp},
                original_error=e
            )
    
    def process_observation(self, position_state: PositionState,
                          current_price: float,
                          market_cache_key: Optional[str] = None,
                          market_metadata: Optional[Dict[str, Any]] = None) -> ProcessedObservation:
        """
        Process complete observation for neural network input
        
        CRITICAL: This method ensures identical processing for training and live systems
        
        Args:
            position_state: Current position state
            current_price: Current market price  
            market_cache_key: Cache key for WST transform
            market_metadata: Additional market information
            
        Returns:
            ProcessedObservation with all features and metadata
        """
        try:
            # Extract market features
            market_features = self._extract_market_features(market_cache_key)
            
            # Extract position features (EXACT training compatibility)
            position_features = self._extract_position_features(
                position_state, current_price, market_metadata
            )
            
            # Combine features
            combined_features = np.concatenate([market_features, position_features])
            
            # Create processed observation
            observation = ProcessedObservation(
                market_features=market_features,
                position_features=position_features,
                combined_features=combined_features,
                metadata={
                    "market_buffer_size": len(self.market_extractor._price_buffer),
                    "current_price": current_price,
                    "position_type": position_state.position_type.name,
                    "processing_timestamp": self._get_current_timestamp(),
                    "feature_dimensions": {
                        "market": market_features.shape,
                        "position": position_features.shape,
                        "combined": combined_features.shape
                    }
                }
            )
            
            # Validate observation
            self._validate_observation(observation)
            
            # Update statistics
            self._processing_stats["total_observations_processed"] += 1
            
            return observation
            
        except Exception as e:
            self._processing_stats["validation_failures"] += 1
            raise FeatureProcessingError(
                f"Observation processing failed: {str(e)}",
                context={
                    "position_type": position_state.position_type.name,
                    "current_price": current_price
                },
                original_error=e
            )
    
    def _extract_market_features(self, cache_key: Optional[str] = None) -> np.ndarray:
        """Extract market features using WST transform"""
        try:
            if not self.market_extractor.is_ready():
                raise FeatureProcessingError(
                    f"Insufficient market data: need {self.market_extractor.window_size}, "
                    f"have {len(self.market_extractor._price_buffer)}"
                )
            
            market_features = self.market_extractor.extract_features(cache_key=cache_key)
            
            # Validate market features
            if market_features.shape != (128,):
                raise FeatureProcessingError(
                    f"Market features shape mismatch: expected (128,), got {market_features.shape}"
                )
            
            return market_features
            
        except Exception as e:
            self._processing_stats["market_feature_errors"] += 1
            raise
    
    def _extract_position_features(self, position_state: PositionState,
                                 current_price: float,
                                 market_metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Extract position features with EXACT training compatibility"""
        try:
            position_features = self.position_extractor.extract_features(
                position_state, current_price, market_metadata
            )
            
            # Validate position features
            if position_features.shape != (9,):
                raise FeatureProcessingError(
                    f"Position features shape mismatch: expected (9,), got {position_features.shape}"
                )
            
            return position_features
            
        except Exception as e:
            self._processing_stats["position_feature_errors"] += 1
            raise
    
    def _validate_observation(self, observation: ProcessedObservation) -> None:
        """Validate processed observation against observation space"""
        # Basic dimension validation
        if not observation.validate(self.observation_space):
            raise DataValidationError(
                "Observation validation failed",
                context={
                    "market_shape": observation.market_features.shape,
                    "position_shape": observation.position_features.shape,
                    "expected_market": (self.observation_space.market_state_dim,),
                    "expected_position": (self.observation_space.position_state_dim,)
                }
            )
        
        # Combined features validation
        expected_combined_dim = self.observation_space.market_state_dim + self.observation_space.position_state_dim
        if observation.combined_features.shape != (expected_combined_dim,):
            raise DataValidationError(
                f"Combined features shape mismatch: expected ({expected_combined_dim},), "
                f"got {observation.combined_features.shape}"
            )
        
        # Value validation
        for feature_array, name in [(observation.market_features, "market"), 
                                   (observation.position_features, "position")]:
            if np.isnan(feature_array).any():
                nan_indices = np.where(np.isnan(feature_array))[0]
                raise DataValidationError(
                    f"{name} features contain NaN at indices: {nan_indices.tolist()}"
                )
            
            if np.isinf(feature_array).any():
                inf_indices = np.where(np.isinf(feature_array))[0]
                raise DataValidationError(
                    f"{name} features contain Inf at indices: {inf_indices.tolist()}"
                )
    
    def is_ready(self) -> bool:
        """Check if processor is ready for feature extraction"""
        return self.market_extractor.is_ready()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        market_status = self.market_extractor.get_buffer_status()
        wst_stats = self.market_extractor.get_wst_cache_stats()
        
        return {
            "is_ready": self.is_ready(),
            "market_extractor": market_status,
            "wst_cache": wst_stats,
            "processing_stats": self._processing_stats.copy(),
            "observation_space": {
                "market_dim": self.observation_space.market_state_dim,
                "position_dim": self.observation_space.position_state_dim,
                "total_dim": self.observation_space.total_dim
            },
            "feature_config": {
                "wst_J": self.config.feature_config.wst_config.J,
                "wst_Q": self.config.feature_config.wst_config.Q,
                "price_window": 256,  # From config/features.yaml market_data.price_window_size
                "normalization": "zscore"  # From config/features.yaml market_data.price_normalization
            }
        }
    
    def reset(self, clear_market_data: bool = False) -> None:
        """
        Reset processor state
        
        Args:
            clear_market_data: Whether to clear market data buffers
        """
        if clear_market_data:
            self.market_extractor.clear_buffers()
            logger.info("🔄 Market data buffers cleared")
        
        # Reset position feature extractor
        self.position_extractor.reset_price_history()
        
        # Reset processing statistics
        self._processing_stats = {
            "total_observations_processed": 0,
            "market_feature_errors": 0,
            "position_feature_errors": 0,
            "validation_failures": 0
        }
        
        logger.info("🔄 FeatureProcessor reset complete")
    
    def get_diagnostics(self, position_state: PositionState,
                       current_price: float) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        diagnostics = {
            "system_status": self.get_system_status(),
            "market_extractor": {
                "buffer_status": self.market_extractor.get_buffer_status(),
                "price_statistics": self.market_extractor.get_price_statistics(),
                "recent_prices": self.market_extractor.get_recent_prices(10)
            }
        }
        
        # Add position diagnostics if possible
        try:
            position_diagnostics = self.position_extractor.get_diagnostics(
                position_state, current_price
            )
            diagnostics["position_extractor"] = position_diagnostics
        except Exception as e:
            diagnostics["position_extractor"] = {"error": str(e)}
        
        return diagnostics
    
    def save_cache(self, cache_dir: Union[str, Path]) -> None:
        """Save WST cache to disk"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        wst_cache_file = cache_path / "wst_cache.pkl"
        self.market_extractor.wst_processor.save_cache(wst_cache_file)
        
        logger.info(f"💾 Feature processor cache saved to {cache_path}")
    
    def load_cache(self, cache_dir: Union[str, Path]) -> None:
        """Load WST cache from disk"""
        cache_path = Path(cache_dir)
        wst_cache_file = cache_path / "wst_cache.pkl"
        
        if wst_cache_file.exists():
            self.market_extractor.wst_processor.load_cache(wst_cache_file)
            logger.info(f"📁 Feature processor cache loaded from {cache_path}")
        else:
            logger.info(f"ℹ️ No cache file found at {wst_cache_file}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_mock_observation(self) -> ProcessedObservation:
        """Create mock observation for testing"""
        market_features = np.random.randn(128).astype(np.float32)
        position_features = np.zeros(9, dtype=np.float32)  # Flat position
        combined_features = np.concatenate([market_features, position_features])
        
        return ProcessedObservation(
            market_features=market_features,
            position_features=position_features,
            combined_features=combined_features,
            metadata={
                "mock_observation": True,
                "timestamp": self._get_current_timestamp()
            }
        )