#!/usr/bin/env python3
"""
SWT Market State Encoder
Combines WST market features with position features for complete state representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import logging
import json

from swt_models.swt_wavelet_scatter import EnhancedWSTCNN

logger = logging.getLogger(__name__)


class SWTMarketStateEncoder(nn.Module):
    """
    Complete market state encoder that combines:
    1. WST-enhanced market features (256 price series → 128 features)  
    2. Position features (9 dimensions)
    3. Final fusion layer (137 → 128 for MuZero compatibility)
    """
    
    def __init__(self, config_path: str = None, config_dict: dict = None):
        super().__init__()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        self.wst_config = self.config['wst']
        self.fusion_config = self.config['fusion']
        
        # WST market feature extractor
        self.market_encoder = EnhancedWSTCNN(
            wst_config=self.wst_config,
            output_dim=self.wst_config['market_output_dim']
        )
        
        # Position feature processing
        position_dim = self.fusion_config['position_features']
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.wst_config.get('dropout_p', 0.1)),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion layer to combine market + position features
        market_dim = self.wst_config['market_output_dim']  # 128
        position_processed_dim = 16
        fusion_input_dim = market_dim + position_processed_dim  # 144
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_config['fusion_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.wst_config.get('dropout_p', 0.1)),
            nn.Linear(self.fusion_config['fusion_hidden_dim'], self.fusion_config['combined_output_dim']),
            nn.ReLU(),
            nn.LayerNorm(self.fusion_config['combined_output_dim'])  # Add layer norm for stability
        )
        
        # Feature dimensions for logging
        self.market_dim = market_dim
        self.position_dim = position_dim
        self.output_dim = self.fusion_config['combined_output_dim']
        
        logger.info(f"🎯 SWT Market State Encoder initialized")
        logger.info(f"   Market: 256 prices → {self.market_dim} features (WST+CNN)")
        logger.info(f"   Position: {self.position_dim} → {position_processed_dim} features")  
        logger.info(f"   Fusion: {fusion_input_dim} → {self.output_dim} final state")
        
    def _get_default_config(self) -> dict:
        """Default configuration if none provided"""
        return {
            'wst': {
                'J': 2,
                'Q': 6,
                'input_length': 256,
                'market_output_dim': 128,
                'dropout_p': 0.1,
                'use_batch_norm': True,
                'use_residual': True
            },
            'fusion': {
                'position_features': 9,
                'combined_output_dim': 128,
                'fusion_hidden_dim': 256
            }
        }
    
    def forward(self, market_data: torch.Tensor, position_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete market state encoder
        
        Args:
            market_data: Price series (B, 1, 256) or (B, 256)
            position_data: Position features (B, 9)
            
        Returns:
            Combined state representation (B, 128)
        """
        # Ensure market data has correct shape
        if market_data.dim() == 2:
            market_data = market_data.unsqueeze(1)  # (B, 256) → (B, 1, 256)
        
        # Extract market features using WST
        market_features = self.market_encoder(market_data)  # (B, 128)
        
        # Process position features
        position_features = self.position_encoder(position_data)  # (B, 16)
        
        # Combine features
        combined_features = torch.cat([market_features, position_features], dim=-1)  # (B, 144)
        
        # Apply fusion layers
        final_state = self.fusion_layers(combined_features)  # (B, 128)
        
        return final_state
    
    def get_feature_breakdown(self, market_data: torch.Tensor, position_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed feature breakdown for analysis
        
        Returns:
            Dictionary with intermediate features for debugging/analysis
        """
        # Ensure correct shapes
        if market_data.dim() == 2:
            market_data = market_data.unsqueeze(1)
        
        # Get intermediate features
        wst_features = self.market_encoder.wst(market_data)
        market_features = self.market_encoder(market_data)
        position_features = self.position_encoder(position_data)
        combined_features = torch.cat([market_features, position_features], dim=-1)
        final_state = self.fusion_layers(combined_features)
        
        return {
            'wst_raw': wst_features,
            'market_processed': market_features,
            'position_processed': position_features,
            'combined': combined_features,
            'final_state': final_state
        }
    
    def analyze_feature_importance(self, market_data: torch.Tensor, position_data: torch.Tensor) -> Dict[str, float]:
        """
        Analyze relative importance of market vs position features
        """
        with torch.no_grad():
            features = self.get_feature_breakdown(market_data, position_data)
            
            market_norm = torch.norm(features['market_processed'], dim=-1).mean().item()
            position_norm = torch.norm(features['position_processed'], dim=-1).mean().item()
            total_norm = market_norm + position_norm
            
            return {
                'market_importance': market_norm / total_norm if total_norm > 0 else 0.5,
                'position_importance': position_norm / total_norm if total_norm > 0 else 0.5,
                'market_magnitude': market_norm,
                'position_magnitude': position_norm
            }


class SWTPositionFeatureExtractor:
    """
    Utility class to extract position features for the SWT system
    Maintains compatibility with existing position tracking
    """
    
    @staticmethod
    def extract_position_features(position_info: Dict[str, Any]) -> np.ndarray:
        """
        Extract 9D position feature vector - EXACT V7/V8 compatibility
        
        Features (matching V7/V8 exactly):
        1. current_equity_pips - Current position P&L in pips (arctan scaled)
        2. bars_since_entry - Time in current position (arctan scaled)
        3. position_efficiency - Performance efficiency metric [0, 1]
        4. pips_from_peak - Drawdown from best position point (arctan scaled)
        5. max_drawdown_pips - Maximum drawdown in pips (arctan scaled)
        6. amddp1_reward - Actual AMDDP1 reward value (arctan scaled)
        7. is_long - Long position flag (binary)
        8. is_short - Short position flag (binary)  
        9. has_position - Position active flag (binary)
        """
        features = np.zeros(9, dtype=np.float32)
        
        # V7/V8 arctan scaling function for proper normalization
        def arctan_scale(value: float, scale_factor: float = 150.0) -> float:
            """Arctan scaling to [-1, 1] range"""
            return np.tanh(value / scale_factor)
        
        # Features 0-5: Position metrics (scaled to [-1, 1])
        # Note: arctan scaling handles values > 150 pips smoothly
        features[0] = arctan_scale(position_info.get('current_equity_pips', 0.0), 150)
        features[1] = arctan_scale(position_info.get('bars_since_entry', 0), 2000)
        features[2] = position_info.get('position_efficiency', 0.0)  # Already in [0, 1]
        features[3] = arctan_scale(position_info.get('pips_from_peak', 0.0), 150)
        features[4] = arctan_scale(position_info.get('max_drawdown_pips', 0.0), 150)
        features[5] = arctan_scale(position_info.get('amddp1_reward', 0.0), 150)  # AMDDP1 for SWT
        
        # Features 6-8: Binary flags (exact V7/V8 logic)
        direction = position_info.get('direction', 0)
        features[6] = float(direction > 0)   # is_long
        features[7] = float(direction < 0)   # is_short  
        features[8] = float(direction != 0)  # has_position
        
        return features


def test_market_encoder():
    """Test function for market encoder"""
    
    logger.info("🧪 Testing SWT Market State Encoder")
    
    # Create test data
    batch_size = 4
    market_data = torch.randn(batch_size, 1, 256)  # Price series
    position_data = torch.randn(batch_size, 9)  # Position features
    
    # Test with default config
    encoder = SWTMarketStateEncoder()
    
    # Forward pass
    final_state = encoder(market_data, position_data)
    logger.info(f"   Final state shape: {final_state.shape}")
    
    # Test feature breakdown
    features = encoder.get_feature_breakdown(market_data, position_data)
    for name, tensor in features.items():
        logger.info(f"   {name}: {tensor.shape}")
    
    # Test feature importance analysis
    importance = encoder.analyze_feature_importance(market_data, position_data)
    logger.info(f"   Feature importance: {importance}")
    
    # Verify output shape
    assert final_state.shape == (batch_size, 128), f"Output shape mismatch: {final_state.shape}"
    
    logger.info("✅ All market encoder tests passed!")
    
    return encoder


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_market_encoder()