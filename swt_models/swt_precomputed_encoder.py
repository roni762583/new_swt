#!/usr/bin/env python3
"""
SWT Precomputed Market State Encoder
Uses precomputed WST features for faster training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
import json
from pathlib import Path

from precomputed_wst_dataloader import PrecomputedWSTDataLoader

logger = logging.getLogger(__name__)


class SWTPrecomputedMarketStateEncoder(nn.Module):
    """
    Market state encoder using precomputed WST features:
    1. Precomputed WST features (16 dimensions) → CNN → 128 features
    2. Position features (9 dimensions) → direct passthrough
    3. Direct concatenation → 137 features for MuZero
    """
    
    def __init__(
        self, 
        precomputed_path: str = "precomputed_wst/GBPJPY_WST_CLEAN_2022-2025.h5",
        fallback_csv_path: str = "data/GBPJPY_M1_REAL_2022-2025.csv",
        config_dict: dict = None
    ):
        super().__init__()
        
        # Configuration
        if config_dict is None:
            config_dict = self._get_default_config()
            
        self.wst_config = config_dict['wst']
        self.fusion_config = config_dict['fusion']
        
        # Precomputed WST data loader
        self.wst_loader = PrecomputedWSTDataLoader(
            precomputed_path=precomputed_path,
            fallback_csv_path=fallback_csv_path
        )
        
        # CNN layers to process precomputed WST features
        if self.wst_loader.use_precomputed:
            wst_input_dim = self.wst_loader.wst_output_dim  # Raw WST dimension (16)
            logger.info(f"Using precomputed WST features: {wst_input_dim} dimensions")
        else:
            wst_input_dim = 784  # Fallback dimension
            logger.warning(f"Using fallback mode: {wst_input_dim} dimensions")
        
        # CNN processing of precomputed WST features (16 → 128)
        self.wst_processor = nn.Sequential(
            nn.Linear(wst_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.wst_config.get('dropout_p', 0.1)),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(self.wst_config.get('dropout_p', 0.1)),
            nn.Linear(128, self.wst_config['market_output_dim'])  # 128
        )
        
        # Position encoder (identity - no processing)
        self.position_encoder = nn.Identity()
        
        # Output dimensions
        market_dim = self.wst_config['market_output_dim']  # 128
        position_dim = self.fusion_config['position_features']  # 9
        self.output_dim = market_dim + position_dim  # 137
        
        # Feature dimensions for logging
        self.market_dim = market_dim
        self.position_dim = position_dim
        
        logger.info(f"🚀 SWT Precomputed Market State Encoder initialized")
        logger.info(f"   WST precomputed: {self.wst_loader.use_precomputed}")
        logger.info(f"   WST input: {wst_input_dim} → Market output: {self.market_dim}")
        logger.info(f"   Position: {self.position_dim} features (direct passthrough)")
        logger.info(f"   Total output: {self.output_dim} features (137)")
    
    def _get_default_config(self) -> dict:
        """Default configuration"""
        return {
            'wst': {
                'market_output_dim': 128,
                'dropout_p': 0.1
            },
            'fusion': {
                'position_features': 9
            }
        }
    
    def forward(
        self, 
        market_data_index: torch.Tensor,  # Window indices, not raw prices
        position_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass using precomputed WST features
        
        Args:
            market_data_index: Tensor of window indices (B,) 
            position_data: Position features (B, 9)
            
        Returns:
            Combined state representation (B, 137)
        """
        batch_size = market_data_index.shape[0]
        
        # Get precomputed WST features
        if self.wst_loader.use_precomputed:
            wst_features = []
            for i in range(batch_size):
                idx = market_data_index[i].item()
                wst_feat = self.wst_loader.get_wst_features_at_index(idx)
                wst_features.append(wst_feat)
            wst_features = torch.stack(wst_features)  # (B, 16)
        else:
            # Fallback - would need raw price processing
            raise NotImplementedError("Fallback mode not implemented for precomputed encoder")
        
        # Process WST features through CNN
        market_features = self.wst_processor(wst_features)  # (B, 128)
        
        # Position features pass through directly
        position_features = self.position_encoder(position_data)  # (B, 9)
        
        # Direct concatenation
        combined_features = torch.cat([market_features, position_features], dim=-1)  # (B, 137)
        
        return combined_features
    
    def get_loader_info(self) -> Dict[str, Any]:
        """Get information about the data loader"""
        return self.wst_loader.get_info()
    
    def analyze_feature_importance(self, market_data_index: torch.Tensor, position_data: torch.Tensor) -> Dict[str, float]:
        """
        Analyze relative importance of market vs position features
        """
        with torch.no_grad():
            batch_size = market_data_index.shape[0]
            
            # Get WST features
            wst_features = []
            for i in range(batch_size):
                idx = market_data_index[i].item()
                wst_feat = self.wst_loader.get_wst_features_at_index(idx)
                wst_features.append(wst_feat)
            wst_features = torch.stack(wst_features)
            
            market_features = self.wst_processor(wst_features)
            position_features = self.position_encoder(position_data)
            
            market_norm = torch.norm(market_features, dim=-1).mean().item()
            position_norm = torch.norm(position_features, dim=-1).mean().item()
            total_norm = market_norm + position_norm
            
            return {
                'market_importance': market_norm / total_norm if total_norm > 0 else 0.5,
                'position_importance': position_norm / total_norm if total_norm > 0 else 0.5,
                'market_magnitude': market_norm,
                'position_magnitude': position_norm
            }


def test_precomputed_encoder():
    """Test function for precomputed market encoder"""
    
    logger.info("🧪 Testing SWT Precomputed Market State Encoder")
    
    # Test data - using indices instead of raw prices
    batch_size = 4
    market_indices = torch.randint(1000, 2000, (batch_size,))  # Window indices
    position_data = torch.randn(batch_size, 9)  # Position features
    
    # Test encoder
    try:
        encoder = SWTPrecomputedMarketStateEncoder()
        
        # Forward pass
        final_state = encoder(market_indices, position_data)
        logger.info(f"   Final state shape: {final_state.shape}")
        
        # Test feature importance analysis
        importance = encoder.analyze_feature_importance(market_indices, position_data)
        logger.info(f"   Feature importance: {importance}")
        
        # Verify output shape
        assert final_state.shape == (batch_size, 137), f"Output shape mismatch: {final_state.shape}"
        
        logger.info("✅ All precomputed encoder tests passed!")
        
        return encoder
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_precomputed_encoder()