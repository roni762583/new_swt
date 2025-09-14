#!/usr/bin/env python3
"""
Precomputed WST Data Loader for SWT Training
Loads precomputed WST features from HDF5 files for fast training
"""

import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PrecomputedWSTDataLoader:
    """Data loader for precomputed WST features"""
    
    def __init__(
        self, 
        precomputed_path: str,
        fallback_csv_path: str = None,
        use_precomputed: bool = True
    ):
        """
        Initialize data loader
        
        Args:
            precomputed_path: Path to precomputed HDF5 file
            fallback_csv_path: Fallback CSV if precomputed not available
            use_precomputed: Whether to use precomputed features
        """
        self.precomputed_path = Path(precomputed_path)
        self.fallback_csv_path = fallback_csv_path
        self.use_precomputed = use_precomputed and self.precomputed_path.exists()
        
        if self.use_precomputed:
            self._load_precomputed()
            logger.info(f"🚀 Using precomputed WST features from {precomputed_path}")
        else:
            self._load_fallback()
            logger.info(f"⚠️ Using fallback CSV data from {fallback_csv_path}")
    
    def _load_precomputed(self):
        """Load precomputed WST features from HDF5"""
        with h5py.File(self.precomputed_path, 'r') as f:
            # Load WST features (raw WST output, not CNN-processed)
            self.wst_features = f['wst_features'][:]
            self.timestamps = f['timestamps'][:]
            self.indices = f['indices'][:]
            self.price_windows = f['price_windows'][:]
            
            # Metadata
            self.window_size = f.attrs['window_size']
            self.wst_output_dim = f.attrs['wst_output_dim']
            self.n_windows = f.attrs['n_windows']
            
        logger.info(f"📚 Loaded precomputed data: {self.n_windows:,} windows")
        logger.info(f"   WST features: {self.wst_features.shape}")
        logger.info(f"   Price windows: {self.price_windows.shape}")
    
    def _load_fallback(self):
        """Load CSV data as fallback"""
        if not self.fallback_csv_path or not Path(self.fallback_csv_path).exists():
            raise ValueError("Fallback CSV path not available")
        
        df = pd.read_csv(self.fallback_csv_path)
        self.price_data = df[['open', 'high', 'low', 'close']].values
        self.n_windows = len(self.price_data) - 256 + 1
        self.window_size = 256
        
        logger.info(f"📊 Loaded fallback CSV: {len(self.price_data):,} bars")
    
    def get_wst_features_at_index(self, idx: int) -> torch.Tensor:
        """Get WST features at specific index"""
        if self.use_precomputed:
            features = self.wst_features[idx].astype(np.float32)
            return torch.from_numpy(features)
        else:
            # Would need to compute WST on-the-fly here
            raise NotImplementedError("On-the-fly WST computation not implemented")
    
    def get_price_window_at_index(self, idx: int) -> torch.Tensor:
        """Get price window at specific index"""
        if self.use_precomputed:
            window = self.price_windows[idx].astype(np.float32)
            return torch.from_numpy(window)
        else:
            window = self.price_data[idx:idx + self.window_size]
            return torch.from_numpy(window.astype(np.float32))
    
    def get_batch(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of WST features and price windows
        
        Args:
            indices: List of window indices
            
        Returns:
            Tuple of (wst_features, price_windows)
        """
        if self.use_precomputed:
            wst_batch = torch.stack([self.get_wst_features_at_index(i) for i in indices])
            price_batch = torch.stack([self.get_price_window_at_index(i) for i in indices])
        else:
            # Fallback batch loading
            price_batch = []
            for i in indices:
                window = self.price_data[i:i + self.window_size]
                price_batch.append(torch.from_numpy(window.astype(np.float32)))
            price_batch = torch.stack(price_batch)
            wst_batch = None  # Would need WST computation
        
        return wst_batch, price_batch
    
    def __len__(self) -> int:
        """Get number of available windows"""
        return self.n_windows
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information"""
        info = {
            'use_precomputed': self.use_precomputed,
            'n_windows': self.n_windows,
            'window_size': self.window_size,
        }
        
        if self.use_precomputed:
            info.update({
                'wst_output_dim': self.wst_output_dim,
                'precomputed_path': str(self.precomputed_path)
            })
        else:
            info.update({
                'fallback_path': self.fallback_csv_path
            })
        
        return info


class PrecomputedSWTMarketEncoder(torch.nn.Module):
    """
    SWT Market Encoder that uses precomputed WST features
    Bypasses WST computation by loading precomputed features
    """
    
    def __init__(
        self, 
        precomputed_path: str,
        fallback_csv_path: str = None,
        config_dict: dict = None
    ):
        super().__init__()
        
        # Configuration
        if config_dict is None:
            config_dict = {
                'wst': {
                    'market_output_dim': 128,
                    'dropout_p': 0.1
                },
                'fusion': {
                    'position_features': 9
                }
            }
        
        self.wst_config = config_dict['wst']
        self.fusion_config = config_dict['fusion']
        
        # Precomputed data loader
        self.wst_loader = PrecomputedWSTDataLoader(
            precomputed_path=precomputed_path,
            fallback_csv_path=fallback_csv_path
        )
        
        # CNN layers to process WST features (similar to EnhancedWSTCNN)
        if self.wst_loader.use_precomputed:
            wst_input_dim = self.wst_loader.wst_output_dim  # Raw WST dimension
        else:
            wst_input_dim = 784  # Fallback dimension
        
        # CNN processing of WST features
        self.wst_cnn = torch.nn.Sequential(
            torch.nn.Linear(wst_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.wst_config['dropout_p']),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.wst_config['dropout_p']),
            torch.nn.Linear(128, self.wst_config['market_output_dim'])
        )
        
        # Position encoder (identity - no processing)
        self.position_encoder = torch.nn.Identity()
        
        # Output dimensions
        market_dim = self.wst_config['market_output_dim']  # 128
        position_dim = self.fusion_config['position_features']  # 9
        self.output_dim = market_dim + position_dim  # 137
        
        logger.info(f"🎯 Precomputed SWT Market Encoder initialized")
        logger.info(f"   WST precomputed: {self.wst_loader.use_precomputed}")
        logger.info(f"   Market features: {market_dim}")
        logger.info(f"   Position features: {position_dim}")
        logger.info(f"   Total output: {self.output_dim}")
    
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
        
        # Get WST features from precomputed data
        if self.wst_loader.use_precomputed:
            wst_features = []
            for i in range(batch_size):
                idx = market_data_index[i].item()
                wst_feat = self.wst_loader.get_wst_features_at_index(idx)
                wst_features.append(wst_feat)
            wst_features = torch.stack(wst_features)
        else:
            # Fallback - would need to implement WST computation
            raise NotImplementedError("Fallback WST computation not implemented")
        
        # Process WST features through CNN
        market_features = self.wst_cnn(wst_features)  # (B, 128)
        
        # Position features pass through directly
        position_features = self.position_encoder(position_data)  # (B, 9)
        
        # Direct concatenation
        combined_features = torch.cat([market_features, position_features], dim=-1)  # (B, 137)
        
        return combined_features
    
    def get_loader_info(self) -> Dict[str, Any]:
        """Get information about the data loader"""
        return self.wst_loader.get_info()


def test_precomputed_loader():
    """Test function for precomputed data loader"""
    
    # Test with mock precomputed file
    precomputed_path = "precomputed_wst/GBPJPY_WST_3.5years.h5"
    fallback_path = "data/GBPJPY_M1_3.5years_20250912.csv"
    
    try:
        loader = PrecomputedWSTDataLoader(
            precomputed_path=precomputed_path,
            fallback_csv_path=fallback_path
        )
        
        logger.info(f"✅ Loader initialized successfully")
        logger.info(f"   Info: {loader.get_info()}")
        
        # Test getting single features
        if loader.use_precomputed and len(loader) > 1000:
            wst_features = loader.get_wst_features_at_index(1000)
            price_window = loader.get_price_window_at_index(1000)
            
            logger.info(f"   Sample WST features shape: {wst_features.shape}")
            logger.info(f"   Sample price window shape: {price_window.shape}")
        
        # Test batch loading
        indices = [100, 200, 300, 400, 500]
        if loader.use_precomputed:
            wst_batch, price_batch = loader.get_batch(indices)
            logger.info(f"   Batch WST shape: {wst_batch.shape}")
            logger.info(f"   Batch price shape: {price_batch.shape}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    logger.info("✅ Precomputed loader test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_precomputed_loader()