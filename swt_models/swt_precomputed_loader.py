#!/usr/bin/env python3
"""
Precomputed WST Features Loader
Loads precomputed WST features from HDF5 files for training
"""

import h5py
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from threading import Lock

logger = logging.getLogger(__name__)


class PrecomputedWSTLoader:
    """
    Loads precomputed WST features from HDF5 files
    Thread-safe and memory-efficient with caching
    """
    
    def __init__(self, hdf5_path: str, cache_size: int = 10000):
        """
        Initialize precomputed WST loader
        
        Args:
            hdf5_path: Path to HDF5 file with precomputed WST features
            cache_size: Number of features to keep in memory cache
        """
        self.hdf5_path = Path(hdf5_path)
        self.cache_size = cache_size
        self.cache = {}
        self.lock = Lock()
        
        # Verify file exists
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Precomputed WST file not found: {hdf5_path}")
        
        # Load metadata
        self._load_metadata()
        
        logger.info(f"🗃️ Precomputed WST Loader initialized")
        logger.info(f"   File: {self.hdf5_path}")
        logger.info(f"   Windows: {self.n_windows:,}")
        logger.info(f"   WST dimension: {self.wst_dim}")
        logger.info(f"   Cache size: {cache_size}")
    
    def _load_metadata(self):
        """Load metadata from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_windows = f['wst_features'].shape[0]
            self.wst_dim = f['wst_features'].shape[1]
            
            # Store metadata
            self.metadata = {
                'window_size': f.attrs.get('window_size', 256),
                'stride': f.attrs.get('stride', 1),
                'wst_J': f.attrs.get('wst_J', 2),
                'wst_Q': f.attrs.get('wst_Q', 6),
                'data_path': f.attrs.get('data_path', '').decode() if isinstance(f.attrs.get('data_path', ''), bytes) else f.attrs.get('data_path', ''),
                'creation_time': f.attrs.get('creation_time', '').decode() if isinstance(f.attrs.get('creation_time', ''), bytes) else f.attrs.get('creation_time', ''),
                'method': f.attrs.get('method', '').decode() if isinstance(f.attrs.get('method', ''), bytes) else f.attrs.get('method', '')
            }
    
    def get_wst_features(self, start_idx: int, end_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get WST features for given range
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive). If None, returns single feature at start_idx
            
        Returns:
            WST features tensor of shape (n_samples, wst_dim) or (wst_dim,) for single sample
        """
        if end_idx is None:
            # Single feature
            return self.get_single_wst_feature(start_idx)
        
        # Range of features
        start_idx = max(0, min(start_idx, self.n_windows - 1))
        end_idx = max(start_idx + 1, min(end_idx, self.n_windows))
        
        # Check cache first
        cache_key = (start_idx, end_idx)
        with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key].clone()
        
        # Load from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            features = f['wst_features'][start_idx:end_idx]
        
        # Convert to tensor
        tensor_features = torch.from_numpy(features.astype(np.float32))
        
        # Update cache
        with self.lock:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = tensor_features.clone()
        
        return tensor_features
    
    def get_single_wst_feature(self, idx: int) -> torch.Tensor:
        """
        Get single WST feature vector
        
        Args:
            idx: Index of feature to retrieve
            
        Returns:
            WST feature tensor of shape (wst_dim,)
        """
        idx = max(0, min(idx, self.n_windows - 1))
        
        # Check cache first
        cache_key = (idx, idx + 1)
        with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key][0].clone()
        
        # Load from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            feature = f['wst_features'][idx]
        
        # Convert to tensor
        tensor_feature = torch.from_numpy(feature.astype(np.float32))
        
        # Update cache
        with self.lock:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = tensor_feature.unsqueeze(0)
        
        return tensor_feature
    
    def get_batch_wst_features(self, indices: list) -> torch.Tensor:
        """
        Get batch of WST features for given indices
        
        Args:
            indices: List of indices to retrieve
            
        Returns:
            WST features tensor of shape (batch_size, wst_dim)
        """
        batch_features = []
        
        for idx in indices:
            feature = self.get_single_wst_feature(idx)
            batch_features.append(feature)
        
        return torch.stack(batch_features)
    
    def get_window_range_for_data_index(self, data_idx: int) -> Tuple[int, int]:
        """
        Get window range that covers a specific data index
        Useful for mapping CSV row indices to WST window indices
        
        Args:
            data_idx: Index in original CSV data
            
        Returns:
            Tuple of (start_window_idx, end_window_idx) covering the data index
        """
        window_size = self.metadata['window_size']
        stride = self.metadata['stride']
        
        # Calculate which windows contain this data index
        # Window i covers data indices [i * stride, i * stride + window_size)
        start_window = max(0, (data_idx - window_size + 1) // stride)
        end_window = min(self.n_windows, (data_idx // stride) + 1)
        
        return start_window, end_window
    
    def clear_cache(self):
        """Clear feature cache"""
        with self.lock:
            self.cache.clear()
        logger.info("🧹 WST feature cache cleared")
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information"""
        return {
            'hdf5_path': str(self.hdf5_path),
            'n_windows': self.n_windows,
            'wst_dim': self.wst_dim,
            'cache_size': self.cache_size,
            'cached_items': len(self.cache),
            'metadata': self.metadata
        }
    
    def __len__(self) -> int:
        """Number of available WST features"""
        return self.n_windows