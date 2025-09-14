#!/usr/bin/env python3
"""
Memory-Efficient WST Precomputer
Processes entire dataset one row at a time with Numba JIT optimization
Streams directly to HDF5 without loading full dataset into memory
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import h5py
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional, Iterator
import gc
import psutil
from numba import njit, types
from numba.typed import Dict as NumbaDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "experimental_research"))

from swt_models.swt_wavelet_scatter import EnhancedWSTCNN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@njit(cache=True)
def numba_wst_single_window(
    price_window: np.ndarray,
    J: int = 2,
    Q: int = 6
) -> np.ndarray:
    """
    Numba JIT-compiled WST computation for single price window
    Memory-optimized version that processes one window at a time
    
    Args:
        price_window: Single price window (256,)
        J: Number of octaves
        Q: Number of wavelets per octave
        
    Returns:
        WST coefficients (16,) for memory efficiency
    """
    N = len(price_window)
    
    # Pre-allocate arrays with exact sizes
    coefficients = np.zeros(16, dtype=np.float32)  # Fixed size output
    
    # Morlet wavelet parameters
    sigma = 0.8
    
    # First order scattering
    coeff_idx = 0
    
    for j1 in range(J):
        scale1 = 2 ** j1
        
        for q1 in range(Q):
            if coeff_idx >= 16:
                break
                
            # Frequency parameter
            xi1 = 2.0 * np.pi * (q1 + 1) / (Q + 1) / scale1
            
            # Apply wavelet transform (simplified for memory efficiency)
            wavelet_response = 0.0
            for t in range(N):
                # Simplified Morlet wavelet
                gauss = np.exp(-0.5 * ((t - N//2) / (sigma * scale1)) ** 2)
                complex_exp = np.cos(xi1 * (t - N//2)) 
                wavelet = gauss * complex_exp / np.sqrt(scale1)
                
                # Convolution (simplified)
                if t < N:
                    wavelet_response += price_window[t] * wavelet
            
            # Store absolute value (real-valued output)
            coefficients[coeff_idx] = abs(wavelet_response)
            coeff_idx += 1
            
            if coeff_idx >= 16:
                break
    
    return coefficients


class MemoryEfficientWSTPrecomputer:
    """
    Memory-efficient WST precomputer that processes one row at a time
    Streams results directly to HDF5 without memory accumulation
    """
    
    def __init__(
        self,
        data_path: str,
        output_path: str,
        window_size: int = 256,
        stride: int = 1,
        chunk_size: int = 10000,  # HDF5 chunk size for streaming writes
        max_memory_mb: int = 4000  # Maximum memory usage limit
    ):
        """
        Initialize memory-efficient WST precomputer
        
        Args:
            data_path: Path to CSV with OHLC data
            output_path: Path to save HDF5 with precomputed features
            window_size: Size of price window (256 bars)
            stride: Step size between windows (1 = every minute)
            chunk_size: HDF5 chunk size for streaming writes
            max_memory_mb: Maximum memory usage in MB
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        
        # WST parameters
        self.J = 2
        self.Q = 6
        self.wst_output_dim = 16  # Fixed for memory efficiency
        
        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory_mb = 0
        
        logger.info(f"🚀 Memory-Efficient WST Precomputer initialized")
        logger.info(f"   Window size: {window_size}")
        logger.info(f"   Stride: {stride}")
        logger.info(f"   WST output dimension: {self.wst_output_dim}")
        logger.info(f"   HDF5 chunk size: {chunk_size}")
        logger.info(f"   Memory limit: {max_memory_mb} MB")
    
    def _monitor_memory(self):
        """Monitor and log memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        
        if memory_mb > self.max_memory_mb * 0.9:  # 90% warning threshold
            logger.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB (limit: {self.max_memory_mb} MB)")
            gc.collect()  # Force garbage collection
        
        return memory_mb
    
    def _get_total_windows(self) -> int:
        """Get total number of windows without loading full dataset"""
        # Read just the first few rows to get row count efficiently
        with open(self.data_path, 'r') as f:
            # Count lines (excluding header)
            line_count = sum(1 for _ in f) - 1
        
        n_windows = max(0, (line_count - self.window_size) // self.stride + 1)
        logger.info(f"📊 Dataset: {line_count:,} bars → {n_windows:,} windows")
        return n_windows
    
    def _stream_price_windows(self) -> Iterator[Tuple[int, np.ndarray, str, int]]:
        """
        Stream price windows one at a time from CSV
        Generator yields: (window_idx, price_window, timestamp, original_idx)
        """
        logger.info(f"📂 Streaming data from {self.data_path}")
        
        # Use pandas iterator to read CSV in chunks
        # Use larger chunks for reading CSV to ensure we get all data
        csv_chunk_size = 50000  # Larger chunks for reading
        chunk_iter = pd.read_csv(
            self.data_path, 
            chunksize=csv_chunk_size,
            dtype={'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32}
        )
        
        # Buffer to maintain sliding window and timestamps
        price_buffer = []
        timestamp_buffer = []
        data_row_idx = 0  # Track actual data row index
        window_idx = 0
        
        for chunk in chunk_iter:
            # Handle timestamp column
            time_cols = ['time', 'timestamp', 'datetime', 'date']
            time_col = None
            for col in time_cols:
                if col in chunk.columns:
                    time_col = col
                    break
            
            if time_col:
                chunk['timestamp'] = pd.to_datetime(chunk[time_col])
            else:
                # Create synthetic timestamps based on current position
                chunk['timestamp'] = pd.date_range(
                    start='2022-01-01', 
                    periods=len(chunk), 
                    freq='1min'
                )
            
            # Extract close prices and timestamps
            close_prices = chunk['close'].values.astype(np.float32)
            timestamps = chunk['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
            
            # Add to buffers
            price_buffer.extend(close_prices)
            timestamp_buffer.extend(timestamps)
            
            # Generate windows from buffer
            while len(price_buffer) >= self.window_size:
                # Extract window
                price_window = np.array(price_buffer[:self.window_size], dtype=np.float32)
                # Use timestamp from the end of the window
                window_timestamp = timestamp_buffer[self.window_size - 1]

                yield window_idx, price_window, window_timestamp, data_row_idx + self.window_size - 1

                # Slide window by stride
                window_idx += 1
                # Remove stride elements from the beginning
                for _ in range(self.stride):
                    if price_buffer:
                        price_buffer.pop(0)
                        timestamp_buffer.pop(0)
                data_row_idx += self.stride  # Increment data_row_idx once by stride amount

                # Memory monitoring periodically
                if window_idx % 1000 == 0:
                    self._monitor_memory()
    
    def precompute_streaming(self):
        """
        Main streaming precomputation with direct HDF5 writes
        Processes one window at a time to minimize memory usage
        """
        start_time = time.time()
        
        # Get total windows for progress tracking
        total_windows = self._get_total_windows()
        
        # Create output directory
        self.output_path.parent.mkdir(exist_ok=True)
        
        logger.info(f"💾 Creating HDF5 file: {self.output_path}")
        
        # Open HDF5 file for streaming writes
        with h5py.File(self.output_path, 'w') as f:
            # Create datasets with exact sizes and chunking for streaming
            wst_dataset = f.create_dataset(
                'wst_features',
                shape=(total_windows, self.wst_output_dim),
                dtype=np.float32,
                chunks=(self.chunk_size, self.wst_output_dim),
                compression='gzip',
                compression_opts=1  # Fast compression
            )
            
            timestamp_dataset = f.create_dataset(
                'timestamps',
                shape=(total_windows,),
                dtype=h5py.string_dtype(),
                chunks=(self.chunk_size,),
                compression='gzip',
                compression_opts=1
            )
            
            indices_dataset = f.create_dataset(
                'indices',
                shape=(total_windows,),
                dtype=np.int64,
                chunks=(self.chunk_size,),
                compression='gzip',
                compression_opts=1
            )
            
            # Store metadata
            f.attrs['window_size'] = self.window_size
            f.attrs['stride'] = self.stride
            f.attrs['wst_output_dim'] = self.wst_output_dim
            f.attrs['wst_J'] = self.J
            f.attrs['wst_Q'] = self.Q
            f.attrs['data_path'] = str(self.data_path)
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['method'] = 'streaming_numba_jit'
            
            logger.info(f"🔄 Starting streaming WST computation...")
            
            # Process windows one at a time
            windows_processed = 0
            batch_features = []
            batch_timestamps = []
            batch_indices = []
            
            with tqdm(total=total_windows, desc="Computing WST") as pbar:
                for window_idx, price_window, timestamp, original_idx in self._stream_price_windows():
                    
                    # Compute WST for single window using Numba JIT
                    wst_features = numba_wst_single_window(price_window, self.J, self.Q)
                    
                    # Add to batch
                    batch_features.append(wst_features)
                    batch_timestamps.append(timestamp)
                    batch_indices.append(original_idx)
                    
                    # Write batch when it reaches chunk size
                    if len(batch_features) >= self.chunk_size:
                        self._write_batch_to_hdf5(
                            f, windows_processed, 
                            batch_features, batch_timestamps, batch_indices
                        )
                        
                        # Clear batch and update progress
                        windows_processed += len(batch_features)
                        pbar.update(len(batch_features))
                        
                        batch_features = []
                        batch_timestamps = []
                        batch_indices = []
                        
                        # Force memory cleanup
                        gc.collect()
                        self._monitor_memory()
                    
                    # Check if we've reached the expected total (with small margin for rounding)
                    if window_idx >= total_windows - 1:
                        break
                
                # Write remaining batch
                if batch_features:
                    self._write_batch_to_hdf5(
                        f, windows_processed,
                        batch_features, batch_timestamps, batch_indices
                    )
                    windows_processed += len(batch_features)
                    pbar.update(len(batch_features))
            
            # Update final metadata
            f.attrs['n_windows'] = windows_processed
        
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        
        logger.info(f"✅ Streaming precomputation complete!")
        logger.info(f"   Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"   Windows processed: {windows_processed:,}")
        logger.info(f"   Output file size: {file_size_mb:.1f} MB")
        logger.info(f"   Processing speed: {windows_processed/elapsed:.1f} windows/sec")
        logger.info(f"   Peak memory usage: {self.peak_memory_mb:.1f} MB")
        
        return windows_processed
    
    def _write_batch_to_hdf5(
        self, 
        f: h5py.File, 
        start_idx: int, 
        features: list, 
        timestamps: list, 
        indices: list
    ):
        """Write batch of features to HDF5 datasets"""
        batch_size = len(features)
        end_idx = start_idx + batch_size
        
        # Write WST features
        f['wst_features'][start_idx:end_idx] = np.array(features, dtype=np.float32)
        
        # Write timestamps
        f['timestamps'][start_idx:end_idx] = np.array(timestamps, dtype='S19')
        
        # Write indices  
        f['indices'][start_idx:end_idx] = np.array(indices, dtype=np.int64)
        
        # Flush to disk
        f.flush()
    
    def verify_output(self):
        """Verify the saved precomputed data"""
        logger.info(f"🔍 Verifying output file...")
        
        with h5py.File(self.output_path, 'r') as f:
            # Check datasets
            logger.info(f"   Datasets: {list(f.keys())}")
            logger.info(f"   WST features shape: {f['wst_features'].shape}")
            logger.info(f"   Timestamps shape: {f['timestamps'].shape}")
            logger.info(f"   Indices shape: {f['indices'].shape}")
            
            # Check metadata
            logger.info(f"   Metadata:")
            for key, value in f.attrs.items():
                logger.info(f"     {key}: {value}")
            
            # Sample features
            if f['wst_features'].shape[0] > 0:
                sample_features = f['wst_features'][:min(5, f['wst_features'].shape[0])]
                logger.info(f"   Sample features (first {len(sample_features)}):") 
                logger.info(f"     Shape: {sample_features.shape}")
                logger.info(f"     Mean: {sample_features.mean():.6f}")
                logger.info(f"     Std: {sample_features.std():.6f}")
                logger.info(f"     Min: {sample_features.min():.6f}")
                logger.info(f"     Max: {sample_features.max():.6f}")


def main():
    """Main execution with memory-efficient streaming"""
    
    # Configuration
    data_path = "data/GBPJPY_M1_3.5years_20250912.csv"
    output_dir = Path("precomputed_wst")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "GBPJPY_WST_3.5years_streaming.h5"
    
    # Check if data exists
    if not Path(data_path).exists():
        data_path = "data/GBPJPY_M1_3years_20250912.csv"
        if not Path(data_path).exists():
            logger.error(f"❌ Data file not found: {data_path}")
            return 1
    
    # Initialize memory-efficient precomputer
    precomputer = MemoryEfficientWSTPrecomputer(
        data_path=data_path,
        output_path=str(output_path),
        window_size=256,
        stride=1,
        chunk_size=5000,  # Smaller chunks for memory efficiency
        max_memory_mb=6000  # 6GB memory limit
    )
    
    # Run streaming precomputation
    logger.info("=" * 60)
    logger.info("MEMORY-EFFICIENT WST DATASET PRECOMPUTATION")
    logger.info("=" * 60)
    
    try:
        windows_processed = precomputer.precompute_streaming()
        precomputer.verify_output()
        
        logger.info(f"✅ Success! Processed {windows_processed:,} windows")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Precomputation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())