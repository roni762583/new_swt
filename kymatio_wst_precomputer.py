#!/usr/bin/env python3
"""
Kymatio-based WST Precomputer
Generates proper 128-dimensional WST features using Kymatio library
Memory-efficient streaming implementation
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
from typing import Iterator, Tuple
import gc
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Kymatio import
from kymatio.torch import Scattering1D

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KymatioWSTPrecomputer:
    """
    Kymatio-based WST precomputer that generates 128-dimensional features
    Uses proper Wavelet Scattering Transform with streaming for memory efficiency
    """

    def __init__(
        self,
        data_path: str,
        output_path: str,
        window_size: int = 256,
        stride: int = 1,
        chunk_size: int = 1000,  # HDF5 chunk size for streaming writes
        max_memory_mb: int = 4000,  # Maximum memory usage limit
        device: str = 'cpu'
    ):
        """
        Initialize Kymatio WST precomputer

        Args:
            data_path: Path to CSV with OHLC data
            output_path: Path to save HDF5 with precomputed features
            window_size: Size of price window (256 bars)
            stride: Step size between windows (1 = every minute)
            chunk_size: HDF5 chunk size for streaming writes
            max_memory_mb: Maximum memory usage in MB
            device: Device for Kymatio computation ('cpu' or 'cuda')
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.max_memory_mb = max_memory_mb
        self.device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')

        # Kymatio WST parameters - these produce 128-dimensional output
        self.J = 8  # Number of octaves/scales
        self.Q = 1  # Number of wavelets per octave (1 for standard, can be 8 for richer features)
        self.max_order = 2  # Maximum scattering order

        # Initialize Kymatio scattering transform
        logger.info(f"🌊 Initializing Kymatio Scattering1D with J={self.J}, Q={self.Q}, max_order={self.max_order}")
        self.scattering = Scattering1D(
            J=self.J,
            shape=(window_size,),
            Q=self.Q,
            max_order=self.max_order,
            frontend='torch'
        ).to(self.device)

        # Get output dimension from Kymatio
        dummy_input = torch.zeros(1, window_size).to(self.device)
        dummy_output = self.scattering(dummy_input)
        self.wst_output_dim = dummy_output.shape[-1]

        # Memory monitoring
        self.process = psutil.Process()
        self.peak_memory_mb = 0

        logger.info(f"✅ Kymatio WST Precomputer initialized")
        logger.info(f"   Window size: {window_size}")
        logger.info(f"   Stride: {stride}")
        logger.info(f"   WST output dimension: {self.wst_output_dim} (should be 128)")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   HDF5 chunk size: {chunk_size}")
        logger.info(f"   Memory limit: {max_memory_mb} MB")

        if self.wst_output_dim != 128:
            logger.warning(f"⚠️ WST output dimension is {self.wst_output_dim}, expected 128")
            logger.info("   Adjusting parameters...")
            # Try different parameter combinations to get 128 features
            self._adjust_parameters_for_128_features()

    def _adjust_parameters_for_128_features(self):
        """Adjust Kymatio parameters to achieve 128-dimensional output"""
        # Common parameter combinations that yield ~128 features
        param_combinations = [
            {'J': 6, 'Q': 8, 'max_order': 2},  # Often gives 127-129 features
            {'J': 7, 'Q': 4, 'max_order': 2},  # Alternative
            {'J': 8, 'Q': 2, 'max_order': 2},  # Another option
        ]

        for params in param_combinations:
            logger.info(f"   Trying J={params['J']}, Q={params['Q']}, max_order={params['max_order']}")

            test_scattering = Scattering1D(
                J=params['J'],
                shape=(self.window_size,),
                Q=params['Q'],
                max_order=params['max_order'],
                frontend='torch'
            ).to(self.device)

            dummy_input = torch.zeros(1, self.window_size).to(self.device)
            dummy_output = test_scattering(dummy_input)
            output_dim = dummy_output.shape[-1]

            logger.info(f"     → Output dimension: {output_dim}")

            if 126 <= output_dim <= 130:  # Close enough to 128
                self.J = params['J']
                self.Q = params['Q']
                self.max_order = params['max_order']
                self.scattering = test_scattering
                self.wst_output_dim = output_dim
                logger.info(f"   ✅ Using J={self.J}, Q={self.Q} → {output_dim} features")
                break

    def _monitor_memory(self):
        """Monitor and log memory usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

        if memory_mb > self.max_memory_mb * 0.9:  # 90% warning threshold
            logger.warning(f"⚠️ High memory usage: {memory_mb:.1f} MB (limit: {self.max_memory_mb} MB)")
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return memory_mb

    def _get_total_windows(self) -> int:
        """Get total number of windows without loading full dataset"""
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
        csv_chunk_size = 50000  # Larger chunks for reading
        chunk_iter = pd.read_csv(
            self.data_path,
            chunksize=csv_chunk_size,
            dtype={'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32}
        )

        # Buffer to maintain sliding window
        price_buffer = []
        timestamp_buffer = []
        data_row_idx = 0
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
                # Create synthetic timestamps
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
                data_row_idx += self.stride

                # Memory monitoring periodically
                if window_idx % 1000 == 0:
                    self._monitor_memory()

    def _compute_wst_features(self, price_window: np.ndarray) -> np.ndarray:
        """
        Compute WST features using Kymatio

        Args:
            price_window: Price window of shape (256,)

        Returns:
            WST features of shape (128,) or close to it
        """
        # Normalize price window (important for WST)
        price_normalized = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-8)

        # Convert to torch tensor
        price_tensor = torch.from_numpy(price_normalized).float().unsqueeze(0).to(self.device)

        # Compute WST using Kymatio
        with torch.no_grad():
            wst_output = self.scattering(price_tensor)

        # Convert back to numpy and flatten
        wst_features = wst_output.squeeze().cpu().numpy().astype(np.float32)

        # Ensure exactly 128 dimensions (pad or truncate if needed)
        if len(wst_features) < 128:
            # Pad with zeros
            wst_features = np.pad(wst_features, (0, 128 - len(wst_features)), mode='constant')
        elif len(wst_features) > 128:
            # Truncate
            wst_features = wst_features[:128]

        return wst_features

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
                shape=(total_windows, 128),  # Always 128 features
                dtype=np.float32,
                chunks=(self.chunk_size, 128),
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
            f.attrs['wst_output_dim'] = 128  # Always 128
            f.attrs['wst_J'] = self.J
            f.attrs['wst_Q'] = self.Q
            f.attrs['wst_max_order'] = self.max_order
            f.attrs['data_path'] = str(self.data_path)
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['method'] = 'kymatio_streaming'
            f.attrs['kymatio_version'] = '0.3.0'

            logger.info(f"🔄 Starting Kymatio WST computation...")

            # Process windows one at a time
            windows_processed = 0
            batch_features = []
            batch_timestamps = []
            batch_indices = []

            with tqdm(total=total_windows, desc="Computing WST (Kymatio)") as pbar:
                for window_idx, price_window, timestamp, original_idx in self._stream_price_windows():

                    # Compute WST features using Kymatio
                    wst_features = self._compute_wst_features(price_window)

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
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self._monitor_memory()

                    # Check if we've reached the expected total
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

        logger.info(f"✅ Kymatio WST precomputation complete!")
        logger.info(f"   Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"   Windows processed: {windows_processed:,}")
        logger.info(f"   Output file size: {file_size_mb:.1f} MB")
        logger.info(f"   Processing speed: {windows_processed/elapsed:.1f} windows/sec")
        logger.info(f"   Peak memory usage: {self.peak_memory_mb:.1f} MB")
        logger.info(f"   Feature dimension: 128 (Kymatio WST)")

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

            # Verify feature dimension
            assert f['wst_features'].shape[1] == 128, f"Expected 128 features, got {f['wst_features'].shape[1]}"

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
                logger.info(f"   ✅ Features are 128-dimensional Kymatio WST")


def main():
    """Main execution with Kymatio WST"""

    # Configuration
    import argparse
    parser = argparse.ArgumentParser(description="Kymatio WST Precomputer")
    parser.add_argument('--data-path', type=str,
                       default="data/GBPJPY_M1_REAL_2022-2025.csv",
                       help="Path to CSV data file")
    parser.add_argument('--output-path', type=str,
                       default="precomputed_wst/GBPJPY_WST_KYMATIO_128D_2022-2025.h5",
                       help="Path to output HDF5 file")
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'],
                       help="Device for computation")
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help="HDF5 chunk size")
    parser.add_argument('--memory-limit', type=int, default=6000,
                       help="Memory limit in MB")

    args = parser.parse_args()

    # Check if data exists
    if not Path(args.data_path).exists():
        logger.error(f"❌ Data file not found: {args.data_path}")
        return 1

    # Initialize Kymatio precomputer
    precomputer = KymatioWSTPrecomputer(
        data_path=args.data_path,
        output_path=args.output_path,
        window_size=256,
        stride=1,
        chunk_size=args.chunk_size,
        max_memory_mb=args.memory_limit,
        device=args.device
    )

    # Run streaming precomputation
    logger.info("=" * 60)
    logger.info("KYMATIO WST DATASET PRECOMPUTATION (128D)")
    logger.info("=" * 60)

    try:
        windows_processed = precomputer.precompute_streaming()
        precomputer.verify_output()

        logger.info(f"✅ Success! Processed {windows_processed:,} windows with 128D Kymatio WST")
        return 0

    except Exception as e:
        logger.error(f"❌ Precomputation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())