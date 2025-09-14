#!/usr/bin/env python3
"""
Precompute WST features for entire dataset using Numba JIT
Saves results with timestamps for fast training data loading
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import h5py
import logging
from tqdm import tqdm
from typing import Tuple, Dict, Any

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


class WSTDatasetPrecomputer:
    """Precompute WST features for entire trading dataset"""
    
    def __init__(
        self,
        data_path: str,
        output_path: str,
        wst_config: Dict[str, Any] = None,
        window_size: int = 256,
        stride: int = 1,  # Compute WST for every minute
        batch_size: int = 1000  # Process in batches for memory efficiency
    ):
        """
        Initialize WST precomputer
        
        Args:
            data_path: Path to CSV with OHLC data
            output_path: Path to save HDF5 with precomputed features
            wst_config: WST configuration (J, Q, etc.)
            window_size: Size of price window (256 bars)
            stride: Step size between windows (1 = every minute)
            batch_size: Number of windows to process at once
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        
        # Default WST config
        if wst_config is None:
            wst_config = {
                'J': 2,
                'Q': 6,
                'backend': 'fallback'  # Numba JIT
            }
        
        # Initialize WST transform using EnhancedWSTCNN
        self.wst_cnn = EnhancedWSTCNN(
            wst_config=wst_config,
            output_dim=wst_config.get('market_output_dim', 128)
        )
        # Access the WST component directly
        self.wst = self.wst_cnn.wst
        
        # Get output dimension using the WST component only (not CNN)
        dummy_input = torch.randn(1, 1, window_size)
        with torch.no_grad():
            dummy_output = self.wst(dummy_input)
        self.wst_output_dim = dummy_output.shape[-1]
        
        logger.info(f"🌊 WST Precomputer initialized")
        logger.info(f"   Window size: {window_size}")
        logger.info(f"   Stride: {stride} (compute every {stride} bar(s))")
        logger.info(f"   WST output dimension: {self.wst_output_dim}")
        logger.info(f"   Batch size: {batch_size}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare price data"""
        logger.info(f"📂 Loading data from {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        # Handle timestamp column (various possible names)
        time_cols = ['time', 'timestamp', 'datetime', 'date']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            df['timestamp'] = pd.to_datetime(df[time_col])
        else:
            # Create synthetic timestamps (1 minute intervals)
            df['timestamp'] = pd.date_range(
                start='2022-01-01', 
                periods=len(df), 
                freq='1min'
            )
        
        logger.info(f"✅ Loaded {len(df):,} bars")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_price_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sliding windows of price data
        
        Returns:
            price_windows: Array of shape (n_windows, window_size, 4) for OHLC
            timestamps: Timestamp for each window (end timestamp)
            indices: Original dataframe indices for each window
        """
        n_bars = len(df)
        n_windows = (n_bars - self.window_size) // self.stride + 1
        
        logger.info(f"📊 Preparing {n_windows:,} windows")
        
        # Prepare OHLC data
        ohlc = df[['open', 'high', 'low', 'close']].values
        
        # Create sliding windows
        price_windows = []
        timestamps = []
        indices = []
        
        for i in range(0, n_bars - self.window_size + 1, self.stride):
            window = ohlc[i:i + self.window_size]
            price_windows.append(window)
            
            # Use the last timestamp of the window
            timestamps.append(df['timestamp'].iloc[i + self.window_size - 1])
            indices.append(i + self.window_size - 1)
        
        price_windows = np.array(price_windows)
        timestamps = np.array(timestamps)
        indices = np.array(indices)
        
        logger.info(f"✅ Prepared windows shape: {price_windows.shape}")
        
        return price_windows, timestamps, indices
    
    def compute_wst_batch(self, price_batch: np.ndarray) -> np.ndarray:
        """
        Compute WST features for a batch of price windows
        
        Args:
            price_batch: Shape (batch, window_size, 4) for OHLC
            
        Returns:
            wst_features: Shape (batch, wst_output_dim)
        """
        batch_size = price_batch.shape[0]
        
        # Use close prices for WST (could also use OHLC average)
        close_prices = price_batch[:, :, 3]  # Close is 4th column
        
        # Reshape for WST: (batch, 1, window_size)
        input_tensor = torch.tensor(close_prices, dtype=torch.float32).unsqueeze(1)
        
        # Compute WST features
        with torch.no_grad():
            wst_features = self.wst(input_tensor)
        
        return wst_features.numpy()
    
    def precompute_dataset(self):
        """Main precomputation pipeline"""
        start_time = time.time()
        
        # Load data
        df = self.load_data()
        
        # Prepare windows
        price_windows, timestamps, indices = self.prepare_price_windows(df)
        n_windows = len(price_windows)
        
        # Initialize storage for WST features
        wst_features = np.zeros((n_windows, self.wst_output_dim), dtype=np.float32)
        
        # Process in batches
        logger.info(f"🔄 Computing WST features in batches...")
        
        with tqdm(total=n_windows, desc="Computing WST") as pbar:
            for i in range(0, n_windows, self.batch_size):
                batch_end = min(i + self.batch_size, n_windows)
                batch = price_windows[i:batch_end]
                
                # Compute WST for batch
                batch_features = self.compute_wst_batch(batch)
                wst_features[i:batch_end] = batch_features
                
                pbar.update(batch_end - i)
        
        # Save to HDF5
        logger.info(f"💾 Saving precomputed features to {self.output_path}")
        
        with h5py.File(self.output_path, 'w') as f:
            # Save WST features
            f.create_dataset('wst_features', data=wst_features, compression='gzip')
            
            # Save timestamps (convert to strings for HDF5)
            timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
            f.create_dataset('timestamps', data=np.array(timestamp_strs, dtype='S19'))
            
            # Save indices
            f.create_dataset('indices', data=indices)
            
            # Save metadata
            f.attrs['window_size'] = self.window_size
            f.attrs['stride'] = self.stride
            f.attrs['wst_output_dim'] = self.wst_output_dim
            f.attrs['n_windows'] = n_windows
            f.attrs['wst_J'] = self.wst_cnn.wst_config['J']
            f.attrs['wst_Q'] = self.wst_cnn.wst_config['Q']
            f.attrs['data_path'] = str(self.data_path)
            f.attrs['creation_time'] = datetime.now().isoformat()
            
            # Save original OHLC data for reference
            f.create_dataset('price_windows', data=price_windows, compression='gzip')
        
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        
        logger.info(f"✅ Precomputation complete!")
        logger.info(f"   Time elapsed: {elapsed:.1f} seconds")
        logger.info(f"   Windows processed: {n_windows:,}")
        logger.info(f"   Output file size: {file_size_mb:.1f} MB")
        logger.info(f"   Processing speed: {n_windows/elapsed:.1f} windows/sec")
        
        return wst_features, timestamps, indices
    
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
            
            # Sample a few features
            sample_features = f['wst_features'][:5]
            logger.info(f"   Sample features (first 5):")
            logger.info(f"     Mean: {sample_features.mean():.4f}")
            logger.info(f"     Std: {sample_features.std():.4f}")
            logger.info(f"     Min: {sample_features.min():.4f}")
            logger.info(f"     Max: {sample_features.max():.4f}")


class PrecomputedWSTDataLoader:
    """Fast data loader for precomputed WST features"""
    
    def __init__(self, precomputed_path: str):
        """Load precomputed WST features"""
        self.path = precomputed_path
        
        with h5py.File(self.path, 'r') as f:
            # Load into memory for fast access
            self.wst_features = f['wst_features'][:]
            self.timestamps = f['timestamps'][:]
            self.indices = f['indices'][:]
            self.price_windows = f['price_windows'][:]
            
            # Load metadata
            self.window_size = f.attrs['window_size']
            self.wst_output_dim = f.attrs['wst_output_dim']
            self.n_windows = f.attrs['n_windows']
        
        logger.info(f"📚 Loaded precomputed WST data")
        logger.info(f"   Windows: {self.n_windows:,}")
        logger.info(f"   WST dimension: {self.wst_output_dim}")
    
    def get_features_at_index(self, idx: int) -> np.ndarray:
        """Get WST features at specific index"""
        return self.wst_features[idx]
    
    def get_features_for_timestamp(self, timestamp: str) -> np.ndarray:
        """Get WST features for specific timestamp"""
        idx = np.where(self.timestamps == timestamp.encode())[0]
        if len(idx) == 0:
            raise ValueError(f"Timestamp {timestamp} not found")
        return self.wst_features[idx[0]]
    
    def get_batch(self, start_idx: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch of features and prices"""
        end_idx = min(start_idx + batch_size, self.n_windows)
        return (
            self.wst_features[start_idx:end_idx],
            self.price_windows[start_idx:end_idx]
        )


def main():
    """Main execution"""
    
    # Configuration
    data_path = "data/GBPJPY_M1_3.5years_20250912.csv"
    output_dir = Path("precomputed_wst")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "GBPJPY_WST_3.5years.h5"
    
    # Check if data exists
    if not Path(data_path).exists():
        # Try alternative path
        data_path = "data/GBPJPY_M1_3years_20250912.csv"
        if not Path(data_path).exists():
            logger.error(f"❌ Data file not found: {data_path}")
            return 1
    
    # Initialize precomputer
    precomputer = WSTDatasetPrecomputer(
        data_path=data_path,
        output_path=str(output_path),
        wst_config={
            'J': 2,
            'Q': 6,
            'backend': 'fallback'  # Numba JIT
        },
        window_size=256,
        stride=1,  # Compute for every minute
        batch_size=1000  # Process 1000 windows at a time
    )
    
    # Run precomputation
    logger.info("=" * 60)
    logger.info("WST DATASET PRECOMPUTATION")
    logger.info("=" * 60)
    
    precomputer.precompute_dataset()
    precomputer.verify_output()
    
    # Test loading
    logger.info("\n📖 Testing data loader...")
    loader = PrecomputedWSTDataLoader(str(output_path))
    
    # Get sample features
    sample_features = loader.get_features_at_index(1000)
    logger.info(f"   Sample features at index 1000: shape={sample_features.shape}")
    
    logger.info("\n✅ Precomputation complete! Use PrecomputedWSTDataLoader for fast training.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())