#!/usr/bin/env python3
"""
WST Coefficient Precomputation for SWT Training
Precomputes Wavelet Scattering Transform coefficients for entire GBPJPY dataset
Saves aligned WST coefficients with timestamps for efficient training
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import gc

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "experimental_research"))

from experimental_research.swt_models.swt_wavelet_scatter import EnhancedWSTCNN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WSTPrecomputeEngine:
    """Precomputes and saves WST coefficients for entire dataset"""
    
    def __init__(self, data_file: str, output_dir: str = "precomputed_wst"):
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # WST Configuration (matching SWT training config)
        self.wst_config = {
            'J': 2,                    # Number of scales
            'Q': 6,                    # Number of wavelets per octave  
            'input_length': 256,       # Price series length
            'market_output_dim': 128,  # Output feature dimensions
            'dropout_p': 0.0,          # No dropout for precomputation
            'use_batch_norm': True,
            'use_residual': True
        }
        
        # Processing parameters
        self.chunk_size = 10000      # Process in chunks to manage memory
        self.price_series_length = 256
        
        logger.info(f"🚀 WST Precompute Engine initialized")
        logger.info(f"📁 Input: {self.data_file}")
        logger.info(f"📂 Output: {self.output_dir}")
        logger.info(f"⚙️  WST Config: J={self.wst_config['J']}, Q={self.wst_config['Q']}")
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate input data"""
        logger.info("📊 Loading GBPJPY data...")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Load data
        df = pd.read_csv(self.data_file)
        logger.info(f"✅ Loaded {len(df):,} bars")
        
        # Validate columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check data quality
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if df[col].isnull().any():
                logger.warning(f"⚠️ NaN values in {col}, forward filling...")
                df[col] = df[col].fillna(method='ffill')
        
        logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    
    def create_price_series(self, df: pd.DataFrame, end_idx: int) -> np.ndarray:
        """Create 256-length price series ending at end_idx"""
        start_idx = max(0, end_idx - self.price_series_length + 1)
        
        if end_idx < self.price_series_length - 1:
            # Not enough history - pad with first available price
            available_prices = df['close'].iloc[:end_idx + 1].values
            if len(available_prices) == 0:
                return np.zeros(self.price_series_length, dtype=np.float32)
            
            # Pad with first price
            padding_length = self.price_series_length - len(available_prices)
            padded = np.concatenate([
                np.full(padding_length, available_prices[0]),
                available_prices
            ])
            return padded.astype(np.float32)
        else:
            # Full history available
            prices = df['close'].iloc[start_idx:end_idx + 1].values
            return prices.astype(np.float32)
    
    def initialize_wst_model(self) -> EnhancedWSTCNN:
        """Initialize WST model for coefficient computation"""
        logger.info("🧠 Initializing WST model...")
        
        try:
            model = EnhancedWSTCNN(
                wst_config=self.wst_config,
                output_dim=self.wst_config['market_output_dim']
            )
            
            # Set to evaluation mode and disable gradients
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            logger.info("✅ WST model initialized")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WST model: {e}")
            raise
    
    def precompute_coefficients(self) -> None:
        """Main precomputation process"""
        logger.info("🚀 Starting WST coefficient precomputation")
        
        # Load data
        df = self.load_and_validate_data()
        total_bars = len(df)
        
        # Initialize WST model  
        wst_model = self.initialize_wst_model()
        
        # Determine output files
        timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        wst_file = self.output_dir / f"wst_coefficients_{timestamp_suffix}.h5"
        metadata_file = self.output_dir / f"wst_metadata_{timestamp_suffix}.json"
        
        logger.info(f"💾 Output files:")
        logger.info(f"   WST coefficients: {wst_file}")
        logger.info(f"   Metadata: {metadata_file}")
        
        # Precompute in chunks
        logger.info(f"⚡ Processing {total_bars:,} bars in chunks of {self.chunk_size:,}")
        
        with h5py.File(wst_file, 'w') as h5f:
            # Create datasets
            timestamps_ds = h5f.create_dataset(
                'timestamps', 
                (total_bars,), 
                dtype='S32'  # String timestamps
            )
            coefficients_ds = h5f.create_dataset(
                'wst_coefficients',
                (total_bars, self.wst_config['market_output_dim']),
                dtype=np.float32,
                compression='gzip',
                compression_opts=6
            )
            
            # Process in chunks
            processed_count = 0
            
            for chunk_start in tqdm(range(0, total_bars, self.chunk_size), desc="Processing chunks"):
                chunk_end = min(chunk_start + self.chunk_size, total_bars)
                chunk_size_actual = chunk_end - chunk_start
                
                # Prepare batch data
                batch_timestamps = []
                batch_prices = []
                
                for i in range(chunk_start, chunk_end):
                    # Get timestamp
                    timestamp = df['timestamp'].iloc[i]
                    batch_timestamps.append(timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
                    
                    # Create price series
                    price_series = self.create_price_series(df, i)
                    batch_prices.append(price_series)
                
                # Convert to tensor
                batch_tensor = torch.FloatTensor(batch_prices).unsqueeze(1)  # (batch, 1, 256)
                
                # Compute WST coefficients
                with torch.no_grad():
                    try:
                        # Get WST coefficients from model
                        wst_coeffs = wst_model(batch_tensor).numpy()  # (batch, 128)
                        
                        # Save to HDF5
                        timestamps_ds[chunk_start:chunk_end] = [
                            ts.encode('utf-8') for ts in batch_timestamps
                        ]
                        coefficients_ds[chunk_start:chunk_end] = wst_coeffs
                        
                        processed_count += chunk_size_actual
                        
                        # Log progress
                        if processed_count % (self.chunk_size * 10) == 0:
                            logger.info(f"   Processed {processed_count:,}/{total_bars:,} bars ({processed_count/total_bars*100:.1f}%)")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing chunk {chunk_start}-{chunk_end}: {e}")
                        raise
                
                # Memory cleanup
                del batch_tensor, wst_coeffs, batch_prices
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_file': str(self.data_file),
            'total_bars': total_bars,
            'wst_config': self.wst_config,
            'price_series_length': self.price_series_length,
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'file_info': {
                'wst_file': str(wst_file),
                'file_size_mb': wst_file.stat().st_size / 1024 / 1024
            }
        }
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("🎯 WST PRECOMPUTATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Processed: {processed_count:,} bars")
        logger.info(f"💾 WST coefficients: {wst_file} ({wst_file.stat().st_size / 1024 / 1024:.1f} MB)")
        logger.info(f"📄 Metadata: {metadata_file}")
        logger.info(f"📊 Feature dimensions: {self.wst_config['market_output_dim']}")
        logger.info(f"⚡ Ready for SWT training!")
        logger.info("=" * 60)
    
    def verify_precomputed_data(self, wst_file: str) -> bool:
        """Verify precomputed WST coefficients"""
        logger.info("🔍 Verifying precomputed WST data...")
        
        try:
            with h5py.File(wst_file, 'r') as h5f:
                timestamps = h5f['timestamps']
                coefficients = h5f['wst_coefficients']
                
                logger.info(f"   Timestamps: {len(timestamps):,}")
                logger.info(f"   Coefficients: {coefficients.shape}")
                
                # Sample verification
                sample_coeffs = coefficients[1000:1010]  # Sample 10 entries
                
                # Check for NaN/Inf
                if np.isnan(sample_coeffs).any() or np.isinf(sample_coeffs).any():
                    logger.error("❌ Found NaN/Inf values in coefficients")
                    return False
                
                # Check reasonable ranges
                coeff_mean = np.mean(sample_coeffs)
                coeff_std = np.std(sample_coeffs)
                logger.info(f"   Sample stats: mean={coeff_mean:.4f}, std={coeff_std:.4f}")
                
                logger.info("✅ WST data verification passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False


def main():
    """Main execution"""
    try:
        # Default to the new 3.5 year dataset
        data_file = "data/GBPJPY_M1_3.5years_20250912.csv"
        
        if len(sys.argv) > 1:
            data_file = sys.argv[1]
        
        if not Path(data_file).exists():
            logger.error(f"❌ Data file not found: {data_file}")
            logger.info("Usage: python precompute_wst_coefficients.py [data_file.csv]")
            return 1
        
        # Initialize and run precomputation
        engine = WSTPrecomputeEngine(data_file)
        engine.precompute_coefficients()
        
        # Find and verify the latest WST file
        wst_files = list(engine.output_dir.glob("wst_coefficients_*.h5"))
        if wst_files:
            latest_wst = max(wst_files, key=lambda x: x.stat().st_mtime)
            engine.verify_precomputed_data(str(latest_wst))
        
        logger.info("🎉 WST precomputation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Precomputation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())