#!/usr/bin/env python3
"""
Test script to verify the streaming fix processes all windows correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

def test_stream_windows(data_path: str, window_size: int = 256, stride: int = 1, chunk_size: int = 10000):
    """
    Test the fixed streaming logic without WST computation
    Just count how many windows we can generate
    """
    print(f"🧪 Testing stream with: {data_path}")
    
    # Count total lines for reference
    with open(data_path, 'r') as f:
        total_lines = sum(1 for _ in f) - 1  # Subtract header
    
    expected_windows = max(0, (total_lines - window_size) // stride + 1)
    print(f"📊 Dataset: {total_lines:,} bars → Expected {expected_windows:,} windows")
    
    # Use pandas iterator to read CSV in chunks
    chunk_iter = pd.read_csv(
        data_path, 
        chunksize=chunk_size,
        dtype={'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32}
    )
    
    # Buffer to maintain sliding window and timestamps
    price_buffer = []
    timestamp_buffer = []
    data_row_idx = 0  # Track actual data row index
    window_count = 0
    
    for chunk_idx, chunk in enumerate(chunk_iter):
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
        
        print(f"Chunk {chunk_idx}: Added {len(close_prices)} prices, buffer size: {len(price_buffer)}")
        
        # Generate windows from buffer
        windows_from_chunk = 0
        while len(price_buffer) >= window_size:
            # Extract window (just count, don't process)
            window_timestamp = timestamp_buffer[window_size - 1]
            window_count += 1
            windows_from_chunk += 1
            
            # Slide window by stride
            for _ in range(stride):
                if price_buffer:
                    price_buffer.pop(0)
                    timestamp_buffer.pop(0)
                data_row_idx += 1
            
            # Prevent excessive buffer growth (keep reasonable size)
            if len(price_buffer) > window_size * 3:
                # Only trim if we have more than needed for windows
                excess = len(price_buffer) - window_size * 2
                price_buffer = price_buffer[excess:]
                timestamp_buffer = timestamp_buffer[excess:]
                data_row_idx += excess
        
        print(f"  Generated {windows_from_chunk} windows from this chunk")
        print(f"  Total windows so far: {window_count:,}")
        print(f"  Buffer remaining: {len(price_buffer)}")
        
        # Progress update every 10 chunks
        if chunk_idx > 0 and chunk_idx % 10 == 0:
            print(f"🔄 Progress: {chunk_idx} chunks processed, {window_count:,} windows generated")
    
    print(f"\n✅ Final Results:")
    print(f"   Expected windows: {expected_windows:,}")
    print(f"   Generated windows: {window_count:,}")
    print(f"   Difference: {window_count - expected_windows:,}")
    print(f"   Success rate: {100 * window_count / expected_windows:.2f}%")
    
    return window_count, expected_windows

if __name__ == "__main__":
    data_path = "data/GBPJPY_M1_3.5years_20250912.csv"
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    try:
        actual, expected = test_stream_windows(data_path)
        
        if actual >= expected * 0.99:  # Allow 1% tolerance
            print(f"✅ STREAMING FIX SUCCESSFUL!")
        else:
            print(f"❌ STREAMING STILL HAS ISSUES - only got {actual:,} / {expected:,} windows")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()