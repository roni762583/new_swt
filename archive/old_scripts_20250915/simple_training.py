#!/usr/bin/env python3
"""
Simple training script without matplotlib dependency for 8-core utilization
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# Set thread counts for all 8 cores
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMBA_NUM_THREADS'] = '8'

print(f"🚀 Simple Training Starting with {torch.get_num_threads()} CPU threads")
print(f"   PyTorch version: {torch.__version__}")
print(f"   NumPy version: {np.__version__}")
print(f"   CPU cores available: {os.cpu_count()}")

# Simulate training loop
epoch = 0
while True:
    epoch += 1

    # Simulate batch processing with all cores
    batch_size = 32
    feature_dim = 137

    # Generate random batch
    batch_data = torch.randn(batch_size, feature_dim)

    # Simulate forward pass with multi-threading
    with torch.no_grad():
        # Matrix multiplication to utilize multiple cores
        hidden = torch.mm(batch_data, torch.randn(feature_dim, 256))
        output = torch.mm(hidden, torch.randn(256, 128))

    # Print status every 10 epochs
    if epoch % 10 == 0:
        print(f"   Epoch {epoch}: Processing batch shape {batch_data.shape} using {torch.get_num_threads()} threads")

    # Sleep briefly to avoid spinning
    time.sleep(0.1)

    # Stop after 100 epochs for demo
    if epoch >= 100:
        print(f"✅ Training demo completed after {epoch} epochs")
        break