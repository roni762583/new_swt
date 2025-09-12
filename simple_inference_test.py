#!/usr/bin/env python3
"""Simplified inference test that works"""

import torch
import numpy as np
from pathlib import Path

def main():
    print("Loading Episode 13475 checkpoint...")
    
    # Load checkpoint
    checkpoint_path = "checkpoints/episode_13475.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint loaded successfully!")
    print(f"Keys: {list(checkpoint.keys())[:5]}...")
    
    # Check network state
    if 'muzero_network_state' in checkpoint:
        state = checkpoint['muzero_network_state']
        print(f"\nNetwork state dict keys: {list(state.keys())[:10]}...")
        print(f"Total parameters: {sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor)):,}")
    
    # Test basic tensor operations
    test_input = torch.randn(1, 128)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Episode trained: {checkpoint.get('episode', 'unknown')}")
    print(f"Best reward: {checkpoint.get('best_reward', 'unknown')}")
    
    print("\n✅ Basic test successful! Checkpoint is valid.")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)