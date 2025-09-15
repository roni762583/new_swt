#!/usr/bin/env python3
"""Analyze checkpoint files"""

import torch
import sys
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print("=" * 60)
    print(f"CHECKPOINT ANALYSIS: {Path(checkpoint_path).name}")
    print("=" * 60)

    # Basic info
    print(f"\n📊 Basic Information:")
    print(f"  Episode: {checkpoint.get('episode', 'N/A')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")

    # Performance
    if 'best_performance' in checkpoint:
        print(f"  Best Performance: {checkpoint['best_performance']:.4f}")

    # Episode result
    if 'episode_result' in checkpoint:
        result = checkpoint['episode_result']
        print(f"\n📈 Episode Result:")
        for key, value in result.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")

    # Model architecture
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))

        print(f"\n🏗️ Model Architecture:")
        print(f"  Total Layers: {len(state_dict)}")
        print(f"  Total Parameters: {total_params:,}")

        # Sample layers
        print(f"\n  Sample Layers:")
        for i, (key, tensor) in enumerate(state_dict.items()):
            if i < 5:
                print(f"    {key}: shape={list(tensor.shape)}")
            elif i == 5:
                print(f"    ... and {len(state_dict) - 5} more layers")
                break

    # Optimizer state
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f"\n🔧 Optimizer State:")
        if 'param_groups' in opt_state:
            for i, group in enumerate(opt_state['param_groups']):
                print(f"  Group {i}: lr={group.get('lr', 'N/A')}")

if __name__ == "__main__":
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/app/checkpoints/best_model.pth"
    analyze_checkpoint(checkpoint_path)