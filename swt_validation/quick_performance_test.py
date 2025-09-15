#!/usr/bin/env python3
"""Quick performance comparison between Episode 10 and 775."""

import time
import torch
import numpy as np
from pathlib import Path
from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

def test_checkpoint(checkpoint_path: str, name: str):
    """Test loading and inference speed."""
    print(f"\n{'='*60}")
    print(f"Testing {name}: {checkpoint_path}")
    print('='*60)

    # Check file size
    size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    # Load checkpoint
    start = time.time()
    checkpoint = load_checkpoint_with_proper_config(checkpoint_path)
    load_time = time.time() - start
    print(f"Load time: {load_time:.2f}s")

    # Get config
    config = checkpoint['config']
    print(f"Architecture: hidden_dim={config.get('hidden_dim')}, support_size={config.get('support_size')}")

    # Test inference speed
    network = checkpoint['networks']
    network.eval()

    # Create full feature vector (128 WST + 9 position features)
    def create_features(batch_size):
        wst_features = torch.randn(batch_size, 128)
        position_features = torch.zeros(batch_size, 9)
        return torch.cat([wst_features, position_features], dim=1)

    # Warm up
    test_input = create_features(1)
    with torch.no_grad():
        # Network expects all 137 features
        _ = network.representation_network(test_input)

    # Time 100 inferences
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            test_batch = create_features(10)  # Batch of 10
            _ = network.representation_network(test_batch)
    inference_time = time.time() - start

    avg_ms = (inference_time / 100) * 1000
    throughput = 1000 / avg_ms  # inferences per second

    print(f"Inference (batch=10): {avg_ms:.1f}ms avg, {throughput:.1f} batches/sec")

    # Return metrics
    return {
        'size_mb': size_mb,
        'load_time': load_time,
        'inference_ms': avg_ms,
        'throughput': throughput
    }

def main():
    print("\n" + "🚀 PERFORMANCE COMPARISON: Episode 10 vs 775 ".center(60, "="))

    # Test Episode 10
    ep10_metrics = test_checkpoint('checkpoints/episode_10_best.pth', 'Episode 10')

    # Test Episode 775 (aggressive)
    ep775_metrics = test_checkpoint('checkpoints/episode_775_aggressive.pth', 'Episode 775 (Aggressive)')

    # Summary
    print("\n" + "📊 SUMMARY ".center(60, "="))
    print(f"\nSize Reduction: {ep775_metrics['size_mb']:.1f}MB vs {ep10_metrics['size_mb']:.1f}MB")
    print(f"Load Time: {ep775_metrics['load_time']:.2f}s vs {ep10_metrics['load_time']:.2f}s")
    print(f"Inference Speed: {ep775_metrics['inference_ms']:.1f}ms vs {ep10_metrics['inference_ms']:.1f}ms")

    # Performance difference
    speed_diff = ((ep10_metrics['inference_ms'] - ep775_metrics['inference_ms']) / ep10_metrics['inference_ms']) * 100
    if abs(speed_diff) < 5:
        print(f"\n✅ Performance: Similar inference speed (within 5%)")
    elif speed_diff > 0:
        print(f"\n✅ Performance: Episode 775 is {speed_diff:.1f}% faster")
    else:
        print(f"\n⚠️ Performance: Episode 775 is {abs(speed_diff):.1f}% slower")

    print("\n" + "="*60)

if __name__ == '__main__':
    main()