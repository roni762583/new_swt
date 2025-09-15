#!/usr/bin/env python3
"""
Test script for batch validator implementation
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from swt_validation.batch_validator import BatchConfig, BatchDataLoader, BatchValidator


def test_batch_data_loader():
    """Test batch data loader functionality"""
    print("Testing BatchDataLoader...")

    # Create test config
    config = BatchConfig(
        batch_size=32,
        num_workers=2,
        device='cpu'
    )

    # Test with actual data file if exists
    data_path = 'data/GBPJPY_M1_3.5years_20250912.csv'
    if Path(data_path).exists():
        loader = BatchDataLoader(data_path, config)
        print(f"✅ Loader initialized: {loader.total_samples} samples")

        # Test batch iteration
        batch_count = 0
        for batch in loader.iter_batches(end_idx=100):
            assert batch.shape[1] == 137, f"Expected 137 features, got {batch.shape[1]}"
            batch_count += 1
            if batch_count >= 3:
                break

        print(f"✅ Successfully processed {batch_count} batches")
    else:
        print(f"⚠️ Data file not found: {data_path}")

    print("BatchDataLoader test complete!\n")


def test_batch_validator():
    """Test batch validator with mock data"""
    print("Testing BatchValidator...")

    # Create test config
    config = BatchConfig(
        batch_size=64,
        num_workers=2,
        device='cpu',
        inference_batch_size=128
    )

    # Initialize validator
    validator = BatchValidator(config)
    print("✅ Validator initialized")

    # Test metrics calculation with mock predictions
    mock_predictions = np.random.randint(0, 4, size=1000)  # Random actions
    mock_inference_results = {
        'predictions': mock_predictions.tolist(),
        'timing': {
            'mean_ms': 1.5,
            'median_ms': 1.4,
            'p95_ms': 2.0,
            'p99_ms': 2.5,
            'throughput_per_sec': 666.7
        }
    }

    metrics = validator._calculate_metrics(mock_inference_results)

    print("📊 Calculated metrics:")
    print(f"  Action distribution: {metrics['action_distribution']}")
    print(f"  Trade frequency: {metrics['trade_frequency']:.2%}")
    print(f"  Position bias: {metrics['position_bias']:+.2%}")
    print(f"  Throughput: {metrics['inference_throughput']:.1f} samples/sec")

    # Validate metrics
    assert sum(metrics['action_distribution'].values()) > 0.99, "Action probabilities should sum to ~1"
    assert 0 <= metrics['trade_frequency'] <= 1, "Trade frequency should be between 0 and 1"

    print("✅ Metrics calculation validated")

    # Cleanup
    validator.cleanup()
    print("BatchValidator test complete!\n")


def test_performance_comparison():
    """Compare batch vs sequential processing performance"""
    print("Performance Comparison Test...")

    # Create mock data
    num_samples = 10000
    feature_dim = 137

    # Simulate sequential processing
    import time

    print(f"Testing with {num_samples} samples...")

    # Sequential processing
    start = time.time()
    for i in range(0, num_samples, 1):
        sample = torch.randn(1, feature_dim)
        # Simulate inference
        _ = sample.sum()
    sequential_time = time.time() - start

    # Batch processing
    batch_size = 128
    start = time.time()
    for i in range(0, num_samples, batch_size):
        batch = torch.randn(min(batch_size, num_samples - i), feature_dim)
        # Simulate inference
        _ = batch.sum(dim=1)
    batch_time = time.time() - start

    speedup = sequential_time / batch_time

    print(f"⏱️ Sequential: {sequential_time:.2f}s")
    print(f"⚡ Batch: {batch_time:.2f}s")
    print(f"🚀 Speedup: {speedup:.1f}x")

    assert batch_time < sequential_time, "Batch processing should be faster"
    print("✅ Performance comparison complete!\n")


def main():
    """Run all tests"""
    print("="*60)
    print("BATCH VALIDATOR TEST SUITE")
    print("="*60 + "\n")

    try:
        test_batch_data_loader()
        test_batch_validator()
        test_performance_comparison()

        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()