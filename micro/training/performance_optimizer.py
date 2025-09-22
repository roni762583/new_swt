#!/usr/bin/env python3
"""
Performance optimizations for Micro MuZero training.

Key optimizations:
1. Increase worker count to utilize all CPU cores
2. Add Numba JIT compilation for hot paths
3. Enable torch.jit.script for neural networks
4. Optimize data loading
"""

import torch
import numpy as np
from numba import jit, prange
import multiprocessing as mp


def get_optimal_workers() -> int:
    """
    Get optimal number of workers based on CPU count.

    For training with CPU:
    - Use n_cores - 1 to leave one for main process
    - But at least 4 for decent parallelism
    """
    n_cores = mp.cpu_count()
    # Use most cores but leave some for OS/main thread
    optimal = min(n_cores - 1, max(4, int(n_cores * 0.75)))
    return optimal


@jit(nopython=True, cache=True)
def calculate_market_outcome_jit(
    current_price: float,
    next_price: float,
    rolling_stdev: float,
    threshold_mult: float = 0.33
) -> int:
    """
    JIT-compiled market outcome calculation.

    Returns: 0=UP, 1=NEUTRAL, 2=DOWN
    """
    price_change = next_price - current_price
    threshold = threshold_mult * rolling_stdev

    if price_change > threshold:
        return 0  # UP
    elif price_change < -threshold:
        return 2  # DOWN
    else:
        return 1  # NEUTRAL


@jit(nopython=True, cache=True)
def calculate_rolling_std_jit(prices: np.ndarray, window: int = 20) -> float:
    """JIT-compiled rolling standard deviation."""
    if len(prices) < window:
        return 0.001

    recent = prices[-window:]
    mean = np.mean(recent)
    variance = np.sum((recent - mean) ** 2) / window
    return np.sqrt(variance)


@jit(nopython=True, cache=True, parallel=True)
def process_batch_features_jit(
    temporal_batch: np.ndarray,
    static_batch: np.ndarray,
    normalize: bool = True
) -> tuple:
    """
    JIT-compiled batch feature processing with parallel execution.

    Args:
        temporal_batch: (batch, timesteps, features)
        static_batch: (batch, features)

    Returns:
        Processed temporal and static features
    """
    batch_size = temporal_batch.shape[0]

    # Process in parallel
    for i in prange(batch_size):
        if normalize:
            # Normalize temporal features per sample
            for t in range(temporal_batch.shape[1]):
                for f in range(temporal_batch.shape[2]):
                    val = temporal_batch[i, t, f]
                    # Simple normalization (can be customized)
                    temporal_batch[i, t, f] = np.tanh(val)

            # Normalize static features
            for f in range(static_batch.shape[1]):
                static_batch[i, f] = np.tanh(static_batch[i, f])

    return temporal_batch, static_batch


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model using TorchScript for faster inference.

    Note: This creates a traced version that's faster but less flexible.
    Only use for inference, not training.
    """
    model.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    temporal = torch.randn(batch_size, 32, 9)
    static = torch.randn(batch_size, 6)

    try:
        # Try to trace the representation network
        with torch.no_grad():
            traced = torch.jit.trace(model.representation, (temporal, static))
        model.representation = traced
        print("✅ Representation network optimized with TorchScript")
    except Exception as e:
        print(f"⚠️ Could not optimize representation: {e}")

    model.train()
    return model


class OptimizedConfig:
    """Optimized configuration for maximum performance."""

    def __init__(self, base_config):
        # Copy base config
        self.__dict__.update(base_config.__dict__)

        # Optimize worker count
        self.num_workers = get_optimal_workers()

        # Enable performance flags
        self.use_numba = True
        self.use_torch_jit = False  # Enable only for inference
        self.pin_memory = True
        self.num_threads = 1  # Per worker thread count

        # Batch processing
        self.batch_process_episodes = True
        self.episode_batch_size = min(self.num_workers, 8)

        print(f"🚀 Performance Optimizations:")
        print(f"  • CPU cores available: {mp.cpu_count()}")
        print(f"  • Workers configured: {self.num_workers}")
        print(f"  • Numba JIT: {'Enabled' if self.use_numba else 'Disabled'}")
        print(f"  • Torch JIT: {'Enabled' if self.use_torch_jit else 'Disabled'}")
        print(f"  • Episode batch size: {self.episode_batch_size}")


def benchmark_optimizations():
    """Benchmark the performance improvements."""
    import time

    print("\n📊 Benchmarking Optimizations...")

    # Test 1: Market outcome calculation
    prices = np.random.randn(1000) * 0.01 + 1.5

    # Without JIT (Python)
    start = time.time()
    for i in range(10000):
        if i >= 20:
            stdev = np.std(prices[i-20:i])
            change = prices[i] - prices[i-1]
            threshold = 0.33 * stdev
            if change > threshold:
                outcome = 0
            elif change < -threshold:
                outcome = 2
            else:
                outcome = 1
    python_time = time.time() - start

    # With JIT
    start = time.time()
    for i in range(10000):
        if i >= 20:
            stdev = calculate_rolling_std_jit(prices[:i], 20)
            outcome = calculate_market_outcome_jit(
                prices[i-1], prices[i], stdev, 0.33
            )
    jit_time = time.time() - start

    print(f"  Market outcome calculation:")
    print(f"    Python: {python_time:.4f}s")
    print(f"    Numba:  {jit_time:.4f}s")
    print(f"    Speedup: {python_time/jit_time:.2f}x")

    # Test 2: Batch processing
    temporal = np.random.randn(32, 32, 9).astype(np.float32)
    static = np.random.randn(32, 6).astype(np.float32)

    start = time.time()
    for _ in range(100):
        process_batch_features_jit(temporal.copy(), static.copy(), True)
    jit_batch_time = time.time() - start

    print(f"\n  Batch feature processing (32 samples):")
    print(f"    Numba parallel: {jit_batch_time:.4f}s")
    print(f"    Throughput: {3200/jit_batch_time:.0f} samples/sec")


if __name__ == "__main__":
    print("=" * 60)
    print("MICRO MUZERO PERFORMANCE OPTIMIZER")
    print("=" * 60)

    # Check optimal configuration
    optimal_workers = get_optimal_workers()
    print(f"\n🔧 Recommended Configuration:")
    print(f"  num_workers: {optimal_workers} (current: 4)")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Expected speedup: ~{optimal_workers/4:.1f}x from workers alone")

    # Run benchmarks
    benchmark_optimizations()

    print("\n💡 To apply optimizations:")
    print("  1. Update train_micro_muzero.py: num_workers = %d" % optimal_workers)
    print("  2. Import and use JIT functions from this module")
    print("  3. Rebuild Docker container")
    print("  4. Expected total speedup: 2-3x")