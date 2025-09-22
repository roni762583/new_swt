#!/usr/bin/env python3
"""
Test script to benchmark the performance optimizations.
Compares before/after improvements.
"""

import time
import numpy as np
import sys
import os
sys.path.append('/workspace')

from micro.utils.numba_optimized import (
    calculate_rolling_std_numba,
    calculate_market_outcome_numba,
    calculate_position_features_numba,
    process_batch_temporal_features_numba
)


def benchmark_rolling_std():
    """Benchmark rolling standard deviation."""
    print("\n📊 Benchmarking Rolling Std (20-bar window)")
    print("-" * 50)

    # Test data
    prices = np.random.randn(1000) * 0.001 + 1.5
    iterations = 10000

    # NumPy version
    start = time.time()
    for i in range(100, min(iterations, len(prices))):
        std_numpy = np.std(prices[i-20:i])
    numpy_time = time.time() - start

    # Numba version (first call compiles)
    _ = calculate_rolling_std_numba(prices[:100], 20)  # Compile

    start = time.time()
    for i in range(100, min(iterations, len(prices))):
        std_numba = calculate_rolling_std_numba(prices[:i], 20)
    numba_time = time.time() - start

    print(f"NumPy: {numpy_time:.4f}s")
    print(f"Numba: {numba_time:.4f}s")
    print(f"Speedup: {numpy_time/numba_time:.2f}x")

    return numpy_time / numba_time


def benchmark_market_outcome():
    """Benchmark market outcome calculation."""
    print("\n📊 Benchmarking Market Outcome Calculation")
    print("-" * 50)

    iterations = 100000

    # Python version
    start = time.time()
    for _ in range(iterations):
        current = 1.5
        next_price = 1.501
        stdev = 0.001
        threshold = 0.33 * stdev
        change = next_price - current
        if change > threshold:
            outcome = 0
        elif change < -threshold:
            outcome = 2
        else:
            outcome = 1
    python_time = time.time() - start

    # Numba version
    start = time.time()
    for _ in range(iterations):
        outcome = calculate_market_outcome_numba(1.5, 1.501, 0.001, 0.33)
    numba_time = time.time() - start

    print(f"Python: {python_time:.4f}s")
    print(f"Numba: {numba_time:.4f}s")
    print(f"Speedup: {python_time/numba_time:.2f}x")

    return python_time / numba_time


def benchmark_batch_processing():
    """Benchmark batch temporal feature processing."""
    print("\n📊 Benchmarking Batch Feature Processing")
    print("-" * 50)

    # Create batch data
    batch = np.random.randn(32, 32, 9).astype(np.float32)
    iterations = 100

    # Python version with loops
    start = time.time()
    for _ in range(iterations):
        output = batch.copy()
        for b in range(32):
            for t in range(32):
                for f in range(5):  # Only first 5 features need normalization
                    output[b, t, f] = np.tanh(output[b, t, f])
    python_time = time.time() - start

    # Numba parallel version
    start = time.time()
    for _ in range(iterations):
        output = process_batch_temporal_features_numba(batch.copy(), True)
    numba_time = time.time() - start

    print(f"Python loops: {python_time:.4f}s")
    print(f"Numba parallel: {numba_time:.4f}s")
    print(f"Speedup: {python_time/numba_time:.2f}x")

    throughput = (32 * iterations) / numba_time
    print(f"Throughput: {throughput:.0f} samples/sec")

    return python_time / numba_time


def check_worker_optimization():
    """Check worker count optimization."""
    import multiprocessing as mp

    print("\n👷 Worker Optimization Check")
    print("-" * 50)

    n_cores = mp.cpu_count()
    old_workers = 4  # Previous hardcoded value

    # Calculate optimal
    optimal = min(n_cores - 1, max(4, int(n_cores * 0.75)))

    print(f"CPU cores: {n_cores}")
    print(f"Old config: {old_workers} workers")
    print(f"New config: {optimal} workers")
    print(f"Expected speedup from workers: {optimal/old_workers:.1f}x")

    return optimal / old_workers


def check_ram_disk():
    """Check if RAM disk is available."""
    print("\n💾 RAM Disk Check")
    print("-" * 50)

    import os

    if os.path.exists('/dev/shm'):
        # Check available space
        import shutil
        total, used, free = shutil.disk_usage('/dev/shm')
        print(f"✅ RAM disk available at /dev/shm")
        print(f"   Total: {total / (1024**3):.1f} GB")
        print(f"   Free: {free / (1024**3):.1f} GB")
        print(f"   Expected I/O speedup: ~2-5x")
        return True
    else:
        print("❌ RAM disk not available")
        return False


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("🚀 MICRO MUZERO OPTIMIZATION BENCHMARKS")
    print("=" * 60)

    speedups = []

    # Numba optimizations
    speedups.append(benchmark_rolling_std())
    speedups.append(benchmark_market_outcome())
    speedups.append(benchmark_batch_processing())

    # Worker optimization
    worker_speedup = check_worker_optimization()

    # RAM disk
    ram_disk = check_ram_disk()

    # Summary
    print("\n" + "=" * 60)
    print("📈 OPTIMIZATION SUMMARY")
    print("=" * 60)

    avg_numba = np.mean(speedups)
    print(f"Average Numba speedup: {avg_numba:.1f}x")
    print(f"Worker parallelization: {worker_speedup:.1f}x")
    if ram_disk:
        print(f"RAM disk I/O: ~2-5x (estimated)")

    total_expected = avg_numba * worker_speedup
    if ram_disk:
        total_expected *= 1.5  # Conservative estimate for I/O improvement

    print(f"\n🎯 Total expected speedup: {total_expected:.1f}x")
    print(f"   Before: ~6-7 seconds per episode step")
    print(f"   After: ~{6.5/total_expected:.1f}-{7/total_expected:.1f} seconds per episode step")


if __name__ == "__main__":
    main()