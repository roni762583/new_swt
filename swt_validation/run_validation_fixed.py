#!/usr/bin/env python3
"""
Fixed validation script that properly handles the 137-feature architecture
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspace')

from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config

def run_timed_comparison():
    """Run timed comparison of episodes 10 and 775"""

    print("="*60)
    print("⏱️  TIMED VALIDATION COMPARISON")
    print("="*60)

    # Load episode 10
    print("\n📦 Loading Episode 10...")
    start = time.time()
    ep10 = load_checkpoint_with_proper_config('checkpoints/episode_10_best.pth')
    ep10_load_time = time.time() - start
    print(f"✅ Episode 10 loaded in {ep10_load_time:.2f}s - hidden_dim={ep10['config'].get('hidden_dim')}")

    # Load episode 775
    print("\n📦 Loading Episode 775...")
    start = time.time()
    ep775 = load_checkpoint_with_proper_config('checkpoints/episode_775_test.pth')
    ep775_load_time = time.time() - start
    print(f"✅ Episode 775 loaded in {ep775_load_time:.2f}s - hidden_dim={ep775['config'].get('hidden_dim')}")

    # Run inference performance test
    print("\n🔬 Running inference performance test (1000 samples)...")

    # Generate test data - IMPORTANT: Use ALL 137 features!
    num_samples = 1000
    test_inputs = torch.randn(num_samples, 137)  # Full 137 features (128 WST + 9 position)

    # Episode 10 inference timing
    print("\n📊 Episode 10 Inference Performance:")
    ep10_times = []

    with torch.no_grad():
        for i in range(num_samples):
            start = time.time()

            # Pass ALL 137 features to representation network
            hidden_state = ep10['networks'].representation_network(test_inputs[i:i+1])
            latent_z = ep10['networks'].chance_encoder(hidden_state)
            policy = ep10['networks'].policy_network(hidden_state, latent_z)
            value = ep10['networks'].value_network(hidden_state, latent_z)

            ep10_times.append((time.time() - start) * 1000)  # Convert to ms

    ep10_times = np.array(ep10_times)
    print(f"  Mean:   {np.mean(ep10_times):.2f} ms")
    print(f"  Median: {np.median(ep10_times):.2f} ms")
    print(f"  Std:    {np.std(ep10_times):.2f} ms")
    print(f"  P95:    {np.percentile(ep10_times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(ep10_times, 99):.2f} ms")
    print(f"  Throughput: {1000/np.mean(ep10_times):.0f} samples/sec")

    # Episode 775 inference timing
    print("\n📊 Episode 775 Inference Performance:")
    ep775_times = []

    with torch.no_grad():
        for i in range(num_samples):
            start = time.time()

            # Pass ALL 137 features to representation network
            hidden_state = ep775['networks'].representation_network(test_inputs[i:i+1])
            latent_z = ep775['networks'].chance_encoder(hidden_state)
            policy = ep775['networks'].policy_network(hidden_state, latent_z)
            value = ep775['networks'].value_network(hidden_state, latent_z)

            ep775_times.append((time.time() - start) * 1000)  # Convert to ms

    ep775_times = np.array(ep775_times)
    print(f"  Mean:   {np.mean(ep775_times):.2f} ms")
    print(f"  Median: {np.median(ep775_times):.2f} ms")
    print(f"  Std:    {np.std(ep775_times):.2f} ms")
    print(f"  P95:    {np.percentile(ep775_times, 95):.2f} ms")
    print(f"  P99:    {np.percentile(ep775_times, 99):.2f} ms")
    print(f"  Throughput: {1000/np.mean(ep775_times):.0f} samples/sec")

    # Comparison
    print("\n📈 COMPARISON RESULTS:")
    print("="*60)

    print(f"\n⏱️  Loading Time:")
    print(f"  Episode 10:  {ep10_load_time:.2f} seconds")
    print(f"  Episode 775: {ep775_load_time:.2f} seconds")
    print(f"  Difference:  {abs(ep775_load_time - ep10_load_time):.2f} seconds")

    print(f"\n⚡ Inference Speed (mean):")
    print(f"  Episode 10:  {np.mean(ep10_times):.2f} ms")
    print(f"  Episode 775: {np.mean(ep775_times):.2f} ms")
    speed_diff = ((np.mean(ep775_times) - np.mean(ep10_times)) / np.mean(ep10_times)) * 100
    if speed_diff > 0:
        print(f"  Episode 10 is {abs(speed_diff):.1f}% faster")
    else:
        print(f"  Episode 775 is {abs(speed_diff):.1f}% faster")

    print(f"\n🔄 Throughput:")
    ep10_throughput = 1000/np.mean(ep10_times)
    ep775_throughput = 1000/np.mean(ep775_times)
    print(f"  Episode 10:  {ep10_throughput:.0f} samples/sec")
    print(f"  Episode 775: {ep775_throughput:.0f} samples/sec")

    print(f"\n📊 Architecture:")
    print(f"  Both use: hidden_dim=256, support_size=601")
    print(f"  Input features: 137 (128 WST + 9 position)")

    # Quick policy comparison
    print(f"\n🎯 Policy Comparison (on random input):")
    test_input = torch.randn(1, 137)

    with torch.no_grad():
        # Episode 10
        h10 = ep10['networks'].representation_network(test_input)
        z10 = ep10['networks'].chance_encoder(h10)
        p10 = torch.softmax(ep10['networks'].policy_network(h10, z10), dim=-1)

        # Episode 775
        h775 = ep775['networks'].representation_network(test_input)
        z775 = ep775['networks'].chance_encoder(h775)
        p775 = torch.softmax(ep775['networks'].policy_network(h775, z775), dim=-1)

        print(f"  Episode 10 action probs:  {p10.squeeze().numpy()}")
        print(f"  Episode 775 action probs: {p775.squeeze().numpy()}")

        # Policy divergence
        kl_div = torch.nn.functional.kl_div(
            torch.log(p10), p775, reduction='sum'
        ).item()
        print(f"  KL divergence: {kl_div:.4f}")

    print("\n" + "="*60)
    print("✅ Validation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    run_timed_comparison()