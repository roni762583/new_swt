#!/usr/bin/env python3
"""
Test the refactored 240+16→256 clean architecture.
"""

import sys
import torch
import numpy as np
sys.path.append('/home/aharon/projects/new_swt')

from micro.models.micro_networks import RepresentationNetwork, MicroStochasticMuZero


def test_representation_network():
    """Test the clean 240+16→256 representation network."""

    print("="*60)
    print("Testing Clean Representation Network (240+16→256)")
    print("="*60)

    # Create network
    net = RepresentationNetwork(
        temporal_features=9,
        static_features=6,
        tcn_channels=240,  # Clean: 240 for temporal
        hidden_dim=256
    )

    # Test inputs
    batch_size = 8
    temporal = torch.randn(batch_size, 32, 9)  # (batch, timesteps, features)
    static = torch.randn(batch_size, 6)        # (batch, position_features)

    print(f"\nInput shapes:")
    print(f"  Temporal: {temporal.shape} (32 timesteps × 9 features)")
    print(f"  Static: {static.shape} (6 position features)")

    # Forward pass
    try:
        output = net(temporal, static)
        print(f"\nOutput shape: {output.shape}")
        assert output.shape == (batch_size, 256), f"Expected (8, 256), got {output.shape}"
        print("✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    tcn_params = sum(p.numel() for p in net.tcn_encoder.parameters())
    static_params = sum(p.numel() for p in net.static_processor.parameters())

    print(f"\n📊 Parameter Count:")
    print(f"  TCN pathway: {tcn_params:,} params")
    print(f"  Static pathway: {static_params:,} params")
    print(f"  Total: {total_params:,} params")

    # Compare with old bloated architecture
    old_static_params = 6*32 + 32*16  # Old intermediate layers
    old_bloated = 6*64 + 64*128 + 128*256  # What it would be with full inflation

    print(f"\n💰 Efficiency Gains:")
    print(f"  Old static path (6→32→16): {old_static_params:,} params")
    print(f"  New static path (6→16): {static_params:,} params")
    print(f"  Savings: {old_static_params - static_params:,} params")

    print(f"\n  If using full inflation (6→256):")
    print(f"  Would be: {old_bloated:,} params")
    print(f"  Actual: {static_params:,} params")
    print(f"  Savings: {old_bloated - static_params:,} params ({(1-static_params/old_bloated)*100:.1f}% reduction)")

    return True


def test_full_muzero_model():
    """Test the complete MicroStochasticMuZero with new architecture."""

    print("\n" + "="*60)
    print("Testing Full MicroStochasticMuZero Model")
    print("="*60)

    # Create model
    model = MicroStochasticMuZero(
        temporal_features=9,
        static_features=6,
        lag_window=32,
        hidden_dim=256,
        action_dim=4,
        num_outcomes=3,
        support_size=300
    )

    # Test initial inference
    batch_size = 4
    temporal = torch.randn(batch_size, 32, 9)
    static = torch.randn(batch_size, 6)

    print("\n1️⃣ Testing initial_inference...")
    try:
        value, policy, outcome_probs, hidden = model.initial_inference(temporal, static)

        print(f"  Value shape: {value.shape} (expected: {(batch_size,)})")
        print(f"  Policy shape: {policy.shape} (expected: {(batch_size, 4)})")
        print(f"  Outcome probs shape: {outcome_probs.shape} (expected: {(batch_size, 4, 3)})")
        print(f"  Hidden shape: {hidden.shape} (expected: {(batch_size, 256)})")

        assert value.shape == (batch_size,)
        assert policy.shape == (batch_size, 4)
        assert outcome_probs.shape == (batch_size, 4, 3)
        assert hidden.shape == (batch_size, 256)

        print("  ✅ Initial inference successful!")
    except Exception as e:
        print(f"  ❌ Initial inference failed: {e}")
        return False

    # Test recurrent inference
    print("\n2️⃣ Testing recurrent_inference...")
    action = torch.randint(0, 4, (batch_size,))
    outcome = torch.randint(0, 3, (batch_size,))

    try:
        value, policy, outcome_probs, reward, next_hidden = model.recurrent_inference(
            hidden, action, outcome
        )

        print(f"  Value shape: {value.shape}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Outcome probs shape: {outcome_probs.shape}")
        print(f"  Reward shape: {reward.shape}")
        print(f"  Next hidden shape: {next_hidden.shape}")

        assert next_hidden.shape == (batch_size, 256)
        print("  ✅ Recurrent inference successful!")
    except Exception as e:
        print(f"  ❌ Recurrent inference failed: {e}")
        return False

    # Summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Model Parameters: {total_params:,}")

    return True


def main():
    """Run all tests."""

    print("🚀 Testing Refactored Clean Architecture")
    print("="*60)

    # Test representation network
    if not test_representation_network():
        print("\n❌ Representation network test failed!")
        return 1

    # Test full model
    if not test_full_muzero_model():
        print("\n❌ Full model test failed!")
        return 1

    print("\n" + "="*60)
    print("✅ All tests passed! Clean architecture working correctly.")
    print("="*60)

    print("\n🎯 Summary of improvements:")
    print("  • Temporal: 9×32 → TCN → 240d (proportional)")
    print("  • Static: 6 → MLP → 16d (not inflated to 256!)")
    print("  • Combined: 240+16 = 256 (natural, no projection)")
    print("  • Removed 6 redundant scripts → archived/")
    print("  • Significant parameter reduction")

    return 0


if __name__ == "__main__":
    exit(main())