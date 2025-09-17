#!/usr/bin/env python3
"""
Test script to verify micro system components.
"""

import os
import sys
import duckdb
import numpy as np

def test_micro_features_db():
    """Test micro_features.duckdb exists and has correct structure."""
    db_path = "/home/aharon/projects/new_swt/data/micro_features.duckdb"

    if not os.path.exists(db_path):
        print("❌ micro_features.duckdb not found")
        return False

    conn = duckdb.connect(db_path, read_only=True)

    # Check table exists
    tables = conn.execute("SHOW TABLES").fetchall()
    if ('micro_features',) not in tables:
        print("❌ micro_features table not found")
        return False

    # Check column count
    cols = conn.execute("DESCRIBE micro_features").fetchall()
    if len(cols) != 297:
        print(f"❌ Wrong column count: {len(cols)} != 297")
        return False

    # Check row count
    rows = conn.execute("SELECT COUNT(*) FROM micro_features").fetchone()[0]
    print(f"✅ micro_features.duckdb: {rows:,} rows, {len(cols)} columns")

    # Sample data
    sample = conn.execute("SELECT * FROM micro_features LIMIT 1").fetchone()
    if sample:
        print(f"✅ Sample data available, bar_index: {sample[1]}")

    conn.close()
    return True

def test_micro_models():
    """Test micro models can be imported."""
    try:
        sys.path.insert(0, '/home/aharon/projects/new_swt')
        from micro.models.micro_networks import MicroStochasticMuZero
        from micro.models.tcn_block import TCNBlock

        # Try creating model
        model = MicroStochasticMuZero()
        params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully: {params:,} parameters")

        # Test forward pass
        import torch
        obs = torch.randn(1, 32, 15)
        hidden, policy, value = model.initial_inference(obs)
        print(f"✅ Forward pass successful: hidden={hidden.shape}, policy={policy.shape}")

        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_incremental_builder():
    """Test incremental feature builder."""
    try:
        from data.incremental_feature_builder import IncrementalFeatureBuilder

        builder = IncrementalFeatureBuilder()
        print("✅ IncrementalFeatureBuilder created")

        # Test feature generation
        test_close = 199.5
        test_features = builder.process_new_bar(test_close)

        if test_features is not None:
            print(f"✅ Feature generation test: shape={test_features.shape}")
            if len(test_features) != 297:
                print(f"❌ Wrong feature count: {len(test_features)}")
                return False
        else:
            print("⚠️ Feature generation returned None (needs initialization)")

        return True
    except Exception as e:
        print(f"❌ Incremental builder failed: {e}")
        return False

def test_training_script():
    """Test training script can be imported."""
    try:
        from micro.training.train_micro_muzero import TrainingConfig, MicroMuZeroTrainer

        config = TrainingConfig()
        print(f"✅ Training config created: {config.num_episodes} episodes target")

        # Don't actually create trainer (would load data)
        print("✅ Training script imports successfully")
        return True
    except Exception as e:
        print(f"❌ Training script failed: {e}")
        return False

def test_directories():
    """Test required directories exist."""
    dirs = [
        "/home/aharon/projects/new_swt/micro/checkpoints",
        "/home/aharon/projects/new_swt/micro/logs",
        "/home/aharon/projects/new_swt/micro/validation_results",
        "/home/aharon/projects/new_swt/micro/live_state"
    ]

    all_exist = True
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ Directory exists: {d}")
        else:
            print(f"❌ Directory missing: {d}")
            all_exist = False

    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("MICRO MUZERO SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Micro Features Database", test_micro_features_db),
        ("Micro Models", test_micro_models),
        ("Incremental Builder", test_incremental_builder),
        ("Training Script", test_training_script),
        ("Directory Structure", test_directories)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System ready for training.")
        print("\nTo start training, run:")
        print("  docker compose up -d --build")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running.")

    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)