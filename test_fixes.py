#!/usr/bin/env python3
"""Test NaN handling and R-multiple calculation fixes."""

import numpy as np
import sys
import os
sys.path.append('/home/aharon/projects/new_swt')

from micro.training.train_micro_muzero import Experience, QualityExperienceBuffer
from swt_core.sqn_calculator import SQNCalculator

def test_nan_handling():
    """Test that NaN quality scores are properly handled."""
    print("Testing NaN handling in buffer sampling...")

    buffer = QualityExperienceBuffer(capacity=100)

    # Add some normal experiences
    for i in range(10):
        exp = Experience(
            observation=np.random.randn(32, 15),
            action=np.random.randint(0, 4),
            policy=np.random.rand(4),
            value=np.random.randn(),
            reward=np.random.randn() * 10,
            done=False,
            pip_pnl=np.random.randn() * 20,
            position_change=(i % 2 == 0),
            session_expectancy=np.random.rand()
        )
        buffer.add(exp)

    # Add experience with NaN values
    exp_nan = Experience(
        observation=np.random.randn(32, 15),
        action=0,
        policy=np.random.rand(4),
        value=np.nan,  # NaN value
        reward=np.nan,  # NaN reward
        done=False,
        pip_pnl=np.nan,  # NaN pip_pnl
        position_change=False,
        session_expectancy=0.5
    )
    buffer.add(exp_nan)

    # Try sampling - should not crash
    try:
        batch = buffer.sample(5)
        print(f"✅ Successfully sampled {len(batch)} experiences despite NaN values")

        # Check if NaN experience was filtered
        has_nan = any(np.isnan(exp.quality_score) for exp in batch)
        if not has_nan:
            print("✅ NaN experiences were properly filtered out")
        else:
            print("⚠️  Warning: NaN experience in batch")

    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        return False

    return True


def test_r_multiple_calculation():
    """Test Van Tharp R-multiple calculation."""
    print("\nTesting Van Tharp R-multiple calculation...")

    # Test data: mix of wins and losses
    pnl_values = [100, -50, 200, -75, 150, -25, 300, -100, 50, -60]

    calculator = SQNCalculator()

    # Calculate with automatic risk estimation
    result = calculator.calculate_sqn(pnl_values)

    print(f"📊 SQN Results:")
    print(f"  - SQN: {result.sqn:.3f}")
    print(f"  - Expectancy: {result.expectancy:.3f}")
    print(f"  - Std Dev: {result.std_dev:.3f}")
    print(f"  - Classification: {result.classification}")
    print(f"  - Num Trades: {result.num_trades}")

    # Check R-multiples calculation
    r_multiples = result.r_multiples
    print(f"\n📈 R-Multiples (first 5): {r_multiples[:5]}")

    # Verify risk estimation from losses
    losses = [abs(pnl) for pnl in pnl_values if pnl < 0]
    estimated_risk = np.mean(losses)
    print(f"  - Estimated risk from losses: {estimated_risk:.2f}")

    # Manual verification of first R-multiple
    expected_r = pnl_values[0] / estimated_risk
    actual_r = r_multiples[0]

    if abs(expected_r - actual_r) < 0.01:
        print(f"✅ R-multiple calculation correct: {actual_r:.3f}")
    else:
        print(f"❌ R-multiple mismatch: expected {expected_r:.3f}, got {actual_r:.3f}")
        return False

    # Test with explicit risk values
    risk_values = [50] * len(pnl_values)  # Fixed $50 risk per trade
    result_fixed = calculator.calculate_sqn(pnl_values, risk_values)

    print(f"\n📊 SQN with fixed $50 risk:")
    print(f"  - SQN: {result_fixed.sqn:.3f}")
    print(f"  - Expectancy: {result_fixed.expectancy:.3f}")
    print(f"  - R-Multiples (first 3): {result_fixed.r_multiples[:3]}")

    # Verify fixed risk calculation
    expected_r_fixed = pnl_values[0] / 50
    actual_r_fixed = result_fixed.r_multiples[0]

    if abs(expected_r_fixed - actual_r_fixed) < 0.01:
        print(f"✅ Fixed risk R-multiple correct: {actual_r_fixed:.3f}")
    else:
        print(f"❌ Fixed risk R-multiple mismatch: expected {expected_r_fixed:.3f}, got {actual_r_fixed:.3f}")
        return False

    return True


def test_quality_score_validation():
    """Test quality score NaN validation."""
    print("\nTesting quality score NaN validation...")

    # Create experience with valid values
    exp_valid = Experience(
        observation=np.random.randn(32, 15),
        action=1,
        policy=np.random.rand(4),
        value=0.5,
        reward=10.0,
        done=False,
        pip_pnl=20.0,
        position_change=True,
        session_expectancy=1.5
    )

    score_valid = exp_valid.calculate_quality_score()
    print(f"Valid experience quality score: {score_valid:.2f}")

    if np.isnan(score_valid) or np.isinf(score_valid):
        print("❌ Valid experience produced NaN/Inf score")
        return False
    else:
        print("✅ Valid experience has finite quality score")

    # Create experience with NaN values
    exp_nan = Experience(
        observation=np.random.randn(32, 15),
        action=0,
        policy=np.random.rand(4),
        value=np.nan,
        reward=np.nan,
        done=False,
        pip_pnl=np.nan,
        position_change=False,
        session_expectancy=0.0
    )

    score_nan = exp_nan.calculate_quality_score()
    print(f"NaN experience quality score: {score_nan:.2f}")

    if score_nan == 0.1:  # Should return minimum valid score
        print("✅ NaN experience returned minimum valid score (0.1)")
    else:
        print(f"❌ NaN experience returned unexpected score: {score_nan}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING NaN HANDLING AND R-MULTIPLE FIXES")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_nan_handling()
    all_passed &= test_r_multiple_calculation()
    all_passed &= test_quality_score_validation()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Fixes are working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Review the fixes")
    print("=" * 60)