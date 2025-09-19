#!/usr/bin/env python3
"""
Test script to validate the implementation fixes:
1. BalancedReplayBuffer with quota-based eviction
2. Rolling stdev-based outcome thresholds
3. MCTS with 3x3 configuration
4. AMDDP1 reward system
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
import torch
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test 1: BalancedReplayBuffer
def test_balanced_replay_buffer():
    """Test the new balanced replay buffer with quota-based eviction."""
    logger.info("Testing BalancedReplayBuffer...")

    from micro.training.train_micro_muzero import BalancedReplayBuffer, Experience

    # Create buffer
    buffer = BalancedReplayBuffer(capacity=100, trade_quota=0.3)

    # Add experiences
    for i in range(150):
        exp = Experience(
            observation=np.random.randn(32, 15),
            action=0 if i % 3 == 0 else np.random.choice([1, 2, 3]),  # Mix of holds and trades
            reward=np.random.randn(),
            policy=np.random.dirichlet([1, 1, 1, 1]),
            value=np.random.randn(),
            done=False,
            market_outcome=np.random.choice([0, 1, 2]),
            outcome_probs=np.random.dirichlet([1, 1, 1])
        )
        buffer.add(exp)

    # Check buffer stats
    stats = buffer.get_stats()
    logger.info(f"Buffer size: {stats['buffer_size']}")
    logger.info(f"Trade ratio: {stats['trade_ratio']:.2%}")
    logger.info(f"Quota satisfied: {stats['quota_satisfied']}")

    # Test sampling
    batch = buffer.sample(32)
    logger.info(f"Sampled batch size: {len(batch)}")

    # Check trade diversity in batch
    trade_count = sum(1 for exp in batch if exp.action in [1, 2, 3])
    batch_trade_ratio = trade_count / len(batch)
    logger.info(f"Batch trade ratio: {batch_trade_ratio:.2%}")

    assert stats['buffer_size'] == 100, "Buffer should respect capacity"
    assert stats['trade_ratio'] >= 0.3, "Buffer should maintain trade quota"
    assert batch_trade_ratio >= 0.25, "Batch should have reasonable trade diversity"

    logger.info("✅ BalancedReplayBuffer test passed!\n")


# Test 2: Market Outcome Calculator with Rolling Stdev
def test_market_outcome_calculator():
    """Test rolling stdev-based outcome classification."""
    logger.info("Testing MarketOutcomeCalculator...")

    from micro.utils.market_outcome_calculator import MarketOutcomeCalculator

    calculator = MarketOutcomeCalculator(
        window_size=20,
        threshold_multiplier=0.5
    )

    # Simulate price series
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(100) * 0.01)

    # Feed prices and calculate outcomes
    outcomes = []
    for i in range(len(prices) - 1):
        calculator.add_price(prices[i])

        if i >= 20:  # Need enough history
            outcome = calculator.calculate_outcome(prices[i], prices[i + 1])
            outcomes.append(outcome)

    # Check outcome distribution
    unique, counts = np.unique(outcomes, return_counts=True)
    outcome_dist = dict(zip(unique, counts))

    logger.info(f"Outcome distribution over {len(outcomes)} steps:")
    logger.info(f"UP (0): {outcome_dist.get(0, 0)} ({outcome_dist.get(0, 0)/len(outcomes)*100:.1f}%)")
    logger.info(f"NEUTRAL (1): {outcome_dist.get(1, 0)} ({outcome_dist.get(1, 0)/len(outcomes)*100:.1f}%)")
    logger.info(f"DOWN (2): {outcome_dist.get(2, 0)} ({outcome_dist.get(2, 0)/len(outcomes)*100:.1f}%)")

    # Test rolling stdev calculation
    rolling_stdev = calculator.get_rolling_stdev()
    logger.info(f"Current rolling stdev: {rolling_stdev:.6f}")

    assert rolling_stdev is not None, "Should calculate rolling stdev"
    assert 1 in outcome_dist, "Should have NEUTRAL outcomes"
    assert len(outcome_dist) > 1, "Should have diverse outcomes"

    logger.info("✅ MarketOutcomeCalculator test passed!\n")


# Test 3: MCTS Configuration
def test_mcts_configuration():
    """Test MCTS with 3x3 configuration."""
    logger.info("Testing MCTS 3x3 configuration...")

    from micro.training.stochastic_mcts import StochasticMCTS
    from micro.models.micro_networks import MicroStochasticMuZero

    # Create model
    model = MicroStochasticMuZero(
        input_features=15,
        lag_window=32,
        hidden_dim=256,
        action_dim=4,
        num_outcomes=3,
        support_size=300
    )

    # Create MCTS with 3x3 configuration
    mcts = StochasticMCTS(
        model=model,
        num_actions=4,
        num_outcomes=3,
        discount=0.997,
        num_simulations=10,  # Reduced for faster training
        depth_limit=3,  # Fixed at 3
        dirichlet_alpha=1.0,
        exploration_fraction=0.5
    )

    # Test configuration
    logger.info(f"MCTS simulations: {mcts.num_simulations}")
    logger.info(f"MCTS depth limit: {mcts.depth_limit}")
    logger.info(f"MCTS num outcomes: {mcts.num_outcomes}")

    # Run a test simulation
    observation = torch.randn(1, 32, 15)
    result = mcts.run(observation, temperature=1.0, add_noise=True)

    logger.info(f"Policy shape: {result['policy'].shape}")
    logger.info(f"Selected action: {result['action']}")
    logger.info(f"Root value: {result['value']:.3f}")

    assert mcts.depth_limit == 3, "Depth should be fixed at 3"
    assert mcts.num_simulations == 10, "Simulations should be 10"
    assert mcts.num_outcomes == 3, "Should have 3 outcomes"
    assert result['policy'].shape == (4,), "Policy should have 4 actions"

    logger.info("✅ MCTS configuration test passed!\n")


# Test 4: Training Configuration
def test_training_configuration():
    """Test training configuration matches README specs."""
    logger.info("Testing training configuration...")

    from micro.training.train_micro_muzero import TrainingConfig

    config = TrainingConfig()

    logger.info(f"Buffer size: {config.buffer_size}")
    logger.info(f"Min buffer size: {config.min_buffer_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Num simulations: {config.num_simulations}")
    logger.info(f"Depth limit: {config.depth_limit}")
    logger.info(f"Temperature range: {config.initial_temperature} → {config.final_temperature}")

    assert config.buffer_size == 10000, "Buffer size should be 10000"
    assert config.min_buffer_size == 100, "Min buffer should be 100"
    assert config.learning_rate == 0.002, "Learning rate should be fixed at 0.002"
    assert config.num_simulations == 10, "Simulations should be 10"
    assert config.depth_limit == 3, "Depth should be 3"

    logger.info("✅ Training configuration test passed!\n")


def main():
    """Run all tests."""
    logger.info("=" * 50)
    logger.info("Running Implementation Fix Tests")
    logger.info("=" * 50 + "\n")

    try:
        test_balanced_replay_buffer()
        test_market_outcome_calculator()
        test_mcts_configuration()
        test_training_configuration()

        logger.info("=" * 50)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 50)

    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()