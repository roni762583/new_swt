#!/usr/bin/env python3
"""
Test script to verify the complete episode collection system works.
Tests session index calculation, episode running, and parallel collection.
"""

import sys
sys.path.append('/workspace')

import logging
import time
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_session_indices():
    """Test session index calculation."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Session Index Calculation")
    logger.info("="*60)

    try:
        from micro.utils.session_index_calculator import SessionIndexCalculator

        calculator = SessionIndexCalculator()
        indices = calculator.get_or_calculate_indices()

        logger.info(f"✅ Successfully loaded/calculated indices")
        logger.info(f"   Training sessions: {len(indices['train'])}")
        logger.info(f"   Validation sessions: {len(indices['val'])}")
        logger.info(f"   Test sessions: {len(indices['test'])}")

        # Verify indices are valid
        if len(indices['train']) > 0:
            sample_idx = indices['train'][0]
            logger.info(f"   Sample train index: {sample_idx}")
            logger.info(f"✅ Session indices test PASSED")
            return True
        else:
            logger.error("❌ No training sessions found!")
            return False

    except Exception as e:
        logger.error(f"❌ Session indices test FAILED: {e}")
        return False


def test_single_episode():
    """Test running a single episode."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Single Episode Collection")
    logger.info("="*60)

    try:
        import torch
        from micro.models.micro_networks import MicroStochasticMuZero
        from micro.training.stochastic_mcts import StochasticMCTS
        from micro.training.episode_runner import EpisodeRunner

        # Create model
        model = MicroStochasticMuZero(
            input_features=15,
            lag_window=32,
            hidden_dim=256,
            action_dim=4,
            num_outcomes=3,
            support_size=300
        )
        model.eval()

        # Create MCTS
        mcts = StochasticMCTS(
            model=model,
            num_simulations=5,  # Few simulations for testing
            discount=0.997,
            depth_limit=2,
            dirichlet_alpha=1.0,
            exploration_fraction=0.5
        )

        # Create episode runner
        runner = EpisodeRunner(
            model=model,
            mcts=mcts,
            db_path="/workspace/data/micro_features.duckdb",
            session_indices_path="/workspace/micro/cache/valid_session_indices.pkl",
            device="cpu"
        )

        # Run single episode
        logger.info("Running single episode...")
        start_time = time.time()

        episode = runner.run_episode(
            split='train',
            session_idx=0,
            temperature=1.0,
            add_noise=True
        )

        elapsed = time.time() - start_time

        logger.info(f"✅ Episode completed in {elapsed:.1f} seconds")
        logger.info(f"   Experiences collected: {len(episode.experiences)}")
        logger.info(f"   Total reward: {episode.total_reward:.2f}")
        logger.info(f"   Trades: {episode.num_trades}")
        logger.info(f"   Expectancy: {episode.expectancy:.4f} pips")
        logger.info(f"   Win rate: {episode.winning_trades}/{episode.num_trades}")

        # Verify episode has 360 experiences
        if len(episode.experiences) == 360:
            logger.info(f"✅ Episode has correct length (360)")
        else:
            logger.error(f"❌ Episode has wrong length: {len(episode.experiences)}")
            return False

        # Check action distribution
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for exp in episode.experiences:
            action_counts[exp.action] += 1

        logger.info(f"   Action distribution:")
        logger.info(f"     HOLD: {action_counts[0]} ({action_counts[0]/360:.1%})")
        logger.info(f"     BUY: {action_counts[1]} ({action_counts[1]/360:.1%})")
        logger.info(f"     SELL: {action_counts[2]} ({action_counts[2]/360:.1%})")
        logger.info(f"     CLOSE: {action_counts[3]} ({action_counts[3]/360:.1%})")

        logger.info(f"✅ Single episode test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Single episode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_collection():
    """Test parallel episode collection."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Parallel Episode Collection")
    logger.info("="*60)

    try:
        from micro.training.parallel_episode_collector import ParallelEpisodeCollector

        model_config = {
            'input_features': 15,
            'lag_window': 32,
            'hidden_dim': 256,
            'action_dim': 4,
            'num_outcomes': 3,
            'support_size': 300
        }

        mcts_config = {
            'num_simulations': 5,  # Few for testing
            'discount': 0.997,
            'depth_limit': 2,
            'dirichlet_alpha': 1.0,
            'exploration_fraction': 0.5
        }

        # Create temp model file
        import torch
        from micro.models.micro_networks import MicroStochasticMuZero

        model = MicroStochasticMuZero(**model_config)
        temp_path = "/tmp/test_model.pth"
        torch.save({'model_state': model.get_weights()}, temp_path)

        # Create collector
        collector = ParallelEpisodeCollector(
            model_path=temp_path,
            model_config=model_config,
            mcts_config=mcts_config,
            num_workers=2,  # Just 2 workers for testing
            db_path="/workspace/data/micro_features.duckdb",
            session_indices_path="/workspace/micro/cache/valid_session_indices.pkl"
        )

        # Start workers
        logger.info("Starting workers...")
        collector.start()

        # Collect episodes
        logger.info("Collecting 4 episodes...")
        start_time = time.time()

        episodes = collector.collect_episodes(
            num_episodes=4,
            split='train',
            temperature=1.0,
            add_noise=True,
            timeout=120.0
        )

        elapsed = time.time() - start_time

        logger.info(f"✅ Collected {len(episodes)} episodes in {elapsed:.1f} seconds")
        logger.info(f"   Average time per episode: {elapsed/len(episodes):.1f}s")

        # Verify episodes
        total_experiences = 0
        total_trades = 0
        for i, episode in enumerate(episodes):
            total_experiences += len(episode.experiences)
            total_trades += episode.num_trades
            logger.info(f"   Episode {i}: {len(episode.experiences)} exp, "
                       f"{episode.num_trades} trades, "
                       f"expectancy {episode.expectancy:.2f}")

        logger.info(f"   Total experiences: {total_experiences}")
        logger.info(f"   Total trades: {total_trades}")

        # Get stats
        stats = collector.get_stats()
        logger.info(f"   Collection stats: {stats}")

        # Stop workers
        logger.info("Stopping workers...")
        collector.stop()

        if len(episodes) == 4 and total_experiences == 4 * 360:
            logger.info(f"✅ Parallel collection test PASSED")
            return True
        else:
            logger.error(f"❌ Wrong number of episodes or experiences")
            return False

    except Exception as e:
        logger.error(f"❌ Parallel collection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("MICRO MUZERO EPISODE COLLECTION TEST SUITE")
    logger.info("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Session indices
    if test_session_indices():
        tests_passed += 1
    else:
        tests_failed += 1
        logger.warning("Skipping remaining tests due to session index failure")
        return

    # Test 2: Single episode
    if test_single_episode():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Parallel collection
    if test_parallel_collection():
        tests_passed += 1
    else:
        tests_failed += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests Passed: {tests_passed}")
    logger.info(f"Tests Failed: {tests_failed}")

    if tests_failed == 0:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("\nThe episode collection system is working correctly.")
        logger.info("You can now use train_micro_muzero_fixed.py for proper training.")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Please fix the issues before training.")


if __name__ == "__main__":
    main()