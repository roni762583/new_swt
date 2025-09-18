#!/usr/bin/env python3
"""
Comprehensive tests for Stochastic MuZero components.

Tests:
1. Network output shapes and ranges
2. Information flow between components
3. Outcome probability distributions
4. MCTS tree structure with chance nodes
5. Market outcome calculations
6. End-to-end integration
"""

import torch
import numpy as np
from typing import Tuple
import logging

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from micro.models.micro_networks import (
    MicroStochasticMuZero,
    OutcomeProbabilityNetwork,
    DynamicsNetwork
)
from micro.training.stochastic_mcts import (
    StochasticMCTS,
    DecisionNode,
    ChanceNode,
    MarketOutcome
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestNetworkComponents:
    """Test individual network components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.hidden_dim = 256
        self.action_dim = 4
        self.num_outcomes = 3
        self.lag_window = 32
        self.input_features = 15

    def test_outcome_probability_network(self):
        """Test OutcomeProbabilityNetwork outputs valid probabilities."""
        network = OutcomeProbabilityNetwork(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            num_outcomes=self.num_outcomes
        ).to(self.device)

        # Create test inputs
        hidden = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
        action = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        action[:, 0] = 1  # All HOLD actions

        # Forward pass
        probs = network(hidden, action)

        # Verify shape
        assert probs.shape == (self.batch_size, self.num_outcomes), \
            f"Expected shape {(self.batch_size, self.num_outcomes)}, got {probs.shape}"

        # Verify probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            f"Probabilities don't sum to 1: {prob_sums}"

        # Verify all probabilities are in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), \
            f"Probabilities outside [0, 1] range: min={probs.min()}, max={probs.max()}"

        logger.info("✅ OutcomeProbabilityNetwork test passed")

    def test_dynamics_network_with_outcomes(self):
        """Test DynamicsNetwork with market outcomes."""
        network = DynamicsNetwork(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            outcome_dim=self.num_outcomes
        ).to(self.device)

        # Create test inputs
        hidden = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
        action = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        action[:, 1] = 1  # All BUY actions

        # Test with different outcomes
        outcomes = [
            torch.tensor([[1, 0, 0]], dtype=torch.float32),  # UP
            torch.tensor([[0, 1, 0]], dtype=torch.float32),  # NEUTRAL
            torch.tensor([[0, 0, 1]], dtype=torch.float32),  # DOWN
        ]

        for i, outcome in enumerate(outcomes):
            outcome_batch = outcome.repeat(self.batch_size, 1).to(self.device)
            next_hidden, reward = network(hidden, action, outcome_batch)

            # Verify shapes
            assert next_hidden.shape == (self.batch_size, self.hidden_dim), \
                f"Expected hidden shape {(self.batch_size, self.hidden_dim)}, got {next_hidden.shape}"
            assert reward.shape == (self.batch_size, 1), \
                f"Expected reward shape {(self.batch_size, 1)}, got {reward.shape}"

            # Verify next_hidden is not NaN
            assert not torch.isnan(next_hidden).any(), \
                f"NaN in next_hidden for outcome {i}"
            assert not torch.isnan(reward).any(), \
                f"NaN in reward for outcome {i}"

        logger.info("✅ DynamicsNetwork with outcomes test passed")

    def test_full_model_integration(self):
        """Test full MicroStochasticMuZero model."""
        model = MicroStochasticMuZero(
            input_features=self.input_features,
            lag_window=self.lag_window,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            num_outcomes=self.num_outcomes
        ).to(self.device)

        # Create test observation
        observation = torch.randn(
            self.batch_size, self.lag_window, self.input_features
        ).to(self.device)

        # Initial inference
        hidden, policy_logits, value_probs = model.initial_inference(observation)

        # Verify shapes
        assert hidden.shape == (self.batch_size, self.hidden_dim)
        assert policy_logits.shape == (self.batch_size, self.action_dim)
        assert value_probs.shape[0] == self.batch_size

        # Test outcome prediction
        action = torch.zeros(self.batch_size, self.action_dim).to(self.device)
        action[:, 2] = 1  # All SELL actions

        outcome_probs = model.predict_outcome(hidden, action)
        assert outcome_probs.shape == (self.batch_size, self.num_outcomes)
        assert torch.allclose(outcome_probs.sum(dim=-1), torch.ones(self.batch_size).to(self.device))

        # Test recurrent inference
        next_hidden, reward, next_policy, next_value = model.recurrent_inference(
            hidden, action, outcome_probs
        )

        assert next_hidden.shape == (self.batch_size, self.hidden_dim)
        assert reward.shape == (self.batch_size, 1)
        assert next_policy.shape == (self.batch_size, self.action_dim)

        logger.info("✅ Full model integration test passed")


class TestStochasticMCTS:
    """Test Stochastic MCTS implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MicroStochasticMuZero(
            input_features=15,
            lag_window=32,
            hidden_dim=256,
            action_dim=4,
            num_outcomes=3
        ).to(self.device)

        self.mcts = StochasticMCTS(
            model=self.model,
            num_actions=4,
            num_outcomes=3,
            num_simulations=10,
            depth_limit=2
        )

    def test_mcts_tree_structure(self):
        """Test MCTS builds proper tree with chance nodes."""
        observation = torch.randn(1, 32, 15).to(self.device)

        # Run MCTS
        result = self.mcts.run(observation, temperature=1.0, add_noise=True)

        # Verify result structure
        assert 'action' in result
        assert 'policy' in result
        assert 'value' in result
        assert 'tree_stats' in result

        # Verify policy is valid probability distribution
        policy = result['policy']
        assert len(policy) == 4
        assert np.allclose(policy.sum(), 1.0)
        assert (policy >= 0).all() and (policy <= 1).all()

        # Verify action is valid
        action = result['action']
        assert 0 <= action < 4

        logger.info(f"✅ MCTS tree structure test passed")
        logger.info(f"   Policy: {policy}")
        logger.info(f"   Action: {action}")
        logger.info(f"   Value: {result['value']:.4f}")

    def test_decision_node_expansion(self):
        """Test DecisionNode expansion with chance nodes."""
        hidden = torch.randn(1, 256).to(self.device)

        node = DecisionNode(
            prior=1.0,
            hidden_state=hidden[0],
            reward=0.0
        )

        # Create dummy outcome probabilities
        priors = np.array([0.25, 0.25, 0.25, 0.25])
        outcome_probs = {
            0: np.array([0.2, 0.6, 0.2]),  # HOLD: mostly neutral
            1: np.array([0.6, 0.3, 0.1]),  # BUY: expect up
            2: np.array([0.1, 0.3, 0.6]),  # SELL: expect down
            3: np.array([0.33, 0.34, 0.33])  # CLOSE: uncertain
        }

        node.expand(4, priors, outcome_probs)

        # Verify children are ChanceNodes
        assert len(node.children) == 4
        for action, child in node.children.items():
            assert isinstance(child, ChanceNode)
            assert np.allclose(child.outcome_probabilities.sum(), 1.0)

        logger.info("✅ DecisionNode expansion test passed")

    def test_chance_node_expansion(self):
        """Test ChanceNode expansion with decision nodes."""
        chance_node = ChanceNode(
            prior=0.25,
            action=1,  # BUY
            outcome_probabilities=np.array([0.6, 0.3, 0.1])
        )

        # Create dummy hidden states and rewards
        hidden_states = [
            torch.randn(256).to(self.device) for _ in range(3)
        ]
        rewards = [1.0, 0.0, -1.0]  # UP: positive, NEUTRAL: zero, DOWN: negative

        chance_node.expand(hidden_states, rewards)

        # Verify children are DecisionNodes
        assert len(chance_node.children) == 3
        for outcome, child in chance_node.children.items():
            assert isinstance(child, DecisionNode)
            assert child.reward == rewards[outcome]

        logger.info("✅ ChanceNode expansion test passed")


class TestMarketOutcomeCalculation:
    """Test market outcome calculations based on rolling stdev."""

    def test_outcome_classification(self):
        """Test classification of price movements into outcomes."""
        # Simulate price series
        prices = np.array([100.0, 100.5, 101.0, 100.8, 99.5, 100.2, 101.5, 100.0])

        # Calculate rolling stdev (window=5)
        window = 5
        rolling_stdevs = []
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            rolling_stdevs.append(np.std(window_prices))

        # Classify price changes
        outcomes = []
        for i in range(len(rolling_stdevs)):
            price_change = prices[window + i] - prices[window + i - 1]
            stdev = rolling_stdevs[i]

            if price_change > 0.5 * stdev:
                outcome = MarketOutcome.UP
            elif price_change < -0.5 * stdev:
                outcome = MarketOutcome.DOWN
            else:
                outcome = MarketOutcome.NEUTRAL

            outcomes.append(outcome)

        # Verify we get diverse outcomes
        unique_outcomes = set(outcomes)
        assert len(unique_outcomes) > 1, \
            f"Expected multiple outcome types, got only: {unique_outcomes}"

        logger.info(f"✅ Market outcome classification test passed")
        logger.info(f"   Outcomes: {outcomes}")

    def test_outcome_to_onehot(self):
        """Test conversion of outcome to one-hot encoding."""
        device = torch.device("cpu")

        # Test each outcome
        for outcome_idx, expected in [
            (MarketOutcome.UP, [1, 0, 0]),
            (MarketOutcome.NEUTRAL, [0, 1, 0]),
            (MarketOutcome.DOWN, [0, 0, 1])
        ]:
            onehot = MarketOutcome.to_onehot(outcome_idx, device)
            assert torch.allclose(onehot, torch.tensor(expected, dtype=torch.float32))

        logger.info("✅ Outcome to one-hot conversion test passed")


class TestEndToEndIntegration:
    """Test end-to-end integration of all components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MicroStochasticMuZero(
            input_features=15,
            lag_window=32,
            hidden_dim=256,
            action_dim=4,
            num_outcomes=3
        ).to(self.device)

        self.mcts = StochasticMCTS(
            model=self.model,
            num_actions=4,
            num_outcomes=3,
            num_simulations=5,
            depth_limit=2
        )

    def test_full_planning_cycle(self):
        """Test complete planning cycle with stochastic outcomes."""
        # Create observation
        observation = torch.randn(1, 32, 15).to(self.device)

        # Run MCTS planning
        result = self.mcts.run(observation, temperature=0.5)

        # Extract action and verify it's not always HOLD
        actions_taken = []
        for _ in range(10):
            result = self.mcts.run(
                observation + torch.randn_like(observation) * 0.1,  # Add noise
                temperature=1.0
            )
            actions_taken.append(result['action'])

        # Should have some diversity in actions
        unique_actions = set(actions_taken)
        assert len(unique_actions) > 1, \
            f"Expected diverse actions, got only: {unique_actions}"

        logger.info(f"✅ Full planning cycle test passed")
        logger.info(f"   Actions taken: {actions_taken}")
        logger.info(f"   Unique actions: {unique_actions}")

    def test_information_flow(self):
        """Test information flows correctly through all components."""
        # Initial observation
        observation = torch.randn(1, 32, 15).to(self.device)

        # 1. Initial inference
        with torch.no_grad():
            hidden, policy_logits, value_probs = self.model.initial_inference(observation)

        assert not torch.isnan(hidden).any(), "NaN in initial hidden state"

        # 2. Predict outcomes for each action
        for action_idx in range(4):
            action = torch.zeros(1, 4).to(self.device)
            action[0, action_idx] = 1

            with torch.no_grad():
                outcome_probs = self.model.predict_outcome(hidden, action)

            assert not torch.isnan(outcome_probs).any(), \
                f"NaN in outcome_probs for action {action_idx}"
            assert torch.allclose(outcome_probs.sum(), torch.tensor(1.0)), \
                f"Outcome probs don't sum to 1 for action {action_idx}"

            # 3. Recurrent inference with each outcome
            for outcome_idx in range(3):
                outcome = torch.zeros(1, 3).to(self.device)
                outcome[0, outcome_idx] = 1

                with torch.no_grad():
                    next_hidden, reward, next_policy, next_value = \
                        self.model.recurrent_inference(hidden, action, outcome)

                assert not torch.isnan(next_hidden).any(), \
                    f"NaN in next_hidden for action {action_idx}, outcome {outcome_idx}"
                assert not torch.isnan(reward).any(), \
                    f"NaN in reward for action {action_idx}, outcome {outcome_idx}"

        logger.info("✅ Information flow test passed")


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestNetworkComponents,
        TestStochasticMCTS,
        TestMarketOutcomeCalculation,
        TestEndToEndIntegration
    ]

    logger.info("="*60)
    logger.info("RUNNING STOCHASTIC MUZERO COMPONENT TESTS")
    logger.info("="*60)

    all_passed = True
    for test_class in test_classes:
        logger.info(f"\nTesting {test_class.__name__}...")
        test_instance = test_class()

        # Run all test methods
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                try:
                    # Setup if exists
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()

                    # Run test
                    method = getattr(test_instance, method_name)
                    method()
                    logger.info(f"  ✅ {method_name}")

                except Exception as e:
                    logger.error(f"  ❌ {method_name}: {e}")
                    all_passed = False

    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.error("❌ SOME TESTS FAILED")
    logger.info("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)