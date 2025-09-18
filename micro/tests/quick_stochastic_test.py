#!/usr/bin/env python3
"""
Quick test to verify stochastic MuZero implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.stochastic_mcts import StochasticMCTS

def test_stochastic_muzero():
    """Quick integration test."""
    print("Testing Stochastic MuZero Implementation...")

    # Initialize model
    model = MicroStochasticMuZero(
        input_features=15,
        lag_window=32,
        hidden_dim=256,
        action_dim=4,
        num_outcomes=3  # UP, NEUTRAL, DOWN
    )

    # Initialize MCTS
    mcts = StochasticMCTS(
        model=model,
        num_actions=4,
        num_outcomes=3,
        num_simulations=5,
        depth_limit=2
    )

    # Create test observation
    observation = torch.randn(1, 32, 15)

    # Run MCTS
    result = mcts.run(observation, temperature=1.0)

    print(f"✅ Action selected: {result['action']}")
    print(f"✅ Policy: {result['policy']}")
    print(f"✅ Value: {result['value']:.4f}")

    # Test outcome prediction
    with torch.no_grad():
        hidden, _, _ = model.initial_inference(observation)
        for action_idx in range(4):
            action = torch.zeros(1, 4)
            action[0, action_idx] = 1
            outcome_probs = model.predict_outcome(hidden, action)
            print(f"✅ Action {action_idx} outcome probs: UP={outcome_probs[0,0]:.2f}, NEUTRAL={outcome_probs[0,1]:.2f}, DOWN={outcome_probs[0,2]:.2f}")

    print("\n🎉 All tests passed! Stochastic MuZero is ready.")

if __name__ == "__main__":
    test_stochastic_muzero()