#!/usr/bin/env python3
"""Test that validation networks work correctly"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_core.types import AgentType
from swt_inference.checkpoint_loader import CheckpointLoader

def main():
    print("Loading config...")
    config_manager = ConfigManager()
    config = config_manager.load_config(strict_validation=False)
    config.agent_system = AgentType.STOCHASTIC_MUZERO
    
    print("Loading checkpoint...")
    loader = CheckpointLoader(config)
    checkpoint_data = loader.load_checkpoint("checkpoints/episode_13475.pth")
    
    print(f"Checkpoint keys: {checkpoint_data.keys()}")
    print(f"Networks type: {type(checkpoint_data['networks'])}")
    
    # Test inference
    networks = checkpoint_data['networks']
    
    with torch.no_grad():
        # Create dummy inputs - network expects only 128 dim market features
        market_features = torch.randn(1, 128)
        position_features = torch.randn(1, 9)
        
        print("\nTesting representation network...")
        hidden_state = networks.representation_network(market_features)
        print(f"Hidden state shape: {hidden_state.shape}")
        
        print("\nTesting policy network...")
        policy_logits = networks.policy_network(hidden_state)
        print(f"Policy logits shape: {policy_logits.shape}")
        
        print("\nTesting value network...")
        value = networks.value_network(hidden_state)
        print(f"Value shape: {value.shape}")
        
        # Get action
        action_probs = torch.softmax(policy_logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1).item()
        confidence = action_probs[0, action].item()
        
        print(f"\nSelected action: {action} with confidence: {confidence:.2%}")
        
    print("\n✅ All networks working correctly!")

if __name__ == "__main__":
    main()