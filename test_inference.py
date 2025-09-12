#!/usr/bin/env python3
"""Simple test of inference engine with Episode 13475"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_inference.checkpoint_loader import CheckpointLoader
from swt_features.feature_processor import FeatureProcessor
from experimental_research.swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetwork

def main():
    print("Loading Episode 13475 checkpoint...")
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Load checkpoint directly
    checkpoint_path = "checkpoints/episode_13475.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Create network
    network = SWTStochasticMuZeroNetwork(
        observation_dims={'market': 128, 'position': 9},
        hidden_dim=128,
        action_space_size=4,
        value_support_size=601,
        latent_dim=16
    )
    
    # Load weights
    if 'network_state' in checkpoint:
        network.load_state_dict(checkpoint['network_state'])
    elif 'agent_state' in checkpoint and 'network' in checkpoint['agent_state']:
        network.load_state_dict(checkpoint['agent_state']['network'])
    
    print("Network loaded successfully!")
    
    # Test inference
    market_features = torch.randn(1, 128)
    position_features = torch.randn(1, 9)
    
    with torch.no_grad():
        # Initial inference
        hidden_state = network.representation(market_features, position_features)
        policy, value = network.prediction(hidden_state)
        
        print(f"Hidden state shape: {hidden_state.shape}")
        print(f"Policy shape: {policy.shape}")
        print(f"Value shape: {value.shape}")
        print(f"Policy output: {torch.softmax(policy, dim=-1)}")
        print(f"Predicted action: {torch.argmax(policy, dim=-1).item()}")
    
    print("\n✅ Inference test successful!")

if __name__ == "__main__":
    main()