#!/usr/bin/env python3
"""Debug network loading issue"""

import sys
import torch
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
    
    networks = checkpoint_data['networks']
    print(f"Networks type: {type(networks)}")
    print(f"Networks class: {networks.__class__.__name__}")
    
    # Check if it has __len__
    try:
        print(f"Networks length: {len(networks)}")
    except TypeError as e:
        print(f"Cannot get length: {e}")
    
    # Check attributes
    print(f"Networks attributes: {dir(networks)[:10]}...")  # First 10 attributes
    
    # Check if it's a single network or multiple
    if hasattr(networks, 'representation_network'):
        print("Has representation_network attribute")
    if hasattr(networks, 'policy_network'):
        print("Has policy_network attribute")
    if hasattr(networks, 'value_network'):
        print("Has value_network attribute")

if __name__ == "__main__":
    main()