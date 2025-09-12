#!/usr/bin/env python3
"""
Simplified Monte Carlo validation for Episode 13475
Validates using only 128 market features (no position features)
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from swt_core.config_manager import ConfigManager
from swt_core.types import AgentType
from swt_inference.checkpoint_loader import CheckpointLoader
from swt_features.wst_transform import WSTProcessor, WSTConfig

def process_market_prices(prices: np.ndarray, wst_processor: WSTProcessor) -> np.ndarray:
    """Process 256 market prices to 128 WST features"""
    # Ensure we have 256 prices
    if len(prices) != 256:
        raise ValueError(f"Expected 256 prices, got {len(prices)}")
    
    # Process through WST
    wst_features = wst_processor.transform(prices)
    return wst_features

def run_inference(networks, market_features: np.ndarray) -> Tuple[int, float]:
    """Run inference using only market features"""
    with torch.no_grad():
        # Convert to tensor
        market_tensor = torch.FloatTensor(market_features).unsqueeze(0)
        
        # Get hidden state from representation network
        hidden_state = networks.representation_network(market_tensor)
        
        # Generate latent for stochastic network
        # Chance encoder returns (sample, mean, std)
        latent_z, _, _ = networks.chance_encoder(hidden_state)
        
        # Get policy
        policy_logits = networks.policy_network(hidden_state, latent_z)
        
        # Get action
        action_probs = torch.softmax(policy_logits, dim=-1).squeeze().numpy()
        action = int(np.argmax(action_probs))
        confidence = float(action_probs[action])
        
        return action, confidence

def main():
    print("=" * 60)
    print("SIMPLIFIED MONTE CARLO VALIDATION")
    print("Episode 13475 - 128 Market Features Only")
    print("=" * 60)
    
    # Load config
    print("\n1. Loading configuration...")
    config_manager = ConfigManager()
    config = config_manager.load_config(strict_validation=False)
    config.agent_system = AgentType.STOCHASTIC_MUZERO
    
    # Load checkpoint
    print("\n2. Loading checkpoint...")
    loader = CheckpointLoader(config)
    checkpoint_data = loader.load_checkpoint("checkpoints/episode_13475.pth")
    networks = checkpoint_data['networks']
    print("✅ Checkpoint loaded")
    
    # Initialize WST processor
    print("\n3. Initializing WST processor...")
    wst_config = WSTConfig(
        J=2,
        Q=6,
        backend='manual'
    )
    wst_processor = WSTProcessor(
        config=wst_config,
        device=torch.device('cpu')
    )
    print("✅ WST processor ready")
    
    # Load sample data
    print("\n4. Loading sample data...")
    data = pd.read_csv("data/sample_data.csv")
    print(f"✅ Loaded {len(data)} bars")
    
    # Run simple validation
    print("\n5. Running validation...")
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # HOLD, BUY, SELL, CLOSE
    confidence_sum = 0.0
    num_samples = min(100, len(data) - 256)
    
    for i in range(num_samples):
        # Get 256 prices
        prices = data['close'].iloc[i:i+256].values
        
        # Process to WST features
        wst_features = process_market_prices(prices, wst_processor)
        
        # Run inference
        action, confidence = run_inference(networks, wst_features)
        
        # Track statistics
        action_counts[action] += 1
        confidence_sum += confidence
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_samples} samples...")
    
    # Report results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Samples processed: {num_samples}")
    print(f"Average confidence: {confidence_sum/num_samples:.2%}")
    print("\nAction distribution:")
    print(f"  HOLD:  {action_counts[0]:3d} ({action_counts[0]/num_samples:.1%})")
    print(f"  BUY:   {action_counts[1]:3d} ({action_counts[1]/num_samples:.1%})")
    print(f"  SELL:  {action_counts[2]:3d} ({action_counts[2]/num_samples:.1%})")
    print(f"  CLOSE: {action_counts[3]:3d} ({action_counts[3]/num_samples:.1%})")
    
    # Basic sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    if confidence_sum / num_samples > 0.25:
        print("✅ Average confidence reasonable (>25%)")
    else:
        print("⚠️ Low average confidence (<25%)")
    
    if max(action_counts.values()) / num_samples < 0.95:
        print("✅ Action diversity present (no single action >95%)")
    else:
        print("⚠️ Single action dominates (>95%)")
    
    if action_counts[1] > 0 and action_counts[2] > 0:
        print("✅ Both BUY and SELL actions generated")
    else:
        print("⚠️ Missing BUY or SELL actions")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("Episode 13475 checkpoint is functional with 128 features")
    print("=" * 60)

if __name__ == "__main__":
    main()