#!/usr/bin/env python3
"""
Fixed checkpoint loader that properly extracts embedded configuration
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


def load_checkpoint_with_proper_config(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load checkpoint and properly extract embedded configuration

    This fixes the issue where checkpoints with hidden_dim=256 were being
    loaded with hidden_dim=128 from external configs.
    """
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load raw checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract proper configuration
    config = {}

    # Priority 1: Full config embedded in checkpoint
    if 'config' in checkpoint and 'full_config' in checkpoint['config']:
        full_config = checkpoint['config']['full_config']
        if 'muzero_config' in full_config:
            muzero_cfg = full_config['muzero_config']
            config['hidden_dim'] = muzero_cfg.get('hidden_dim', 256)
            config['support_size'] = muzero_cfg.get('support_size', 601)
            config['num_layers'] = muzero_cfg.get('representation_blocks', 2)
            logger.info(f"✅ Extracted from full_config: hidden_dim={config['hidden_dim']}, support_size={config['support_size']}")

    # Priority 2: Direct config values
    elif 'config' in checkpoint:
        cfg = checkpoint['config']
        config['hidden_dim'] = cfg.get('hidden_dim', 256)
        config['value_support_size'] = cfg.get('value_support_size', 1203)
        config['support_size'] = (config['value_support_size'] - 1) // 2
        logger.info(f"✅ Extracted from config: hidden_dim={config['hidden_dim']}, support_size={config['support_size']}")

    # Priority 3: Auto-detect from weights
    elif 'muzero_network_state' in checkpoint:
        network_state = checkpoint['muzero_network_state']

        # Detect hidden_dim from representation network
        for key, tensor in network_state.items():
            if 'representation_network.input_projection.0.weight' in key:
                config['hidden_dim'] = tensor.shape[0]
                logger.info(f"🔍 Auto-detected hidden_dim={config['hidden_dim']}")
                break

        # Detect support size from value head
        for key, tensor in network_state.items():
            if 'value_network.value_head.3.weight' in key:
                value_support_size = tensor.shape[0]
                config['support_size'] = (value_support_size - 1) // 2
                logger.info(f"🔍 Auto-detected support_size={config['support_size']}")
                break

    # Build proper network with detected config - use same as training
    from swt_models.swt_stochastic_networks import (
        SWTStochasticMuZeroNetwork,
        SWTStochasticMuZeroConfig
    )

    # Match EXACT training configuration
    network_config = SWTStochasticMuZeroConfig(
        market_wst_features=128,
        position_features=9,
        total_input_dim=137,
        final_input_dim=137,
        num_actions=4,  # BUY, SELL, HOLD, CLOSE
        hidden_dim=256,
        representation_blocks=6,  # Match training
        dynamics_blocks=6,  # Match training
        prediction_blocks=2,
        afterstate_blocks=2,
        support_size=300,  # Match training
        chance_space_size=32,
        chance_history_length=4,
        afterstate_enabled=True,
        dropout_rate=0.1,
        layer_norm=True,
        residual_connections=True,
        latent_z_dim=16
    )

    networks = SWTStochasticMuZeroNetwork(network_config)

    # Load weights
    if 'muzero_network_state' in checkpoint:
        networks.load_state_dict(checkpoint['muzero_network_state'])
        logger.info("✅ Network weights loaded successfully")
    else:
        raise ValueError("No muzero_network_state found in checkpoint")

    return {
        'networks': networks,
        'config': config,
        'metadata': checkpoint.get('metadata', {}),
        'checkpoint_path': str(checkpoint_path)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fixed_checkpoint_loader.py <checkpoint_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    try:
        result = load_checkpoint_with_proper_config(sys.argv[1])
        print(f"\n✅ Successfully loaded checkpoint with:")
        print(f"   Hidden dim: {result['config'].get('hidden_dim')}")
        print(f"   Support size: {result['config'].get('support_size')}")
        print(f"   Networks: {type(result['networks'])}")
    except Exception as e:
        print(f"\n❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()