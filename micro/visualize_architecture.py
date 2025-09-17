#!/usr/bin/env python3
"""
Visualize Micro MuZero Neural Networks Architecture using Netron.
Shows all 5 networks with position features and lagged inputs.
"""

import torch
import torch.nn as nn
import netron
import sys
import os

# Add parent directory to path to import micro modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micro.models.micro_networks import (
    RepresentationNetwork,
    DynamicsNetwork,
    PolicyNetwork,
    ValueNetwork,
    AfterstateNetwork
)

def create_dummy_inputs():
    """Create dummy inputs with proper shapes for all networks."""

    # Main input: (batch=1, lag_window=32, features=15)
    # Features breakdown:
    # - 5 technical indicators (position_in_range_60, momentum, volatility, etc)
    # - 4 cyclical time features (dow_cos, dow_sin, hour_cos, hour_sin)
    # - 6 position state features (side, pips, bars_since, peak, drawdown, accumulated_dd)
    observation = torch.randn(1, 32, 15)

    # Hidden state from representation network
    hidden_state = torch.randn(1, 256)

    # Action (one-hot encoded: Hold, Buy, Sell, Close)
    action = torch.zeros(1, 4)
    action[0, 0] = 1  # Hold action

    # Stochastic latent variable
    stochastic_z = torch.randn(1, 16)

    return observation, hidden_state, action, stochastic_z


def export_networks():
    """Export all 5 MuZero networks to ONNX format."""

    print("Creating micro MuZero networks...")

    # Initialize all networks
    repr_net = RepresentationNetwork(
        input_features=15,  # 5 tech + 4 time + 6 position
        lag_window=32,
        hidden_dim=256,
        tcn_channels=48,
        dropout=0.1
    )

    dynamics_net = DynamicsNetwork(
        hidden_dim=256,
        action_dim=4,
        z_dim=16,
        dropout=0.1
    )

    policy_net = PolicyNetwork(
        hidden_dim=256,
        action_dim=4,
        temperature=1.0,
        dropout=0.1
    )

    value_net = ValueNetwork(
        hidden_dim=256,
        support_size=300,  # [-300, +300] pips
        dropout=0.1
    )

    afterstate_net = AfterstateNetwork(
        hidden_dim=256,
        action_dim=4,
        dropout=0.1
    )

    # Set to eval mode
    repr_net.eval()
    dynamics_net.eval()
    policy_net.eval()
    value_net.eval()
    afterstate_net.eval()

    # Get dummy inputs
    obs, hidden, action, z = create_dummy_inputs()

    print("\nExporting networks to ONNX format...")

    # 1. Export Representation Network
    print("1. Representation Network (32x15 lagged inputs → 256D hidden)")
    print("   Input features:")
    print("   - Timesteps 0-31: Historical lagged data")
    print("   - Features 0-4: Technical indicators")
    print("   - Features 5-8: Cyclical time encoding")
    print("   - Features 9-14: Position state")

    torch.onnx.export(
        repr_net,
        obs,
        "representation_network.onnx",
        input_names=['observation_32x15'],
        output_names=['hidden_state_256'],
        dynamic_axes={'observation_32x15': {0: 'batch'}},
        verbose=False,
        opset_version=11
    )

    # 2. Export Dynamics Network
    print("2. Dynamics Network (hidden + action + stochastic → next_hidden + reward)")
    torch.onnx.export(
        dynamics_net,
        (hidden, action, z),
        "dynamics_network.onnx",
        input_names=['hidden_256', 'action_4', 'stochastic_16'],
        output_names=['next_hidden_256', 'reward'],
        dynamic_axes={
            'hidden_256': {0: 'batch'},
            'action_4': {0: 'batch'},
            'stochastic_16': {0: 'batch'}
        },
        verbose=False,
        opset_version=11
    )

    # 3. Export Policy Network
    print("3. Policy Network (hidden → 4 actions)")
    torch.onnx.export(
        policy_net,
        hidden,
        "policy_network.onnx",
        input_names=['hidden_256'],
        output_names=['action_logits_4'],
        dynamic_axes={'hidden_256': {0: 'batch'}},
        verbose=False,
        opset_version=11
    )

    # 4. Export Value Network
    print("4. Value Network (hidden → 601 value bins)")
    torch.onnx.export(
        value_net,
        hidden,
        "value_network.onnx",
        input_names=['hidden_256'],
        output_names=['value_distribution_601'],
        dynamic_axes={'hidden_256': {0: 'batch'}},
        verbose=False,
        opset_version=11
    )

    # 5. Export Afterstate Network
    print("5. Afterstate Network (hidden + action → afterstate)")
    torch.onnx.export(
        afterstate_net,
        (hidden, action),
        "afterstate_network.onnx",
        input_names=['hidden_256', 'action_4'],
        output_names=['afterstate_256'],
        dynamic_axes={
            'hidden_256': {0: 'batch'},
            'action_4': {0: 'batch'}
        },
        verbose=False,
        opset_version=11
    )

    print("\n✅ All networks exported successfully!")
    print("\nFeature breakdown in Representation Network input (32x15):")
    print("=" * 60)
    print("Temporal dimension (32 timesteps):")
    print("  - t=0: Oldest data (32 bars ago)")
    print("  - t=31: Current/most recent data")
    print("\nFeature dimension (15 features per timestep):")
    print("  [0-4] Technical indicators:")
    print("    0: position_in_range_60")
    print("    1: min_max_scaled_momentum_60")
    print("    2: min_max_scaled_rolling_range")
    print("    3: min_max_scaled_momentum_5")
    print("    4: (reserved)")
    print("  [5-8] Cyclical time features:")
    print("    5: dow_cos_final (day of week cosine)")
    print("    6: dow_sin_final (day of week sine)")
    print("    7: hour_cos_final (hour of day cosine)")
    print("    8: hour_sin_final (hour of day sine)")
    print("  [9-14] Position state features:")
    print("    9: position_side (-1/0/1)")
    print("    10: position_pips (tanh scaled)")
    print("    11: bars_since_entry (tanh scaled)")
    print("    12: pips_from_peak (tanh scaled)")
    print("    13: max_drawdown_pips (tanh scaled)")
    print("    14: accumulated_dd (tanh scaled)")
    print("=" * 60)


def visualize_with_netron():
    """Launch Netron to visualize the networks."""

    print("\n🚀 Launching Netron visualization server...")
    print("This will open in your default browser.")
    print("You can view each network by opening the corresponding .onnx file")
    print("\nPress Ctrl+C to stop the server when done.")

    # Start with the representation network (most interesting)
    netron.start("representation_network.onnx", address=("0.0.0.0", 8080))


if __name__ == "__main__":
    # Export all networks
    export_networks()

    # Launch visualization
    visualize_with_netron()