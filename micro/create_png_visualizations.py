#!/usr/bin/env python3
"""
Create PNG visualizations of Micro MuZero architecture.
Shows position features and lagged inputs clearly.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micro.models.micro_networks import (
    RepresentationNetwork,
    DynamicsNetwork,
    PolicyNetwork,
    ValueNetwork,
    AfterstateNetwork
)

def create_network_diagram():
    """Create a comprehensive network architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Colors
    input_color = '#E8F4FD'  # Light blue
    tcn_color = '#FFE6CC'    # Light orange
    repr_color = '#D4E6F1'   # Light blue
    network_color = '#D5F3D0' # Light green
    output_color = '#F8D7DA'  # Light red

    # Title
    ax.text(8, 11.5, 'Micro Stochastic MuZero Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # Input Data (32x15)
    input_box = FancyBboxPatch((0.5, 9), 3, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=input_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 9.75, 'Input Data\n(32×15)', fontsize=12, fontweight='bold', ha='center', va='center')

    # Feature breakdown
    features_text = """Features per timestep (15):
    [0-4] Technical indicators
    [5-8] Cyclical time
    [9-14] Position state"""
    ax.text(0.5, 8.2, features_text, fontsize=9, va='top')

    # Lag breakdown
    lag_text = """Temporal window (32):
    t=0: 32 bars ago
    t=31: Current"""
    ax.text(0.5, 6.8, lag_text, fontsize=9, va='top')

    # TCN Encoder
    tcn_box = FancyBboxPatch((5, 9), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=tcn_color,
                             edgecolor='black', linewidth=2)
    ax.add_patch(tcn_box)
    ax.text(6.5, 9.75, 'TCN Encoder\n15→48D\nDilations: [1,2,4]',
            fontsize=11, fontweight='bold', ha='center', va='center')

    # Representation Network
    repr_box = FancyBboxPatch((9.5, 9), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=repr_color,
                              edgecolor='black', linewidth=2)
    ax.add_patch(repr_box)
    ax.text(11, 9.75, 'Representation\nAttention Pool\n→256D Hidden',
            fontsize=11, fontweight='bold', ha='center', va='center')

    # Hidden State
    hidden_box = FancyBboxPatch((13.5, 9), 2, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#F0F0F0',
                                edgecolor='black', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(14.5, 9.75, 'Hidden\nState\n(256D)',
            fontsize=11, fontweight='bold', ha='center', va='center')

    # Network branches
    networks = [
        ('Dynamics\n(+action+z)\n→next_h+reward', 2, 6, network_color),
        ('Policy\n→4 actions\n[H,B,S,C]', 5.5, 6, network_color),
        ('Value\n→601 bins\n[-300,+300]', 9, 6, network_color),
        ('Afterstate\n(+action)\n→afterstate', 12.5, 6, network_color)
    ]

    for name, x, y, color in networks:
        box = FancyBboxPatch((x-0.75, y-0.75), 1.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center')

    # Outputs
    outputs = [
        ('Next State\n(256D)', 2, 3.5, output_color),
        ('Reward\n(1D)', 2, 2, output_color),
        ('Action Probs\n(4D)', 5.5, 3.5, output_color),
        ('Value Dist\n(601D)', 9, 3.5, output_color),
        ('Afterstate\n(256D)', 12.5, 3.5, output_color)
    ]

    for name, x, y, color in outputs:
        box = FancyBboxPatch((x-0.75, y-0.5), 1.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=9, ha='center', va='center')

    # Arrows - Main flow
    arrows = [
        # Main forward flow
        ((3.5, 9.75), (5, 9.75)),  # Input to TCN
        ((8, 9.75), (9.5, 9.75)),  # TCN to Repr
        ((12.5, 9.75), (13.5, 9.75)),  # Repr to Hidden

        # Hidden to networks
        ((14.5, 9), (14.5, 7.5)),  # Hidden down
        ((14.5, 7.5), (2, 7.5)),   # Horizontal line
        ((2, 7.5), (2, 6.75)),     # To Dynamics
        ((3.5, 7.5), (5.5, 6.75)), # To Policy
        ((7, 7.5), (9, 6.75)),     # To Value
        ((11, 7.5), (12.5, 6.75)), # To Afterstate

        # Networks to outputs
        ((2, 5.25), (2, 4)),       # Dynamics to Next State
        ((2, 5.25), (2, 2.5)),     # Dynamics to Reward
        ((5.5, 5.25), (5.5, 4)),   # Policy to Action
        ((9, 5.25), (9, 4)),       # Value to Dist
        ((12.5, 5.25), (12.5, 4)), # Afterstate to output
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Legend
    legend_x, legend_y = 1, 1.5
    legend_items = [
        ('Input/Features', input_color),
        ('TCN Processing', tcn_color),
        ('Representation', repr_color),
        ('Networks', network_color),
        ('Outputs', output_color)
    ]

    for i, (label, color) in enumerate(legend_items):
        rect = patches.Rectangle((legend_x + i*2.5, legend_y), 0.3, 0.3,
                                facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + i*2.5 + 0.4, legend_y + 0.15, label, fontsize=9, va='center')

    plt.tight_layout()
    plt.savefig('micro_muzero_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: micro_muzero_architecture.png")


def create_feature_breakdown():
    """Create detailed feature breakdown visualization."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Left: Temporal dimension
    ax1.set_title('Temporal Dimension (32 timesteps)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 32)

    # Draw timesteps
    for t in range(0, 32, 2):
        color = plt.cm.Blues(t / 31)
        rect = patches.Rectangle((2, t), 6, 1.8, facecolor=color, edgecolor='black', alpha=0.7)
        ax1.add_patch(rect)
        if t % 4 == 0:
            ax1.text(1.5, t + 0.9, f't={t}', fontsize=9, ha='right', va='center')
            if t == 0:
                ax1.text(8.5, t + 0.9, '32 bars ago', fontsize=9, ha='left', va='center')
            elif t == 28:
                ax1.text(8.5, t + 0.9, '4 bars ago', fontsize=9, ha='left', va='center')

    # Current timestep highlight
    rect = patches.Rectangle((2, 30), 6, 1.8, facecolor='red', edgecolor='black', alpha=0.7)
    ax1.add_patch(rect)
    ax1.text(1.5, 30.9, 't=31', fontsize=9, ha='right', va='center', fontweight='bold')
    ax1.text(8.5, 30.9, 'Current', fontsize=9, ha='left', va='center', fontweight='bold')

    ax1.set_ylabel('Timestep', fontsize=12)
    ax1.text(5, -2, 'Each timestep contains 15 features', fontsize=12, ha='center', fontweight='bold')

    # Right: Feature dimension
    ax2.set_title('Feature Dimension (15 features per timestep)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 15)

    # Feature groups
    feature_groups = [
        ('Technical Indicators', 0, 5, '#FFE6CC', [
            'position_in_range_60',
            'momentum_60',
            'rolling_range',
            'momentum_5',
            '(reserved)'
        ]),
        ('Cyclical Time', 5, 9, '#E6F3FF', [
            'dow_cos_final',
            'dow_sin_final',
            'hour_cos_final',
            'hour_sin_final'
        ]),
        ('Position State', 9, 15, '#E6FFE6', [
            'position_side',
            'position_pips',
            'bars_since_entry',
            'pips_from_peak',
            'max_drawdown_pips',
            'accumulated_dd'
        ])
    ]

    for group_name, start, end, color, features in feature_groups:
        # Group box
        rect = patches.Rectangle((1, start), 10, end-start,
                                facecolor=color, edgecolor='black', linewidth=2, alpha=0.3)
        ax2.add_patch(rect)

        # Group label
        ax2.text(0.5, (start + end) / 2, group_name, fontsize=11, fontweight='bold',
                ha='right', va='center', rotation=90)

        # Individual features
        for i, feature in enumerate(features):
            y_pos = start + i + 0.5
            if y_pos < end:
                feat_rect = patches.Rectangle((1.5, start + i + 0.1), 9, 0.8,
                                            facecolor='white', edgecolor='black', alpha=0.8)
                ax2.add_patch(feat_rect)
                ax2.text(6, y_pos, f'[{int(start + i)}] {feature}',
                        fontsize=9, ha='center', va='center')

    ax2.set_ylabel('Feature Index', fontsize=12)
    ax2.set_yticks(range(0, 16, 2))
    ax2.text(6, -1, 'Each feature is scaled and normalized', fontsize=12, ha='center', fontweight='bold')

    # Remove x-axis for cleaner look
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig('micro_feature_breakdown.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: micro_feature_breakdown.png")


def create_tcn_architecture():
    """Create TCN architecture visualization."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.5, 'TCN Encoder Architecture', fontsize=18, fontweight='bold', ha='center')

    # Input
    input_box = FancyBboxPatch((0.5, 5.5), 2, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#E8F4FD',
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 6.25, 'Input\n(32×15)', fontsize=11, fontweight='bold', ha='center', va='center')

    # TCN Layers
    tcn_layers = [
        ('Dilation 1\nRF=3', 3.5, '#FFE6CC'),
        ('Dilation 2\nRF=7', 5.5, '#FFCC99'),
        ('Dilation 4\nRF=15', 7.5, '#FF9966'),
        ('Attention\nPool', 9.5, '#CC6600')
    ]

    for name, x, color in tcn_layers:
        box = FancyBboxPatch((x-0.75, 5.5), 1.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 6.25, name, fontsize=10, fontweight='bold', ha='center', va='center')

    # Skip connection
    skip_box = FancyBboxPatch((11.5, 5.5), 2, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#D4E6F1',
                              edgecolor='black', linewidth=2)
    ax.add_patch(skip_box)
    ax.text(12.5, 6.25, 'Skip Concat\n48D+15D→63D', fontsize=11, fontweight='bold', ha='center', va='center')

    # Arrows
    arrows = [
        ((2.5, 6.25), (2.75, 6.25)),  # Input to first layer
        ((4.25, 6.25), (4.75, 6.25)), # Between layers
        ((6.25, 6.25), (6.75, 6.25)),
        ((8.25, 6.25), (8.75, 6.25)),
        ((10.25, 6.25), (11.5, 6.25)), # To skip
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Skip connection arrow (curved)
    ax.annotate('', xy=(12.5, 5.4), xytext=(1.5, 5.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='red',
                               connectionstyle="arc3,rad=0.3"))
    ax.text(7, 4.8, 'Skip Connection (Raw Features)', fontsize=10, color='red',
            ha='center', fontweight='bold')

    # Receptive field explanation
    rf_text = """Receptive Field Growth:
    Dilation 1: 3 timesteps
    Dilation 2: 7 timesteps
    Dilation 4: 15 timesteps
    → Captures multi-scale patterns"""

    ax.text(7, 3.5, rf_text, fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F0F0F0', alpha=0.8))

    plt.tight_layout()
    plt.savefig('tcn_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: tcn_architecture.png")


def create_data_flow():
    """Create data flow visualization."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(8, 9.5, 'Micro MuZero Data Flow', fontsize=18, fontweight='bold', ha='center')

    # Data sources
    sources = [
        ('Market Data\nOHLCV', 1, 8, '#E8F4FD'),
        ('Technical\nIndicators', 3.5, 8, '#FFE6CC'),
        ('Time\nFeatures', 6, 8, '#E6F3FF'),
        ('Position\nState', 8.5, 8, '#E6FFE6')
    ]

    for name, x, y, color in sources:
        box = FancyBboxPatch((x-0.75, y-0.5), 1.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center')

    # Consolidation
    consolidate_box = FancyBboxPatch((4, 6), 3, 1,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#F0F0F0',
                                     edgecolor='black', linewidth=2)
    ax.add_patch(consolidate_box)
    ax.text(5.5, 6.5, 'Feature Engineering\n32×15 Windows',
            fontsize=12, fontweight='bold', ha='center', va='center')

    # Neural networks
    networks = [
        ('TCN\nEncoder', 2, 4.5, '#FFE6CC'),
        ('Repr\nNetwork', 4.5, 4.5, '#D4E6F1'),
        ('Policy\nNet', 7, 4.5, '#D5F3D0'),
        ('Value\nNet', 9.5, 4.5, '#D5F3D0'),
        ('Dynamics\nNet', 12, 4.5, '#D5F3D0')
    ]

    for name, x, y, color in networks:
        box = FancyBboxPatch((x-0.75, y-0.5), 1.5, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center')

    # Trading actions
    actions = [
        ('Hold', 3, 2.5, '#F8D7DA'),
        ('Buy', 5.5, 2.5, '#D1ECF1'),
        ('Sell', 8, 2.5, '#FFF3CD'),
        ('Close', 10.5, 2.5, '#D4EDDA')
    ]

    for name, x, y, color in actions:
        box = FancyBboxPatch((x-0.75, y-0.4), 1.5, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center')

    # MCTS
    mcts_box = FancyBboxPatch((13, 2), 2.5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#F5F5F5',
                              edgecolor='black', linewidth=2)
    ax.add_patch(mcts_box)
    ax.text(14.25, 2.75, 'MCTS\nPlanning\n25 sims',
            fontsize=11, fontweight='bold', ha='center', va='center')

    # Flow arrows
    flow_arrows = [
        # Sources to consolidation
        ((1, 7.5), (4.5, 6.8)),
        ((3.5, 7.5), (5, 6.8)),
        ((6, 7.5), (5.5, 6.8)),
        ((8.5, 7.5), (6, 6.8)),

        # Consolidation to networks
        ((5.5, 6), (2, 5)),
        ((5.5, 6), (4.5, 5)),
        ((5.5, 6), (7, 5)),
        ((5.5, 6), (9.5, 5)),
        ((5.5, 6), (12, 5)),

        # Networks to actions
        ((7, 4), (6.75, 3.2)),

        # To MCTS
        ((9.5, 4), (13.5, 3.2)),
        ((12, 4), (13.8, 3.2)),
    ]

    for start, end in flow_arrows:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.7))

    # Labels
    ax.text(1, 1, 'Data Sources', fontsize=12, fontweight='bold', color='blue')
    ax.text(6.75, 1, 'Actions', fontsize=12, fontweight='bold', color='green')
    ax.text(14.25, 1, 'Planning', fontsize=12, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: data_flow_diagram.png")


if __name__ == "__main__":
    print("Creating PNG visualizations for Micro MuZero...")

    # Create all visualizations
    create_network_diagram()
    create_feature_breakdown()
    create_tcn_architecture()
    create_data_flow()

    print("\n🎉 All visualizations created successfully!")
    print("Files created:")
    print("  - micro_muzero_architecture.png")
    print("  - micro_feature_breakdown.png")
    print("  - tcn_architecture.png")
    print("  - data_flow_diagram.png")