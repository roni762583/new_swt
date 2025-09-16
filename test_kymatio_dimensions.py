#!/usr/bin/env python3
"""
Test script to verify Kymatio produces 128-dimensional features
"""

import torch
from kymatio.torch import Scattering1D
import numpy as np

def test_kymatio_dimensions():
    """Test different Kymatio parameter combinations"""

    window_size = 256
    device = torch.device('cpu')

    # Test different parameter combinations
    param_combinations = [
        {'J': 8, 'Q': 1, 'max_order': 2},  # Default
        {'J': 6, 'Q': 8, 'max_order': 2},  # Often gives 127-129 features
        {'J': 7, 'Q': 4, 'max_order': 2},  # Alternative
        {'J': 8, 'Q': 2, 'max_order': 2},  # Another option
        {'J': 5, 'Q': 16, 'max_order': 2}, # High Q
    ]

    print("Testing Kymatio WST parameter combinations:")
    print("=" * 60)

    for params in param_combinations:
        try:
            scattering = Scattering1D(
                J=params['J'],
                shape=(window_size,),
                Q=params['Q'],
                max_order=params['max_order'],
                frontend='torch'
            ).to(device)

            # Test with dummy input
            dummy_input = torch.randn(1, window_size).to(device)
            output = scattering(dummy_input)

            output_dim = output.shape[-1]

            status = "✅" if 126 <= output_dim <= 130 else "❌"
            print(f"{status} J={params['J']:2d}, Q={params['Q']:2d}, max_order={params['max_order']} → {output_dim:3d} features")

        except Exception as e:
            print(f"❌ J={params['J']:2d}, Q={params['Q']:2d}, max_order={params['max_order']} → Error: {e}")

    print("=" * 60)
    print("\nRecommended configuration for 128 features:")
    print("J=6, Q=8, max_order=2 (typically produces 127-129 features)")

if __name__ == "__main__":
    test_kymatio_dimensions()