#!/usr/bin/env python3
"""
Kymatio WST Dimension Analysis
Understanding why we're getting low-dimensional outputs instead of 128D
"""

def calculate_kymatio_dimensions(J, Q, max_order=2):
    """
    Calculate expected WST output dimensions based on kymatio parameters

    For 1D Scattering Transform:
    - Order 0: 1 coefficient (low-pass)
    - Order 1: J×Q coefficients (one per wavelet)
    - Order 2: (J×Q choose 2) + (J×Q × (J-1)×Q) / 2 (pairs of wavelets)

    Total dimensions ≈ 1 + J×Q + (J×Q)²/2 for max_order=2
    """

    # Order 0: Low-pass filter (always 1)
    order_0 = 1

    # Order 1: One coefficient per wavelet
    order_1 = J * Q

    if max_order == 1:
        return order_0 + order_1

    # Order 2: Pairs of wavelets (simplified approximation)
    # This is more complex in reality due to scale constraints
    order_2_approx = (J * Q * (J * Q - 1)) // 2

    total_approx = order_0 + order_1 + order_2_approx

    return {
        'order_0': order_0,
        'order_1': order_1,
        'order_2_approx': order_2_approx,
        'total_approx': total_approx
    }

def analyze_parameter_combinations():
    """Analyze different parameter combinations"""

    print("🔬 KYMATIO WST DIMENSION ANALYSIS")
    print("=" * 60)
    print("Target: ~128 dimensions for rich feature representation")
    print("=" * 60)

    # Test combinations from the failed generation
    failed_combinations = [
        {'J': 8, 'Q': 1, 'max_order': 2},  # Generated 1D
        {'J': 6, 'Q': 8, 'max_order': 2},  # Generated 4D
        {'J': 7, 'Q': 4, 'max_order': 2},  # Generated 2D
        {'J': 8, 'Q': 2, 'max_order': 2},  # Generated 1D
    ]

    print("\n❌ FAILED COMBINATIONS (from logs):")
    for i, params in enumerate(failed_combinations):
        dims = calculate_kymatio_dimensions(**params)
        actual = [1, 4, 2, 1][i]  # From the logs
        print(f"   J={params['J']:2d}, Q={params['Q']:2d} → Predicted: {dims['total_approx']:3d}, Actual: {actual:3d}")

    print(f"\n🎯 FINDING PARAMETERS FOR ~128 DIMENSIONS:")
    print(f"   Target range: 120-135 dimensions")

    # Search for combinations that might work
    promising_combinations = []

    for J in range(4, 10):  # J from 4 to 9
        for Q in range(1, 20):  # Q from 1 to 19
            dims = calculate_kymatio_dimensions(J, Q, max_order=2)
            if 120 <= dims['total_approx'] <= 135:
                promising_combinations.append((J, Q, dims))

    if promising_combinations:
        print(f"\n✅ PROMISING COMBINATIONS:")
        for J, Q, dims in promising_combinations[:10]:  # Show top 10
            print(f"   J={J:2d}, Q={Q:2d} → ~{dims['total_approx']:3d} dims (O0:{dims['order_0']}, O1:{dims['order_1']}, O2:{dims['order_2_approx']})")
    else:
        print(f"\n⚠️  NO EXACT MATCHES FOUND")

        # Find closest matches
        print(f"\n🔍 CLOSEST MATCHES:")
        all_combinations = []
        for J in range(4, 10):
            for Q in range(1, 20):
                dims = calculate_kymatio_dimensions(J, Q, max_order=2)
                diff = abs(dims['total_approx'] - 128)
                all_combinations.append((J, Q, dims, diff))

        # Sort by difference from 128
        all_combinations.sort(key=lambda x: x[3])

        for J, Q, dims, diff in all_combinations[:10]:
            print(f"   J={J:2d}, Q={Q:2d} → ~{dims['total_approx']:3d} dims (Δ{diff:2d})")

    print("\n" + "=" * 60)
    print("🚨 ROOT CAUSE ANALYSIS:")
    print("=" * 60)

    print("""
    The issue is likely one of these:

    1. 📐 WINDOW SIZE TOO SMALL:
       - Current: 256 samples
       - For higher J values, need longer windows
       - Rule: window_size ≥ 2^J for stable results

    2. ⚙️ KYMATIO VERSION/API:
       - Different versions may calculate dimensions differently
       - Some parameters may be ignored or handled differently

    3. 🔧 PARAMETER CONSTRAINTS:
       - Kymatio may internally limit combinations
       - Scale separation requirements may reduce actual output

    4. 🎛️ RECOMMENDED APPROACH:
       - Use empirical testing with actual kymatio calls
       - Try max_order=3 or different window sizes
       - Consider multiple scattering transforms combined
    """)

    print("\n💡 IMMEDIATE SOLUTIONS:")
    print("1. Test with window_size=512 or 1024 instead of 256")
    print("2. Try max_order=3 for more coefficients")
    print("3. Use multiple transforms with different parameters and concatenate")
    print("4. Fall back to manual WST with expansion to 128D")

if __name__ == "__main__":
    analyze_parameter_combinations()