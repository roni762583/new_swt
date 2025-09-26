#!/usr/bin/env python3
"""
Quick test runner for feature analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_analysis import FeatureAnalyzer
import pandas as pd

def quick_test():
    """Run a quick test with smaller sample."""
    print("🚀 Running quick feature analysis test...")
    print("-" * 50)

    analyzer = FeatureAnalyzer()

    # Use smaller sample for quick test
    df = analyzer.load_data(sample_size=10000)

    # Run tests
    results = analyzer.run_linear_tests(df)

    # Print compact results
    print("\n📊 QUICK RESULTS:")
    print("-" * 50)
    for _, row in results.iterrows():
        h = row['horizon_min']
        r2_base = row['r2_base']
        r2_full = row['r2_with_time']
        gain = row['r2_gain']
        print(f"{h:2d} min: R²={r2_base:.4f} → {r2_full:.4f} (gain: {gain:+.4f})")

    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("-" * 50)

    best_horizon = results.loc[results['r2_base'].idxmax(), 'horizon_min']
    best_r2 = results['r2_base'].max()
    print(f"✅ Best base predictability: {best_horizon} min (R²={best_r2:.4f})")

    best_time_gain_horizon = results.loc[results['r2_gain'].idxmax(), 'horizon_min']
    best_time_gain = results['r2_gain'].max()
    print(f"✅ Time features help most at: {best_time_gain_horizon} min (+{best_time_gain:.4f} R²)")

    avg_improvement = results['r2_gain'].mean()
    print(f"✅ Average improvement from time: {avg_improvement:.4f}")

    return results

if __name__ == "__main__":
    results = quick_test()