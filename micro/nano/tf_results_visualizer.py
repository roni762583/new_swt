#!/usr/bin/env python3
"""
Visualize and summarize timeframe comparison results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_visualization():
    """Create comprehensive visualization of TF results"""

    # Results from our analysis
    results = {
        'M1/H1': {'trades': 1216, 'win_rate': 34.8, 'total_pips': 138.5, 'pips_per_trade': 0.11},
        'M1/H4': {'trades': 1098, 'win_rate': 35.5, 'total_pips': 46.8, 'pips_per_trade': 0.04},
        'M5/H1': {'trades': 254, 'win_rate': 38.6, 'total_pips': 444.6, 'pips_per_trade': 1.75},
        'M5/H4': {'trades': 223, 'win_rate': 37.7, 'total_pips': 331.0, 'pips_per_trade': 1.48},
        'M15/H1': {'trades': 99, 'win_rate': 40.4, 'total_pips': 230.5, 'pips_per_trade': 2.33},
        'M15/H4': {'trades': 77, 'win_rate': 40.3, 'total_pips': 252.8, 'pips_per_trade': 3.28},
        'M30/H4': {'trades': 38, 'win_rate': 47.4, 'total_pips': 305.7, 'pips_per_trade': 8.04},
        'H1/H4': {'trades': 20, 'win_rate': 55.0, 'total_pips': 425.0, 'pips_per_trade': 21.25},
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Total Pips Comparison
    ax = axes[0, 0]
    tf_pairs = list(results.keys())
    total_pips = [results[tf]['total_pips'] for tf in tf_pairs]
    colors = ['green' if p > 0 else 'red' for p in total_pips]

    bars = ax.bar(tf_pairs, total_pips, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Total Pips by Timeframe Combination', fontsize=12, fontweight='bold')
    ax.set_xlabel('Timeframe Pair')
    ax.set_ylabel('Total Pips')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, total_pips):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

    # 2. Win Rate vs Pips Per Trade
    ax = axes[0, 1]
    win_rates = [results[tf]['win_rate'] for tf in tf_pairs]
    pips_per_trade = [results[tf]['pips_per_trade'] for tf in tf_pairs]

    scatter = ax.scatter(win_rates, pips_per_trade, s=100, c=total_pips,
                        cmap='RdYlGn', alpha=0.7, edgecolors='black')

    # Add labels for each point
    for i, tf in enumerate(tf_pairs):
        ax.annotate(tf, (win_rates[i], pips_per_trade[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title('Win Rate vs Efficiency', fontsize=12, fontweight='bold')
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Pips per Trade')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Pips', rotation=270, labelpad=15)

    # 3. Trade Frequency Analysis
    ax = axes[1, 0]
    trade_counts = [results[tf]['trades'] for tf in tf_pairs]

    # Create grouped bar chart
    x = np.arange(len(tf_pairs))
    width = 0.35

    bars1 = ax.bar(x - width/2, trade_counts, width, label='Total Trades', alpha=0.7)
    bars2 = ax.bar(x + width/2, [tc * wr/100 for tc, wr in zip(trade_counts, win_rates)],
                  width, label='Winning Trades', alpha=0.7, color='green')

    ax.set_title('Trade Volume Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Timeframe Pair')
    ax.set_ylabel('Number of Trades')
    ax.set_xticks(x)
    ax.set_xticklabels(tf_pairs, rotation=45)
    ax.legend()

    # 4. Risk-Adjusted Performance Matrix
    ax = axes[1, 1]

    # Create performance matrix
    exec_tfs = ['M1', 'M5', 'M15', 'M30', 'H1']
    context_tfs = ['H1', 'H4']

    matrix_data = np.zeros((len(exec_tfs), len(context_tfs)))

    for i, exec_tf in enumerate(exec_tfs):
        for j, context_tf in enumerate(context_tfs):
            pair = f"{exec_tf}/{context_tf}"
            if pair in results:
                matrix_data[i, j] = results[pair]['total_pips']

    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=-500, vmax=500)

    # Add text annotations
    for i in range(len(exec_tfs)):
        for j in range(len(context_tfs)):
            text = ax.text(j, i, f'{matrix_data[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks(np.arange(len(context_tfs)))
    ax.set_yticks(np.arange(len(exec_tfs)))
    ax.set_xticklabels(context_tfs)
    ax.set_yticklabels(exec_tfs)
    ax.set_xlabel('Context Timeframe')
    ax.set_ylabel('Execution Timeframe')
    ax.set_title('Performance Heatmap', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Pips', rotation=270, labelpad=15)

    plt.suptitle('Comprehensive Timeframe Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tf_comparison_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("Visualization saved to tf_comparison_visualization.png")


def generate_final_report():
    """Generate final recommendations"""

    print("\n" + "="*70)
    print("FINAL TIMEFRAME RECOMMENDATIONS FOR RL IMPLEMENTATION")
    print("="*70)

    print("\n🏆 RANKING BY PROFITABILITY:")
    print("1. M5/H1  - 444.6 pips (WINNER)")
    print("2. H1/H4  - 425.0 pips")
    print("3. M5/H4  - 331.0 pips")
    print("4. M30/H4 - 305.7 pips")
    print("5. M15/H4 - 252.8 pips")

    print("\n📊 KEY FINDINGS:")
    print("• M5 execution timeframe is optimal (not M1)")
    print("• H1 context works better with faster execution (M5)")
    print("• H4 context works better with slower execution (H1)")
    print("• M1 has too much noise despite many trades")
    print("• Higher timeframes (H1/H4) have best pips/trade ratio")

    print("\n🤖 RL STATE SPACE RECOMMENDATIONS:")
    print("""
    Option 1 - Best Overall (M5/H1):
    - Execution: M5 bars (5-minute)
    - Context: H1 bars (60-minute)
    - Pros: Best total profit, good trade frequency
    - Cons: Moderate complexity

    Option 2 - Best Efficiency (H1/H4):
    - Execution: H1 bars (60-minute)
    - Context: H4 bars (240-minute)
    - Pros: Highest pips per trade (21.25)
    - Cons: Very few trades (20 total)

    Option 3 - Balanced (M15/H4):
    - Execution: M15 bars (15-minute)
    - Context: H4 bars (240-minute)
    - Pros: Good balance of trades and profit
    - Cons: Slower signals
    """)

    print("\n⚡ FINAL VERDICT:")
    print("Use M5/H1 for RL agent training because:")
    print("1. Highest total profitability (444.6 pips)")
    print("2. Sufficient trade frequency (254 trades)")
    print("3. Reasonable win rate (38.6%)")
    print("4. Good pips per trade (1.75)")
    print("5. Balance between noise filtering and opportunity")

    print("\n💡 IMPLEMENTATION NOTE:")
    print("The original M1/H1 assumption was close but not optimal.")
    print("M5 removes noise while preserving trading opportunities.")
    print("This is crucial for RL agents to learn meaningful patterns.")

    print("\n" + "="*70)


def main():
    """Create visualization and final report"""
    create_visualization()
    generate_final_report()


if __name__ == "__main__":
    main()