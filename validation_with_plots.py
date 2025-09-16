#!/usr/bin/env python3
"""
Comprehensive validation with distribution plots
Shows the variability in validation results across multiple runs
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import subprocess
import sys

def run_validation_batch(checkpoint_path, wst_file, csv_file, num_runs=100, runs_per_batch=10):
    """Run validation multiple times and collect results"""
    all_results = []

    for batch in range(0, num_runs, runs_per_batch):
        batch_size = min(runs_per_batch, num_runs - batch)
        print(f"Running batch {batch//runs_per_batch + 1}/{(num_runs + runs_per_batch - 1)//runs_per_batch} ({batch_size} runs)...")

        cmd = [
            "python", "swt_validation/validate_with_precomputed_wst.py",
            "--checkpoints", checkpoint_path,
            "--wst-file", wst_file,
            "--csv-file", csv_file,
            "--runs", str(batch_size)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Parse results from output
            output = result.stdout + result.stderr
            for line in output.split('\n'):
                if 'Total Return:' in line:
                    try:
                        ret = float(line.split('Total Return:')[1].split('%')[0].strip())
                        all_results.append({'return': ret})
                    except:
                        pass
                elif 'Sharpe Ratio:' in line and all_results and 'sharpe' not in all_results[-1]:
                    try:
                        sharpe = float(line.split('Sharpe Ratio:')[1].strip().split()[0])
                        all_results[-1]['sharpe'] = sharpe
                    except:
                        pass
                elif 'Win Rate:' in line and all_results and 'win_rate' not in all_results[-1]:
                    try:
                        wr = float(line.split('Win Rate:')[1].split('%')[0].strip())
                        all_results[-1]['win_rate'] = wr
                    except:
                        pass

        except subprocess.TimeoutExpired:
            print(f"Batch {batch//runs_per_batch + 1} timed out, continuing...")
        except Exception as e:
            print(f"Error in batch {batch//runs_per_batch + 1}: {e}")

    return all_results

def create_distribution_plots(results, output_dir, checkpoint_name):
    """Create comprehensive distribution plots"""

    if not results:
        print("No results to plot!")
        return

    # Extract metrics
    returns = [r['return'] for r in results if 'return' in r]
    sharpes = [r['sharpe'] for r in results if 'sharpe' in r]
    win_rates = [r['win_rate'] for r in results if 'win_rate' in r]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Validation Distribution Analysis: {checkpoint_name}\n{len(results)} Monte Carlo Runs', fontsize=14)

    # 1. Returns Box Plot
    if returns:
        ax = axes[0, 0]
        box = ax.boxplot(returns, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'Returns Distribution\nMedian: {np.median(returns):.1f}%')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 2. Returns Histogram
    if returns:
        ax = axes[0, 1]
        ax.hist(returns, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=np.median(returns), color='r', linestyle='--', label=f'Median: {np.median(returns):.1f}%')
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Histogram')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Sharpe Ratio Box Plot
    if sharpes:
        ax = axes[0, 2]
        box = ax.boxplot(sharpes, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Sharpe Distribution\nMedian: {np.median(sharpes):.3f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 4. Win Rate Box Plot
    if win_rates:
        ax = axes[1, 0]
        box = ax.boxplot(win_rates, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightyellow')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title(f'Win Rate Distribution\nMedian: {np.median(win_rates):.1f}%')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)

    # 5. Combined Violin Plot
    if returns and sharpes:
        ax = axes[1, 1]
        # Normalize for comparison
        norm_returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)
        norm_sharpes = (np.array(sharpes) - np.mean(sharpes)) / (np.std(sharpes) + 1e-8)

        parts = ax.violinplot([norm_returns, norm_sharpes], positions=[1, 2], showmeans=True, showmedians=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Returns\n(normalized)', 'Sharpe\n(normalized)'])
        ax.set_ylabel('Normalized Value')
        ax.set_title('Distribution Comparison')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # 6. Statistics Summary Table
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = "Statistics Summary\n" + "="*30 + "\n"

    if returns:
        stats_text += f"\nReturns:\n"
        stats_text += f"  Mean:   {np.mean(returns):>8.1f}%\n"
        stats_text += f"  Median: {np.median(returns):>8.1f}%\n"
        stats_text += f"  StdDev: {np.std(returns):>8.1f}%\n"
        stats_text += f"  Min:    {np.min(returns):>8.1f}%\n"
        stats_text += f"  Max:    {np.max(returns):>8.1f}%\n"
        stats_text += f"  >0:     {sum(1 for r in returns if r > 0):>8d}/{len(returns)}\n"

    if sharpes:
        stats_text += f"\nSharpe Ratio:\n"
        stats_text += f"  Mean:   {np.mean(sharpes):>8.3f}\n"
        stats_text += f"  Median: {np.median(sharpes):>8.3f}\n"
        stats_text += f"  >0:     {sum(1 for s in sharpes if s > 0):>8d}/{len(sharpes)}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"validation_plots_{timestamp}_{checkpoint_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved distribution plots to {plot_file}")

    # Also save as PDF
    pdf_file = output_dir / f"validation_plots_{timestamp}_{checkpoint_name}.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📄 Saved PDF report to {pdf_file}")

    plt.close()

    # Save raw results to JSON
    json_file = output_dir / f"validation_results_{timestamp}_{checkpoint_name}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'checkpoint': checkpoint_name,
            'timestamp': timestamp,
            'num_runs': len(results),
            'results': results,
            'summary': {
                'returns': {
                    'mean': float(np.mean(returns)) if returns else None,
                    'median': float(np.median(returns)) if returns else None,
                    'std': float(np.std(returns)) if returns else None,
                    'min': float(np.min(returns)) if returns else None,
                    'max': float(np.max(returns)) if returns else None,
                    'positive_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
                },
                'sharpe': {
                    'mean': float(np.mean(sharpes)) if sharpes else None,
                    'median': float(np.median(sharpes)) if sharpes else None,
                    'positive_rate': sum(1 for s in sharpes if s > 0) / len(sharpes) if sharpes else 0
                }
            }
        }, f, indent=2)
    print(f"💾 Saved raw results to {json_file}")

    return plot_file

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive validation with distribution analysis')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--wst-file', required=True, help='Path to WST file')
    parser.add_argument('--csv-file', required=True, help='Path to CSV file')
    parser.add_argument('--runs', type=int, default=100, help='Number of validation runs')
    parser.add_argument('--output-dir', default='validation_results', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"Running {args.runs} validation runs for {checkpoint_name}")
    print(f"This will show the TRUE distribution of results")
    print(f"{'='*60}\n")

    # Run validations
    results = run_validation_batch(
        args.checkpoint,
        args.wst_file,
        args.csv_file,
        args.runs
    )

    if results:
        print(f"\n✅ Collected {len(results)} validation results")

        # Create plots
        create_distribution_plots(results, output_dir, checkpoint_name)

        # Print summary
        returns = [r['return'] for r in results if 'return' in r]
        if returns:
            print(f"\n📊 SUMMARY:")
            print(f"  Return: {np.median(returns):.1f}% median ({np.min(returns):.1f}% to {np.max(returns):.1f}% range)")
            print(f"  Consistency: {sum(1 for r in returns if r > 0)}/{len(returns)} runs profitable")
    else:
        print("❌ No results collected!")
        sys.exit(1)

if __name__ == "__main__":
    main()