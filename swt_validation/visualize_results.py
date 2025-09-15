#!/usr/bin/env python3
"""
Visualization script for Monte Carlo validation results
Generates comprehensive graphs including broom bristle charts
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ValidationVisualizer:
    """Create visualizations from validation results"""

    def __init__(self, results_dir: str = "validation_results", output_dir: str = "validation_results/graphs"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def parse_validation_log(self, log_file: str) -> Dict:
        """Parse validation log file to extract results"""
        results = {
            'runs': [],
            'returns': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'equity_curves': []
        }

        log_path = self.results_dir / log_file
        if not log_path.exists():
            logger.warning(f"Log file not found: {log_path}")
            return results

        with open(log_path, 'r') as f:
            lines = f.readlines()

        current_run = None
        equity_curve = []

        for line in lines:
            # Parse run results
            if "Run " in line and "/" in line:
                parts = line.split("Run ")[1].split("/")
                current_run = int(parts[0])

            if "Return:" in line and "%" in line:
                try:
                    return_val = float(line.split("Return:")[1].split("%")[0])
                    results['returns'].append(return_val)
                except:
                    pass

            if "Win Rate:" in line and "%" in line:
                try:
                    win_rate = float(line.split("Win Rate:")[1].split("%")[0])
                    results['win_rates'].append(win_rate)
                except:
                    pass

            if "Sharpe:" in line:
                try:
                    sharpe = float(line.split("Sharpe:")[1].split()[0])
                    results['sharpe_ratios'].append(sharpe)
                except:
                    pass

            if "Max DD:" in line and "%" in line:
                try:
                    max_dd = float(line.split("Max DD:")[1].split("%")[0])
                    results['max_drawdowns'].append(max_dd)
                except:
                    pass

            # Try to extract equity curve data
            if "Balance:" in line or "Equity:" in line:
                try:
                    balance = float(line.split()[-1])
                    equity_curve.append(balance)
                except:
                    pass

        if equity_curve:
            results['equity_curves'].append(equity_curve)

        return results

    def create_broom_bristle_chart(self, results: Dict, title: str = "Monte Carlo Simulation Paths"):
        """Create broom bristle chart showing all MC simulation paths"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # 1. Returns Distribution with Bristles
        ax1 = axes[0, 0]
        if results['returns']:
            # Create synthetic equity curves from returns
            n_runs = len(results['returns'])
            n_steps = 100  # Synthetic steps

            for i in range(n_runs):
                # Generate random walk with drift based on return
                daily_return = results['returns'][i] / n_steps / 100
                daily_vol = 0.02  # 2% daily volatility assumption

                returns = np.random.normal(daily_return, daily_vol, n_steps)
                equity = 10000 * np.exp(np.cumsum(returns))

                ax1.plot(equity, alpha=0.3, linewidth=0.5, color='blue')

            # Add mean path
            mean_return = np.mean(results['returns']) / n_steps / 100
            mean_equity = 10000 * np.exp(np.cumsum([mean_return] * n_steps))
            ax1.plot(mean_equity, color='red', linewidth=2, label='Mean Path')

            ax1.set_title('Monte Carlo Equity Paths (Broom Bristle)')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Returns Distribution Histogram
        ax2 = axes[0, 1]
        if results['returns']:
            ax2.hist(results['returns'], bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.mean(results['returns']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(results['returns']):.2f}%')
            ax2.axvline(np.median(results['returns']), color='orange', linestyle='--',
                       label=f'Median: {np.median(results['returns']):.2f}%')
            ax2.set_title('Returns Distribution')
            ax2.set_xlabel('Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Risk Metrics Scatter
        ax3 = axes[1, 0]
        if results['returns'] and results['max_drawdowns']:
            scatter = ax3.scatter(results['max_drawdowns'], results['returns'],
                                 c=results.get('sharpe_ratios', results['returns']),
                                 cmap='RdYlGn', s=50, alpha=0.6)
            ax3.set_title('Risk-Return Profile')
            ax3.set_xlabel('Max Drawdown (%)')
            ax3.set_ylabel('Return (%)')

            # Add colorbar for Sharpe ratio
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Sharpe Ratio')

            # Add quadrant lines
            ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
            ax3.axvline(np.mean(results['max_drawdowns']), color='gray', linestyle='--', alpha=0.3)
            ax3.grid(True, alpha=0.3)

        # 4. Performance Metrics Box Plot
        ax4 = axes[1, 1]
        if results['returns']:
            metrics_data = []
            metrics_labels = []

            if results['returns']:
                metrics_data.append(results['returns'])
                metrics_labels.append('Returns (%)')
            if results['win_rates']:
                metrics_data.append(results['win_rates'])
                metrics_labels.append('Win Rate (%)')
            if results['sharpe_ratios']:
                # Scale Sharpe ratios for visualization
                scaled_sharpe = [s * 10 for s in results['sharpe_ratios']]
                metrics_data.append(scaled_sharpe)
                metrics_labels.append('Sharpe (x10)')

            if metrics_data:
                bp = ax4.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)

                # Color the boxes
                colors = ['lightblue', 'lightgreen', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)

                ax4.set_title('Performance Metrics Distribution')
                ax4.set_ylabel('Value')
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_detailed_analysis(self, results: Dict, checkpoint_name: str):
        """Create detailed analysis plots"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'Detailed Analysis - {checkpoint_name}', fontsize=16, fontweight='bold')

        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        if results['returns']:
            sorted_returns = sorted(results['returns'])
            cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)

            ax1.plot(sorted_returns, cumulative_prob, linewidth=2)
            ax1.fill_between(sorted_returns, 0, cumulative_prob, alpha=0.3)
            ax1.set_title('Cumulative Distribution of Returns')
            ax1.set_xlabel('Return (%)')
            ax1.set_ylabel('Cumulative Probability')
            ax1.grid(True, alpha=0.3)

            # Add percentile markers
            percentiles = [5, 25, 50, 75, 95]
            for p in percentiles:
                val = np.percentile(results['returns'], p)
                ax1.axvline(val, color='red', linestyle='--', alpha=0.5)
                ax1.text(val, 0.5, f'P{p}: {val:.1f}%', rotation=90, va='bottom')

        # 2. Win Rate vs Return Scatter
        ax2 = axes[0, 1]
        if results['returns'] and results['win_rates']:
            ax2.scatter(results['win_rates'], results['returns'], alpha=0.6, s=50)

            # Add trend line
            z = np.polyfit(results['win_rates'], results['returns'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(results['win_rates']), max(results['win_rates']), 100)
            ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend')

            ax2.set_title('Win Rate vs Return Correlation')
            ax2.set_xlabel('Win Rate (%)')
            ax2.set_ylabel('Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Sharpe Ratio Distribution
        ax3 = axes[1, 0]
        if results['sharpe_ratios']:
            ax3.hist(results['sharpe_ratios'], bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(np.mean(results['sharpe_ratios']), color='red', linestyle='--',
                       label=f'Mean: {np.mean(results['sharpe_ratios']):.3f}')
            ax3.set_title('Sharpe Ratio Distribution')
            ax3.set_xlabel('Sharpe Ratio')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Drawdown Analysis
        ax4 = axes[1, 1]
        if results['max_drawdowns']:
            ax4.hist(results['max_drawdowns'], bins=20, alpha=0.7, color='red', edgecolor='black')
            ax4.axvline(np.mean(results['max_drawdowns']), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(results['max_drawdowns']):.2f}%')
            ax4.set_title('Maximum Drawdown Distribution')
            ax4.set_xlabel('Max Drawdown (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Performance Heatmap
        ax5 = axes[2, 0]
        if results['returns'] and len(results['returns']) >= 10:
            # Create correlation matrix
            metrics_df = pd.DataFrame({
                'Return': results['returns'][:min(len(results['returns']), 100)],
                'Win Rate': results['win_rates'][:min(len(results['win_rates']), 100)] if results['win_rates'] else results['returns'][:100],
                'Sharpe': results['sharpe_ratios'][:min(len(results['sharpe_ratios']), 100)] if results['sharpe_ratios'] else [0]*min(len(results['returns']), 100),
                'Max DD': results['max_drawdowns'][:min(len(results['max_drawdowns']), 100)] if results['max_drawdowns'] else [0]*min(len(results['returns']), 100)
            })

            corr_matrix = metrics_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       ax=ax5, cbar_kws={'label': 'Correlation'})
            ax5.set_title('Metrics Correlation Matrix')

        # 6. Summary Statistics
        ax6 = axes[2, 1]
        ax6.axis('off')

        summary_text = f"""
        SUMMARY STATISTICS
        {'='*30}

        Returns:
          Mean: {np.mean(results['returns']):.2f}%
          Std Dev: {np.std(results['returns']):.2f}%
          Min: {np.min(results['returns']):.2f}%
          Max: {np.max(results['returns']):.2f}%

        Win Rate:
          Mean: {np.mean(results['win_rates']):.2f}%
          Std Dev: {np.std(results['win_rates']):.2f}%

        Sharpe Ratio:
          Mean: {np.mean(results['sharpe_ratios']):.3f}
          Std Dev: {np.std(results['sharpe_ratios']):.3f}

        Max Drawdown:
          Mean: {np.mean(results['max_drawdowns']):.2f}%
          Worst: {np.max(results['max_drawdowns']):.2f}%

        Runs Analyzed: {len(results['returns'])}
        """

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        return fig

    def visualize_checkpoint(self, log_file: str, checkpoint_name: str):
        """Create all visualizations for a checkpoint"""
        logger.info(f"Creating visualizations for {checkpoint_name}")

        # Parse results
        results = self.parse_validation_log(log_file)

        if not results['returns']:
            logger.warning(f"No results found in {log_file}")
            return

        # Create broom bristle chart
        fig1 = self.create_broom_bristle_chart(results, f"Monte Carlo Broom Bristle - {checkpoint_name}")
        output_path1 = self.output_dir / f"{checkpoint_name}_broom_bristle.png"
        fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
        logger.info(f"Saved broom bristle chart to {output_path1}")

        # Create detailed analysis
        fig2 = self.create_detailed_analysis(results, checkpoint_name)
        output_path2 = self.output_dir / f"{checkpoint_name}_detailed_analysis.png"
        fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
        logger.info(f"Saved detailed analysis to {output_path2}")

        plt.close('all')

def main():
    """Main function to generate all visualizations"""
    visualizer = ValidationVisualizer()

    # Visualize available validation results
    validation_files = [
        ("episode_10_mc_car25.log", "Episode_10"),
        ("episode_20_validation.log", "Episode_20"),
        ("episode_775_mc_car25_aggressive.log", "Episode_775"),
    ]

    for log_file, checkpoint_name in validation_files:
        if (visualizer.results_dir / log_file).exists():
            visualizer.visualize_checkpoint(log_file, checkpoint_name)
        else:
            logger.info(f"Skipping {log_file} - file not found")

    logger.info("Visualization complete! Check validation_results/graphs/ for outputs")

if __name__ == "__main__":
    main()