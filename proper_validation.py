#!/usr/bin/env python3
"""
Proper validation using the actual SWT trading environment with spread costs
Reports in pips after costs, not percentages
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/workspace')

from swt_environments.swt_forex_env import SWTForexEnvironment as SWTForexEnv, SWTAction
from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config
from swt_features.feature_processor import FeatureProcessor
from swt_features.precomputed_wst_loader import PrecomputedWSTLoader
from swt_core.config_manager import ConfigManager

def run_validation_with_real_environment(checkpoint_path, num_episodes=20):
    """Run validation using the actual trading environment with spread costs"""

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load configuration
    config_manager = ConfigManager('config')
    config_manager.load_config()

    # Load precomputed WST
    wst_loader = PrecomputedWSTLoader(
        'precomputed_wst/GBPJPY_WST_3.5years_streaming.h5',
        cache_size=10000
    )

    # Create feature processor with precomputed WST
    feature_processor = FeatureProcessor(
        config_manager,
        precomputed_loader=wst_loader
    )

    # Load the model
    checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
    network = checkpoint_data['networks']
    network.eval()

    # Create environment with proper spread costs
    env = SWTForexEnv(
        config=config_manager.config,
        csv_file='data/GBPJPY_M1_3.5years_20250912.csv',
        feature_processor=feature_processor,
        test_mode=True  # Use test data
    )

    all_episode_results = []
    all_trades = []

    with torch.no_grad():
        for episode in range(num_episodes):
            print(f"  Episode {episode+1}/{num_episodes}...", end='', flush=True)

            # Reset environment for new episode
            observation = env.reset()
            done = False
            episode_trades = []
            episode_pnl = 0
            steps = 0

            while not done and steps < 360:  # 6-hour sessions
                # Process observation
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

                # Get model prediction
                inference_result = network.initial_inference(obs_tensor)
                policy_logits = inference_result['policy_logits']

                # Select action (greedy for validation)
                action = policy_logits.argmax().item()

                # Step environment - THIS PROPERLY HANDLES SPREAD COSTS
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Collect completed trades
                if 'completed_trades' in info and info['completed_trades']:
                    for trade in info['completed_trades']:
                        episode_trades.append(trade)
                        all_trades.append(trade.pnl_pips)  # Actual pips after spread
                        episode_pnl += trade.pnl_pips

            all_episode_results.append({
                'episode': episode,
                'total_pnl_pips': episode_pnl,
                'num_trades': len(episode_trades),
                'avg_pips_per_trade': episode_pnl / len(episode_trades) if episode_trades else 0
            })

            print(f" {len(episode_trades)} trades, {episode_pnl:.1f} pips")

    return all_episode_results, all_trades

def create_pip_based_report(episodes, all_trades, output_dir, checkpoint_name):
    """Create report with actual pip metrics after costs"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to numpy array
    trades_array = np.array(all_trades) if all_trades else np.array([])

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'PROPER Validation Results (Pips After Spread Costs)\n{checkpoint_name}', fontsize=14)

    # 1. Trade PnL Distribution (Pips)
    if len(trades_array) > 0:
        ax = axes[0, 0]
        ax.hist(trades_array, bins=30, edgecolor='black', alpha=0.7, color='blue')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Breakeven')
        ax.axvline(x=-4, color='orange', linestyle='--', alpha=0.3, label='Spread cost')
        ax.set_xlabel('PnL per Trade (Pips)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Trade Distribution\n{len(trades_array)} trades')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Episode PnL Box Plot
    episode_pnls = [ep['total_pnl_pips'] for ep in episodes]
    if episode_pnls:
        ax = axes[0, 1]
        bp = ax.boxplot(episode_pnls, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        ax.set_ylabel('Total PnL (Pips)')
        ax.set_title(f'Episode PnL Distribution\n{len(episodes)} episodes')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 3. Cumulative PnL
    if len(trades_array) > 0:
        ax = axes[0, 2]
        cumsum = np.cumsum(trades_array)
        ax.plot(cumsum, color='green' if cumsum[-1] > 0 else 'red')
        ax.fill_between(range(len(cumsum)), 0, cumsum, alpha=0.3)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL (Pips)')
        ax.set_title(f'Cumulative Performance\nFinal: {cumsum[-1]:.1f} pips')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 4. Win/Loss Analysis
    if len(trades_array) > 0:
        ax = axes[1, 0]
        wins = trades_array[trades_array > 0]
        losses = trades_array[trades_array <= 0]

        labels = ['Wins', 'Losses']
        sizes = [len(wins), len(losses)]
        colors = ['green', 'red']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Win Rate: {len(wins)/len(trades_array)*100:.1f}%')

    # 5. Metrics by Episode
    ax = axes[1, 1]
    if episodes:
        episode_nums = range(1, len(episodes) + 1)
        episode_pnls = [ep['total_pnl_pips'] for ep in episodes]
        colors = ['green' if pnl > 0 else 'red' for pnl in episode_pnls]
        ax.bar(episode_nums, episode_pnls, color=colors, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('PnL (Pips)')
        ax.set_title('PnL by Episode')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 6. Statistics Table
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = "REAL Trading Statistics (Pips)\n" + "="*35 + "\n\n"

    if len(trades_array) > 0:
        stats_text += f"Total Trades:     {len(trades_array):>10d}\n"
        stats_text += f"Total PnL:        {np.sum(trades_array):>10.1f} pips\n"
        stats_text += f"Mean per Trade:   {np.mean(trades_array):>10.2f} pips\n"
        stats_text += f"Median per Trade: {np.median(trades_array):>10.2f} pips\n"
        stats_text += f"Std Dev:          {np.std(trades_array):>10.2f} pips\n"
        stats_text += f"Win Rate:         {sum(1 for t in trades_array if t > 0)/len(trades_array)*100:>10.1f}%\n"
        stats_text += f"\nSpread Cost:      4.0 pips/trade\n"
        stats_text += f"Total Spread:     {len(trades_array) * 4.0:>10.1f} pips\n"
        stats_text += f"Gross Before Cost:{np.sum(trades_array) + len(trades_array)*4:>10.1f} pips\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plots
    plot_file = output_dir / f"proper_validation_{timestamp}_{checkpoint_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved proper validation plot to {plot_file}")

    pdf_file = output_dir / f"proper_validation_{timestamp}_{checkpoint_name}.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📄 Saved PDF report to {pdf_file}")

    plt.close()

    # Save JSON with pip metrics
    json_data = {
        'checkpoint': checkpoint_name,
        'timestamp': timestamp,
        'spread_pips': 4.0,
        'summary': {
            'total_trades': len(trades_array),
            'total_pnl_pips': float(np.sum(trades_array)) if len(trades_array) > 0 else 0,
            'mean_pips_per_trade': float(np.mean(trades_array)) if len(trades_array) > 0 else 0,
            'median_pips_per_trade': float(np.median(trades_array)) if len(trades_array) > 0 else 0,
            'std_pips': float(np.std(trades_array)) if len(trades_array) > 0 else 0,
            'win_rate': float(sum(1 for t in trades_array if t > 0) / len(trades_array)) if len(trades_array) > 0 else 0,
            'total_spread_cost': len(trades_array) * 4.0,
            'gross_before_spread': float(np.sum(trades_array) + len(trades_array) * 4.0) if len(trades_array) > 0 else 0
        },
        'episodes': episodes
    }

    json_file = output_dir / f"proper_validation_{timestamp}_{checkpoint_name}.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"💾 Saved pip-based results to {json_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Proper validation with real trading environment')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--output-dir', default='validation_results', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"PROPER Validation with Real Trading Environment")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Spread Cost: 4 pips per trade")
    print(f"{'='*60}\n")

    # Run validation
    episodes, all_trades = run_validation_with_real_environment(
        args.checkpoint,
        args.episodes
    )

    # Create report
    create_pip_based_report(episodes, all_trades, output_dir, checkpoint_name)

    # Print summary
    if all_trades:
        total_pips = sum(all_trades)
        avg_pips = np.mean(all_trades)
        win_rate = sum(1 for t in all_trades if t > 0) / len(all_trades) * 100

        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY (After Spread Costs)")
        print(f"{'='*60}")
        print(f"Total Trades:    {len(all_trades)}")
        print(f"Total PnL:       {total_pips:.1f} pips")
        print(f"Avg per Trade:   {avg_pips:.2f} pips")
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Spread Paid:     {len(all_trades) * 4.0:.1f} pips")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()