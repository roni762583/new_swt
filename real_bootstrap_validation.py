#!/usr/bin/env python3
"""
Real bootstrap Monte Carlo validation using actual trading environment
Collects actual trades with proper spread costs and creates spaghetti plots
"""

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/workspace')

from swt_environments.swt_forex_env import SWTForexEnvironment as SWTForexEnv, SWTAction, SWTPositionState
from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config
from swt_features.feature_processor import FeatureProcessor
from swt_features.precomputed_wst_loader import PrecomputedWSTLoader
from swt_core.config_manager import ConfigManager
from swt_core.types import PositionState, PositionType

def collect_real_trades(checkpoint_path, num_sessions=20):
    """Collect real trades from validation sessions using actual trading environment"""

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load configuration
    config_manager = ConfigManager('config')
    config_manager.load_config()
    config = config_manager.merged_config

    # Create precomputed WST loader
    wst_loader = PrecomputedWSTLoader(
        'precomputed_wst/GBPJPY_WST_3.5years_streaming.h5',
        cache_size=10000
    )

    # Create feature processor with precomputed WST loader
    feature_processor = FeatureProcessor(
        config_manager,  # Pass ConfigManager, not just config
        precomputed_loader=wst_loader
    )

    # Load the model
    checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
    network = checkpoint_data['networks']
    network.eval()

    # Create environment with proper spread costs
    env = SWTForexEnv(
        data_path='data/GBPJPY_M1_3.5years_20250912.csv',
        config_dict=config  # Pass the merged config dictionary
    )

    print(f"Collecting trades from {num_sessions} random 6-hour sessions...")
    all_trades = []

    with torch.no_grad():
        for session in range(num_sessions):
            print(f"  Session {session+1}/{num_sessions}...", end='', flush=True)

            # Reset environment for new session
            observation = env.reset()
            done = False
            session_trades = []
            steps = 0

            while not done and steps < 360:  # 6-hour sessions
                # Get current market price and position state from environment
                market_prices = observation['market_prices'] if isinstance(observation, dict) else observation[:256]
                current_price = market_prices[-1]  # Last price in the series
                position_state = env.position  # Get actual position state from environment

                # For precomputed WST, use the current step as window index
                window_index = env.current_step

                # Convert SWTPositionState to PositionState for feature processor
                if position_state.is_long:
                    pos_type = PositionType.LONG
                elif position_state.is_short:
                    pos_type = PositionType.SHORT
                else:
                    pos_type = PositionType.FLAT

                # Create immutable PositionState
                feature_position_state = PositionState(
                    position_type=pos_type,
                    entry_price=position_state.entry_price,
                    unrealized_pnl_pips=position_state.unrealized_pnl_pips,
                    duration_minutes=position_state.duration_bars  # Using bars as minutes proxy
                )

                # Process observation with position state and current price
                processed_obs = feature_processor.process_observation(
                    position_state=feature_position_state,
                    current_price=current_price,
                    window_index=window_index  # Pass window index for precomputed WST
                )

                # Get combined features (128 WST + 9 position)
                wst_features = processed_obs.combined_features  # Already 137 features
                obs_tensor = torch.FloatTensor(wst_features).unsqueeze(0)

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
                        trade_pips = trade.pnl_pips  # Actual pips after 4 pip spread
                        session_trades.append(trade_pips)
                        all_trades.append(trade_pips)

            print(f" ✓ ({len(session_trades)} trades, {sum(session_trades):.1f} pips)")

    return np.array(all_trades)

def bootstrap_with_trajectories(trade_pool, num_bootstrap_samples=100):
    """Perform bootstrap stress testing with trajectory tracking"""

    if len(trade_pool) == 0:
        print("❌ No trades in pool!")
        return None

    print(f"\n🎲 Running {num_bootstrap_samples} bootstrap samples...")
    print(f"   Trade pool size: {len(trade_pool)} trades")
    print(f"   Pool mean: {np.mean(trade_pool):.2f} pips, std: {np.std(trade_pool):.2f} pips")

    # Typical session has 20-30 trades
    trades_per_session = min(30, len(trade_pool))

    results = {
        'original': {'final': [], 'trajectories': []},
        'drop_10pct': {'final': [], 'trajectories': []},
        'drop_20pct_tail': {'final': [], 'trajectories': []},
        'resample_150pct': {'final': [], 'trajectories': []},
        'adverse_selection': {'final': [], 'trajectories': []},
        'early_stop_80pct': {'final': [], 'trajectories': []}
    }

    for i in range(num_bootstrap_samples):
        if i % 20 == 0:
            print(f"   Bootstrap sample {i}/{num_bootstrap_samples}...")

        # 1. Original bootstrap (with replacement)
        sample = np.random.choice(trade_pool, size=trades_per_session, replace=True)
        trajectory = np.cumsum(sample)
        results['original']['trajectories'].append(trajectory)
        results['original']['final'].append(trajectory[-1])

        # 2. Drop 10% randomly (robustness test)
        keep_size = int(trades_per_session * 0.9)
        sample = np.random.choice(trade_pool, size=keep_size, replace=True)
        trajectory = np.cumsum(sample)
        results['drop_10pct']['trajectories'].append(trajectory)
        results['drop_10pct']['final'].append(trajectory[-1])

        # 3. Drop last 20% (early stopping simulation)
        keep_size = int(trades_per_session * 0.8)
        sample = np.random.choice(trade_pool, size=keep_size, replace=True)
        trajectory = np.cumsum(sample)
        results['drop_20pct_tail']['trajectories'].append(trajectory)
        results['drop_20pct_tail']['final'].append(trajectory[-1])

        # 4. Oversample 150% (what if we traded more?)
        sample_size = int(trades_per_session * 1.5)
        sample = np.random.choice(trade_pool, size=sample_size, replace=True)
        trajectory = np.cumsum(sample)
        results['resample_150pct']['trajectories'].append(trajectory)
        results['resample_150pct']['final'].append(trajectory[-1])

        # 5. Adverse selection (sample more from losses)
        losses = trade_pool[trade_pool < 0]
        wins = trade_pool[trade_pool >= 0]
        if len(losses) > 0 and len(wins) > 0:
            # 70% losses, 30% wins (adverse conditions)
            n_losses = int(trades_per_session * 0.7)
            n_wins = trades_per_session - n_losses
            adverse_sample = np.concatenate([
                np.random.choice(losses, size=min(n_losses, len(losses)*10), replace=True),
                np.random.choice(wins, size=min(n_wins, len(wins)*10), replace=True)
            ])[:trades_per_session]
            trajectory = np.cumsum(adverse_sample)
        else:
            sample = np.random.choice(trade_pool, size=trades_per_session, replace=True)
            trajectory = np.cumsum(sample)
        results['adverse_selection']['trajectories'].append(trajectory)
        results['adverse_selection']['final'].append(trajectory[-1])

        # 6. Early stop at 80% (what if we stopped early?)
        sample = np.random.choice(trade_pool, size=trades_per_session, replace=True)
        full_trajectory = np.cumsum(sample)
        trajectory = full_trajectory[:int(trades_per_session * 0.8)]
        results['early_stop_80pct']['trajectories'].append(trajectory)
        results['early_stop_80pct']['final'].append(trajectory[-1] if len(trajectory) > 0 else 0)

    return results

def create_spaghetti_stress_report(results, trade_pool, output_dir, checkpoint_name):
    """Create comprehensive stress test report with spaghetti plots"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # Create grid spec for custom layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)

    scenarios = [
        ('original', 'Original Bootstrap', 'blue'),
        ('drop_10pct', 'Drop 10% Random', 'orange'),
        ('drop_20pct_tail', 'Drop 20% Tail', 'red'),
        ('resample_150pct', 'Oversample 150%', 'green'),
        ('adverse_selection', 'Adverse Selection', 'darkred'),
        ('early_stop_80pct', 'Early Stop 80%', 'purple')
    ]

    fig.suptitle(f'Real Bootstrap Monte Carlo with Trajectory Plots (Pips after 4 pip spread)\n'
                 f'{checkpoint_name} - Pool: {len(trade_pool)} trades, '
                 f'{len(results["original"]["trajectories"])} bootstrap samples',
                 fontsize=16, fontweight='bold')

    # Plot spaghetti plots for each scenario
    for idx, (key, title, color) in enumerate(scenarios):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])

        trajectories = results[key]['trajectories']
        finals = results[key]['final']

        # Plot all trajectories with transparency
        for i, traj in enumerate(trajectories):
            alpha = 0.1 if len(trajectories) > 50 else 0.2
            ax.plot(range(len(traj)), traj, color=color, alpha=alpha, linewidth=0.5)

        # Plot percentile bands
        if len(trajectories) > 0:
            # Find max length for padding
            max_len = max(len(t) for t in trajectories)

            # Pad trajectories to same length
            padded = []
            for traj in trajectories:
                if len(traj) < max_len:
                    padded_traj = np.pad(traj, (0, max_len - len(traj)),
                                        mode='constant', constant_values=traj[-1])
                else:
                    padded_traj = traj
                padded.append(padded_traj)

            padded_array = np.array(padded)

            # Calculate and plot percentiles
            median = np.median(padded_array, axis=0)
            p25 = np.percentile(padded_array, 25, axis=0)
            p75 = np.percentile(padded_array, 75, axis=0)

            ax.plot(range(max_len), median, color='black', linewidth=2,
                   label=f'Median: {np.median(finals):.1f} pips')
            ax.fill_between(range(max_len), p25, p75, alpha=0.3, color=color)

        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.2)

        # Statistics
        positive_pct = sum(1 for x in finals if x > 0) / len(finals) * 100 if finals else 0

        ax.set_title(f'{title}\nWin Rate: {positive_pct:.1f}%', fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative Pips')
        ax.legend(loc='upper left', fontsize=9)

    # Combined spaghetti plot (bottom left)
    ax = fig.add_subplot(gs[2, :])

    # Plot all scenarios overlaid with different colors
    for key, title, color in scenarios:
        trajectories = results[key]['trajectories']
        for traj in trajectories[:20]:  # Limit to 20 trajectories per scenario
            ax.plot(range(len(traj)), traj, color=color, alpha=0.3, linewidth=0.8)

    # Create custom legend
    legend_elements = [Patch(facecolor=color, alpha=0.5, label=title)
                      for key, title, color in scenarios]
    ax.legend(handles=legend_elements, loc='upper left', ncol=3, fontsize=10)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.2)
    ax.set_title('All Scenarios Overlaid - Stress Test Comparison', fontweight='bold', fontsize=12)
    ax.set_xlabel('Trade Number', fontsize=11)
    ax.set_ylabel('Cumulative Pips (after spread)', fontsize=11)

    # Summary statistics table (bottom)
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Create table data
    table_data = [['Scenario', 'Median', 'Mean', 'Std Dev', 'Win Rate', '5% VaR', '95% CI']]

    for key, title, _ in scenarios:
        finals = results[key]['final']
        if finals:
            row = [
                title,
                f"{np.median(finals):.1f} pips",
                f"{np.mean(finals):.1f} pips",
                f"{np.std(finals):.1f} pips",
                f"{sum(1 for x in finals if x > 0) / len(finals) * 100:.1f}%",
                f"{np.percentile(finals, 5):.1f} pips",
                f"{np.percentile(finals, 95):.1f} pips"
            ]
            table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                    colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code rows
    for i in range(1, len(table_data)):
        median_val = float(table_data[i][1].split()[0])
        if median_val > 0:
            color = '#90EE90'  # Light green
        else:
            color = '#FFB6C1'  # Light red
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)

    plt.tight_layout()

    # Save plots
    plot_file = output_dir / f"real_bootstrap_spaghetti_{timestamp}_{checkpoint_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved real bootstrap spaghetti plots to {plot_file}")

    pdf_file = output_dir / f"real_bootstrap_spaghetti_{timestamp}_{checkpoint_name}.pdf"
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"📄 Saved PDF report to {pdf_file}")

    plt.close()

    # Save JSON data
    json_data = {
        'checkpoint': checkpoint_name,
        'timestamp': timestamp,
        'trade_pool_stats': {
            'count': len(trade_pool),
            'mean': float(np.mean(trade_pool)),
            'std': float(np.std(trade_pool)),
            'min': float(np.min(trade_pool)),
            'max': float(np.max(trade_pool)),
            'positive_rate': float(sum(1 for t in trade_pool if t > 0) / len(trade_pool))
        },
        'bootstrap_results': {}
    }

    for key in results:
        finals = results[key]['final']
        json_data['bootstrap_results'][key] = {
            'mean': float(np.mean(finals)),
            'median': float(np.median(finals)),
            'std': float(np.std(finals)),
            'ci_5pct': float(np.percentile(finals, 5)),
            'ci_95pct': float(np.percentile(finals, 95)),
            'positive_rate': float(sum(1 for x in finals if x > 0) / len(finals))
        }

    json_file = output_dir / f"real_bootstrap_spaghetti_{timestamp}_{checkpoint_name}.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"💾 Saved real trade results to {json_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Real bootstrap validation with actual trades')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--sessions', type=int, default=20,
                       help='Number of validation sessions to collect trades from')
    parser.add_argument('--bootstrap-samples', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--output-dir', default='validation_results', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"Real Bootstrap Monte Carlo Validation")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Using actual trading environment with 4 pip spread")
    print(f"{'='*60}\n")

    # Step 1: Collect real trades
    trade_pool = collect_real_trades(args.checkpoint, args.sessions)

    if len(trade_pool) == 0:
        print("❌ Failed to collect trades!")
        sys.exit(1)

    # Step 2: Run bootstrap with trajectory tracking
    results = bootstrap_with_trajectories(trade_pool, args.bootstrap_samples)

    if results:
        # Step 3: Create spaghetti report
        create_spaghetti_stress_report(results, trade_pool, output_dir, checkpoint_name)

        # Print summary
        print(f"\n{'='*60}")
        print("REAL TRADE STRESS TEST SUMMARY")
        print(f"{'='*60}")

        print(f"Trade pool: {len(trade_pool)} trades")
        print(f"Mean: {np.mean(trade_pool):.2f} pips, Std: {np.std(trade_pool):.2f} pips")
        print(f"Win rate: {sum(1 for t in trade_pool if t > 0) / len(trade_pool) * 100:.1f}%\n")

        for key in ['original', 'drop_10pct', 'adverse_selection']:
            finals = results[key]['final']
            median = np.median(finals)
            positive_rate = sum(1 for x in finals if x > 0) / len(finals) * 100
            print(f"{key:20s}: {median:>7.1f} pips median, {positive_rate:>5.1f}% profitable")

        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()