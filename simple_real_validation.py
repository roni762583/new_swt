#!/usr/bin/env python3
"""
Simple validation using actual SWTForexEnvironment - NO synthetic data
Reports actual pip results after spread costs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime
import h5py

# Import the actual trading environment
from swt_environments.swt_forex_env import SWTForexEnvironment as SWTForexEnv
from swt_validation.fixed_checkpoint_loader import load_checkpoint_with_proper_config
from swt_core.config_manager import ConfigManager
from swt_features.precomputed_wst_loader import PrecomputedWSTLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_checkpoint_simple(checkpoint_path: str, csv_file: str, wst_file: str, num_runs: int = 10):
    """
    Validate checkpoint using raw environment with precomputed WST features

    Returns actual pip results after 4 pip spread costs
    """
    print(f"\n{'='*60}")
    print(f"SIMPLE REAL VALIDATION")
    print(f"Checkpoint: {Path(checkpoint_path).name}")
    print(f"Sessions: {num_runs} random 6-hour periods")
    print(f"{'='*60}\n")

    # Load configuration
    config_manager = ConfigManager('config')
    config_manager.load_config()
    config = config_manager.merged_config

    # Load the model
    print("Loading checkpoint...")
    checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
    network = checkpoint_data['networks']
    network.eval()

    # Load precomputed WST features
    print("Loading precomputed WST features...")
    wst_loader = PrecomputedWSTLoader(wst_file, cache_size=10000)

    # Create environment with actual spread costs
    print("Creating trading environment with 4 pip spread...")
    # Convert config to dict if it's an object
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    env = SWTForexEnv(
        data_path=csv_file,
        config_dict=config_dict
    )

    all_trades = []
    session_results = []

    with torch.no_grad():
        for session in range(num_runs):
            print(f"Session {session+1}/{num_runs}...", end='', flush=True)

            # Reset for new 6-hour session
            observation, _ = env.reset()  # Unpack tuple, ignore info

            trades_this_session = []
            steps = 0
            done = False

            while not done and steps < 360:  # 6 hours = 360 minutes
                # Get raw observation from environment
                if isinstance(observation, dict):
                    position_data = observation.get('position_features', np.zeros(9))
                else:
                    # Fallback if not dict
                    position_data = np.zeros(9)

                # Get WST features for current window
                window_index = env.current_step
                wst_features = wst_loader.get_wst_features(window_index)

                # Convert torch tensor to numpy if needed
                if hasattr(wst_features, 'numpy'):
                    wst_features = wst_features.numpy()

                # Combine WST features with position features (128 + 9 = 137)
                combined_features = np.concatenate([wst_features, position_data])

                # Convert to tensor
                obs_tensor = torch.FloatTensor(combined_features).unsqueeze(0)

                # Get model prediction
                inference_result = network.initial_inference(obs_tensor)
                policy_logits = inference_result['policy_logits']

                # Select action (greedy for validation)
                action = policy_logits.argmax().item()

                # Step environment - THIS APPLIES 4 PIP SPREAD COST
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Collect completed trades with ACTUAL pip results after spread
                if 'completed_trades' in info and info['completed_trades']:
                    for trade in info['completed_trades']:
                        pip_result = trade.pnl_pips  # Already includes -4 pip spread
                        trades_this_session.append(pip_result)
                        all_trades.append(pip_result)

            session_pips = sum(trades_this_session) if trades_this_session else 0
            print(f" ✓ ({len(trades_this_session)} trades, {session_pips:.1f} pips)")

            session_results.append({
                'trades': len(trades_this_session),
                'total_pips': session_pips
            })

    # Calculate statistics from REAL trades
    if all_trades:
        total_pips = sum(all_trades)
        avg_pips = np.mean(all_trades)
        win_rate = sum(1 for t in all_trades if t > 0) / len(all_trades)

        print(f"\n{'='*60}")
        print(f"VALIDATION RESULTS (ACTUAL PIPS AFTER SPREAD)")
        print(f"{'='*60}")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Total Pips: {total_pips:.1f}")
        print(f"Avg Pips/Trade: {avg_pips:.2f}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Best Session: {max(s['total_pips'] for s in session_results):.1f} pips")
        print(f"Worst Session: {min(s['total_pips'] for s in session_results):.1f} pips")
        print(f"{'='*60}\n")

        return {
            'total_trades': len(all_trades),
            'total_pips': total_pips,
            'avg_pips': avg_pips,
            'win_rate': win_rate,
            'trades_list': all_trades
        }
    else:
        print("\n⚠️ No trades executed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Simple validation with real environment")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--csv-file", default="data/GBPJPY_M1_3.5years_20250912.csv",
                       help="CSV file with price data")
    parser.add_argument("--wst-file", default="precomputed_wst/GBPJPY_WST_3.5years_streaming.h5",
                       help="Precomputed WST features file")
    parser.add_argument("--runs", type=int, default=10,
                       help="Number of validation runs")

    args = parser.parse_args()

    start_time = datetime.now()

    results = validate_checkpoint_simple(
        args.checkpoint,
        args.csv_file,
        args.wst_file,
        args.runs
    )

    duration = (datetime.now() - start_time).total_seconds()
    print(f"Validation completed in {duration:.1f} seconds")

    # Run bootstrap analysis if we have results
    if results and results['trades_list']:
        trades = np.array(results['trades_list'])

        print(f"\n{'='*60}")
        print("BOOTSTRAP ANALYSIS (1000 samples)")
        print(f"{'='*60}")

        bootstrap_results = []
        for _ in range(1000):
            sample = np.random.choice(trades, size=len(trades), replace=True)
            bootstrap_results.append(sum(sample))

        print(f"Median Total: {np.median(bootstrap_results):.1f} pips")
        print(f"5% Percentile: {np.percentile(bootstrap_results, 5):.1f} pips")
        print(f"95% Percentile: {np.percentile(bootstrap_results, 95):.1f} pips")
        print(f"Probability > 0: {sum(1 for r in bootstrap_results if r > 0) / len(bootstrap_results):.1%}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()