#!/usr/bin/env python3
"""
Session-based validation for SWT checkpoints
Runs Monte Carlo simulations on 6-hour (360 bar) trading sessions
"""

import argparse
import logging
import time
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from fixed_checkpoint_loader import load_checkpoint_with_proper_config
from swt_core.sqn_calculator import SQNCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SESSION_LENGTH = 360  # 6 hours = 360 1-minute bars
MIN_BARS_REQUIRED = SESSION_LENGTH * 2  # Need at least 2 sessions worth of data


class SessionValidator:
    """Validates checkpoints using realistic 6-hour trading sessions"""

    def __init__(self, wst_file: str, csv_file: str):
        """
        Initialize validator with market data

        Args:
            wst_file: Path to precomputed WST features
            csv_file: Path to price data CSV
        """
        logger.info(f"Loading pre-computed WST from {wst_file}")
        with h5py.File(wst_file, 'r') as f:
            self.wst_features = f['wst_features'][:]
            self.wst_indices = f['indices'][:]

        logger.info(f"WST features shape: {self.wst_features.shape}")

        # Load price data
        logger.info(f"Loading price data from {csv_file}")
        self.price_data = pd.read_csv(csv_file)

        # Use last 20% for testing (matching training split)
        split_idx = int(len(self.wst_features) * 0.8)
        self.test_wst = self.wst_features[split_idx:]
        self.test_prices = self.price_data.iloc[split_idx:]

        logger.info(f"Test data: {len(self.test_wst):,} bars = {len(self.test_wst)/60:.1f} hours")
        logger.info(f"Available for {len(self.test_wst)//SESSION_LENGTH:,} complete sessions")

        # Initialize SQN calculator
        self.sqn_calculator = SQNCalculator()

    def expand_wst_features(self, wst_16: np.ndarray) -> np.ndarray:
        """Expand 16-dim WST to 128-dim for network input"""
        # Create 128-dim feature vector
        expanded = np.zeros(128, dtype=np.float32)

        # Fill first 16 dimensions with WST
        expanded[:16] = wst_16

        # Add time-based features
        expanded[16:32] = np.sin(np.linspace(0, np.pi, 16)) * wst_16
        expanded[32:48] = np.cos(np.linspace(0, np.pi, 16)) * wst_16

        # Add rolling statistics
        expanded[48:64] = np.roll(wst_16, 1)
        expanded[64:80] = np.roll(wst_16, -1)

        # Add interactions
        expanded[80:96] = wst_16 ** 2
        expanded[96:112] = np.sqrt(np.abs(wst_16))

        # Add constants for remaining
        expanded[112:] = 0.1

        return expanded

    def run_single_session(self, network, start_idx: int) -> Dict:
        """
        Run a single 6-hour trading session

        Args:
            network: The trading network
            start_idx: Starting index in test data

        Returns:
            Session results dictionary
        """
        # Initialize session
        session_wst = self.test_wst[start_idx:start_idx + SESSION_LENGTH]
        session_prices = self.test_prices.iloc[start_idx:start_idx + SESSION_LENGTH]

        trades = []
        position = None
        total_pnl = 0

        # Run session
        for i in range(SESSION_LENGTH):
            # Prepare observation
            wst_expanded = self.expand_wst_features(session_wst[i])

            # Add position features (9 dims)
            if position is None:
                position_features = np.zeros(9)
            else:
                position_features = np.array([
                    1.0 if position['side'] == 'BUY' else -1.0,
                    position['size'],
                    position['unrealized_pnl'] / 100,
                    position['bars_held'] / 10,
                    position['max_pnl'] / 100,
                    position['min_pnl'] / 100,
                    1.0,  # has_position
                    0.0,  # reserved
                    0.0   # reserved
                ])

            # Combine features
            observation = np.concatenate([wst_expanded, position_features])
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

            # Get action from network
            with torch.no_grad():
                inference = network.initial_inference(obs_tensor)
                policy_logits = inference['policy_logits']
                action = policy_logits.argmax().item()

            # Execute action (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)
            current_price = session_prices.iloc[i]['Close']

            if action == 1 and position is None:  # BUY
                position = {
                    'side': 'BUY',
                    'size': 1.0,
                    'entry_price': current_price,
                    'entry_bar': i,
                    'unrealized_pnl': 0,
                    'max_pnl': 0,
                    'min_pnl': 0,
                    'bars_held': 0
                }

            elif action == 2 and position is None:  # SELL
                position = {
                    'side': 'SELL',
                    'size': 1.0,
                    'entry_price': current_price,
                    'entry_bar': i,
                    'unrealized_pnl': 0,
                    'max_pnl': 0,
                    'min_pnl': 0,
                    'bars_held': 0
                }

            elif action == 3 and position is not None:  # CLOSE
                # Calculate PnL
                if position['side'] == 'BUY':
                    pnl = (current_price - position['entry_price']) * 100  # pips
                else:
                    pnl = (position['entry_price'] - current_price) * 100

                trades.append({
                    'pnl': pnl,
                    'duration': position['bars_held'],
                    'side': position['side']
                })
                total_pnl += pnl
                position = None

            # Update position if held
            if position is not None:
                position['bars_held'] += 1
                if position['side'] == 'BUY':
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * 100
                else:
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * 100
                position['max_pnl'] = max(position['max_pnl'], position['unrealized_pnl'])
                position['min_pnl'] = min(position['min_pnl'], position['unrealized_pnl'])

        # Force close any open position at session end (but don't count in stats)
        if position is not None:
            logger.debug(f"Force-closed position at session end (not counted)")

        # Calculate session metrics
        if trades:
            pnl_values = [t['pnl'] for t in trades]
            sqn_result = self.sqn_calculator.calculate_sqn(pnl_values)
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100
        else:
            sqn_result = self.sqn_calculator.calculate_sqn([])
            win_rate = 0

        return {
            'total_pnl': total_pnl,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'sqn': sqn_result.sqn,
            'sqn_class': sqn_result.classification,
            'trades': trades
        }

    def run_monte_carlo(self, network, num_sessions: int = 100) -> Dict:
        """
        Run Monte Carlo simulation with random 6-hour sessions

        Args:
            network: The trading network
            num_sessions: Number of random sessions to simulate

        Returns:
            Aggregated results from all sessions
        """
        logger.info(f"Running {num_sessions} Monte Carlo sessions...")

        # Maximum starting position for sessions
        max_start = len(self.test_wst) - SESSION_LENGTH

        all_results = []
        all_trades = []

        for i in range(num_sessions):
            # Random starting point
            start_idx = np.random.randint(0, max_start)

            # Run session
            session_result = self.run_single_session(network, start_idx)
            all_results.append(session_result)
            all_trades.extend(session_result['trades'])

            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i+1}/{num_sessions} sessions")

        # Aggregate results
        total_pnl = sum(r['total_pnl'] for r in all_results)
        total_trades = sum(r['num_trades'] for r in all_results)
        avg_pnl_per_session = total_pnl / num_sessions
        avg_trades_per_session = total_trades / num_sessions

        # Calculate overall metrics
        if all_trades:
            all_pnl = [t['pnl'] for t in all_trades]
            overall_sqn = self.sqn_calculator.calculate_sqn(all_pnl)
            overall_win_rate = sum(1 for t in all_trades if t['pnl'] > 0) / len(all_trades) * 100

            # Calculate per-session statistics
            session_pnls = [r['total_pnl'] for r in all_results]
            sharpe_ratio = np.mean(session_pnls) / (np.std(session_pnls) + 1e-8)
            max_drawdown = min(session_pnls)
            best_session = max(session_pnls)
        else:
            overall_sqn = self.sqn_calculator.calculate_sqn([])
            overall_win_rate = 0
            sharpe_ratio = 0
            max_drawdown = 0
            best_session = 0

        return {
            'num_sessions': num_sessions,
            'total_trades': total_trades,
            'avg_trades_per_session': avg_trades_per_session,
            'avg_pnl_per_session': avg_pnl_per_session,
            'total_pnl': total_pnl,
            'win_rate': overall_win_rate,
            'sqn': overall_sqn.sqn,
            'sqn_classification': overall_sqn.classification,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'best_session': best_session,
            'sessions_with_trades': sum(1 for r in all_results if r['num_trades'] > 0)
        }


def validate_checkpoint(checkpoint_path: str, wst_file: str, csv_file: str, num_sessions: int = 100):
    """
    Validate a single checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        wst_file: Path to WST features
        csv_file: Path to price data
        num_sessions: Number of Monte Carlo sessions
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {Path(checkpoint_path).name}")
    logger.info(f"{'='*60}")

    # Load checkpoint and network
    checkpoint_data = load_checkpoint_with_proper_config(checkpoint_path)
    network = checkpoint_data['networks']
    network.eval()

    # Initialize validator
    validator = SessionValidator(wst_file, csv_file)

    # Run Monte Carlo validation
    start_time = time.time()
    results = validator.run_monte_carlo(network, num_sessions)
    duration = time.time() - start_time

    # Display results
    print(f"\n📊 {Path(checkpoint_path).stem} Session-Based Validation Results:")
    print(f"  ⏱️  Sessions: {results['num_sessions']}")
    print(f"  ✅ Total Trades: {results['total_trades']}")
    print(f"  ✅ Avg Trades/Session: {results['avg_trades_per_session']:.1f}")
    print(f"  ✅ Avg PnL/Session: {results['avg_pnl_per_session']:.2f} pips")
    print(f"  ✅ Win Rate: {results['win_rate']:.2f}%")
    print(f"  ✅ SQN: {results['sqn']:.2f} ({results['sqn_classification']})")
    print(f"  ✅ Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  ✅ Worst Session: {results['max_drawdown']:.2f} pips")
    print(f"  ✅ Best Session: {results['best_session']:.2f} pips")
    print(f"  ⏱️  Validation Time: {duration:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='Session-based checkpoint validation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--wst-file', type=str, required=True, help='Path to WST file')
    parser.add_argument('--csv-file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--sessions', type=int, default=100, help='Number of sessions')

    args = parser.parse_args()

    validate_checkpoint(
        args.checkpoint,
        args.wst_file,
        args.csv_file,
        args.sessions
    )


if __name__ == '__main__':
    main()