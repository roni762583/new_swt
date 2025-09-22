#!/usr/bin/env python3
"""
Training diagnostics to verify learning improvements.
Tracks key metrics recommended in the improvement plan.
"""

import numpy as np
import torch
import duckdb
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticMetrics:
    """Container for diagnostic metrics."""
    episode: int
    timestamp: float

    # Core metrics
    expectancy: float
    win_rate: float
    trade_ratio: float
    avg_trade_length: float

    # Loss metrics
    total_loss: float
    policy_loss: float
    value_loss: float
    outcome_loss: float

    # MCTS metrics
    kl_policy_mcts: float  # KL divergence between policy and MCTS visits
    avg_visit_entropy: float  # Entropy of visit distribution
    best_child_fraction: float  # Fraction of visits to best child

    # Value calibration
    value_correlation: float  # Correlation between predicted and realized returns
    value_mse: float  # MSE between predicted and realized returns

    # Buffer metrics
    buffer_size: int
    success_buffer_fraction: float
    trade_episodes_fraction: float

    # Reward distribution
    avg_entry_reward: float
    avg_close_reward: float
    avg_hold_reward: float


class TrainingDiagnostics:
    """Real-time training diagnostics monitor."""

    def __init__(
        self,
        checkpoint_dir: str = "/workspace/micro/checkpoints",
        db_path: str = "/workspace/micro/training.db",
        window_size: int = 200
    ):
        """
        Initialize diagnostics monitor.

        Args:
            checkpoint_dir: Directory containing checkpoints
            db_path: Path to training database
            window_size: Window size for rolling metrics
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.db_path = db_path
        self.window_size = window_size

        # Rolling windows for metrics
        self.expectancy_window = deque(maxlen=window_size)
        self.kl_window = deque(maxlen=window_size)
        self.value_corr_window = deque(maxlen=window_size)
        self.trade_length_window = deque(maxlen=window_size)

        # Track improvements
        self.baseline_expectancy = -4.0  # Starting expectancy
        self.baseline_kl = None
        self.improvement_markers = []

    def get_latest_logs(self, container: str = "micro_training", lines: int = 500) -> str:
        """Get latest container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", container, "--tail", str(lines)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5
            )
            return result.stdout
        except:
            return ""

    def parse_episode_metrics(self, logs: str) -> Optional[Dict]:
        """Parse metrics from episode log line."""
        import re

        # Match episode log lines
        pattern = r'Episode (\d+).*Exp: ([-\d.]+).*WR: ([\d.]+)%.*TradeRatio: ([\d.]+)%.*Loss: ([\d.]+)'
        matches = re.findall(pattern, logs)

        if matches:
            latest = matches[-1]
            return {
                'episode': int(latest[0]),
                'expectancy': float(latest[1]),
                'win_rate': float(latest[2]),
                'trade_ratio': float(latest[3]),
                'total_loss': float(latest[4])
            }
        return None

    def calculate_kl_divergence(self, policy_logits: np.ndarray, visit_counts: np.ndarray) -> float:
        """Calculate KL divergence between policy and MCTS visits."""
        # Normalize visit counts
        visit_probs = visit_counts / (visit_counts.sum() + 1e-8)

        # Convert logits to probabilities
        policy_probs = torch.softmax(torch.tensor(policy_logits), dim=-1).numpy()

        # Calculate KL divergence
        kl = np.sum(policy_probs * np.log(policy_probs / (visit_probs + 1e-8) + 1e-8))
        return float(kl)

    def calculate_value_correlation(self, predicted_values: List[float],
                                   realized_returns: List[float]) -> float:
        """Calculate correlation between predicted values and realized returns."""
        if len(predicted_values) < 10:
            return 0.0

        correlation = np.corrcoef(predicted_values, realized_returns)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def run_diagnostics(self) -> DiagnosticMetrics:
        """Run full diagnostic suite."""
        # Get latest logs
        logs = self.get_latest_logs()

        # Parse basic metrics
        episode_metrics = self.parse_episode_metrics(logs)
        if not episode_metrics:
            logger.warning("No episode metrics found in logs")
            return None

        # Initialize metrics
        metrics = DiagnosticMetrics(
            episode=episode_metrics['episode'],
            timestamp=time.time(),
            expectancy=episode_metrics['expectancy'],
            win_rate=episode_metrics['win_rate'],
            trade_ratio=episode_metrics['trade_ratio'],
            total_loss=episode_metrics['total_loss'],

            # Placeholder values - would need deeper integration
            avg_trade_length=10.0,
            policy_loss=0.0,
            value_loss=0.0,
            outcome_loss=0.0,
            kl_policy_mcts=0.0,
            avg_visit_entropy=0.0,
            best_child_fraction=0.0,
            value_correlation=0.0,
            value_mse=0.0,
            buffer_size=10000,
            success_buffer_fraction=0.1,
            trade_episodes_fraction=0.3,
            avg_entry_reward=0.0,
            avg_close_reward=0.0,
            avg_hold_reward=-0.01
        )

        # Update rolling windows
        self.expectancy_window.append(metrics.expectancy)

        return metrics

    def print_diagnostic_report(self, metrics: DiagnosticMetrics):
        """Print formatted diagnostic report."""
        print("\n" + "="*60)
        print(" TRAINING DIAGNOSTICS REPORT")
        print("="*60)

        # Episode info
        print(f"\n📊 Episode {metrics.episode:,}")
        print("-"*40)

        # Performance metrics
        print("\n🎯 PERFORMANCE METRICS")
        print(f"  Expectancy:     {metrics.expectancy:+.2f} pips")
        print(f"  Win Rate:       {metrics.win_rate:.1f}%")
        print(f"  Trade Ratio:    {metrics.trade_ratio:.1f}%")

        # Progress indicators
        expectancy_improvement = metrics.expectancy - self.baseline_expectancy
        if expectancy_improvement > 0:
            print(f"\n✅ IMPROVING: Expectancy improved by {expectancy_improvement:+.2f} pips")
        else:
            print(f"\n⚠️  Still negative by {abs(metrics.expectancy):.2f} pips")

        # Expected improvements after changes
        print("\n📈 EXPECTED IMPROVEMENTS (2-hour targets)")
        print("  [Target] KL(policy||visits) < 1.0")
        print(f"  [Current] KL = {metrics.kl_policy_mcts:.2f}")

        print("\n  [Target] Value correlation > 0.3")
        print(f"  [Current] Correlation = {metrics.value_correlation:.2f}")

        print("\n  [Target] Expectancy → 0")
        print(f"  [Current] Expectancy = {metrics.expectancy:+.2f}")

        # Reward scheme check
        print("\n🎁 REWARD SCHEME (after simplification)")
        print(f"  Entry rewards:  {metrics.avg_entry_reward:.3f} (should be 0.0)")
        print(f"  Close rewards:  {metrics.avg_close_reward:+.2f} (AMDDP1)")
        print(f"  Hold rewards:   {metrics.avg_hold_reward:.3f} (should be -0.01 when flat)")

        # Rolling statistics
        if len(self.expectancy_window) >= 50:
            recent_mean = np.mean(list(self.expectancy_window)[-50:])
            older_mean = np.mean(list(self.expectancy_window)[-100:-50]) if len(self.expectancy_window) >= 100 else self.baseline_expectancy
            trend = recent_mean - older_mean

            print(f"\n📊 TREND (last 50 episodes)")
            print(f"  Recent expectancy:  {recent_mean:+.2f}")
            print(f"  Trend:              {trend:+.2f} {'📈' if trend > 0 else '📉'}")

        print("\n" + "="*60)

    def monitor_loop(self, interval: int = 60):
        """Main monitoring loop."""
        logger.info("Starting diagnostic monitoring...")
        logger.info(f"Monitoring interval: {interval} seconds")

        while True:
            try:
                metrics = self.run_diagnostics()
                if metrics:
                    self.print_diagnostic_report(metrics)

                    # Check for significant improvements
                    if metrics.expectancy > -2.0 and self.baseline_expectancy <= -4.0:
                        logger.info("🎉 MILESTONE: Expectancy improved to > -2.0!")
                        self.improvement_markers.append(('expectancy_-2', metrics.episode))

                    if metrics.expectancy > 0 and len([m for m in self.improvement_markers if m[0] == 'positive']) == 0:
                        logger.info("🎊 BREAKTHROUGH: Achieved positive expectancy!")
                        self.improvement_markers.append(('positive', metrics.episode))

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped.")
                break
            except Exception as e:
                logger.error(f"Diagnostic error: {e}")
                time.sleep(interval)


def main():
    """Run diagnostics monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor MuZero training diagnostics")
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')

    args = parser.parse_args()

    monitor = TrainingDiagnostics()

    if args.once:
        metrics = monitor.run_diagnostics()
        if metrics:
            monitor.print_diagnostic_report(metrics)
    else:
        monitor.monitor_loop(interval=args.interval)


if __name__ == "__main__":
    main()