#!/usr/bin/env python3
"""
Validation script for Micro Stochastic MuZero checkpoints.

Includes Monte Carlo simulation, Dr. Bandy metrics, and PDF reporting.
"""

import torch
import numpy as np
import duckdb
import os
import time
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from scipy import stats
import seaborn as sns

# Add parent directory to path
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.stochastic_mcts import StochasticMCTS
from micro.training.episode_runner import EpisodeRunner, Episode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroValidator:
    """Validate micro MuZero checkpoints with Monte Carlo simulation."""

    def __init__(
        self,
        checkpoint_dir: str = "/workspace/micro/checkpoints",
        data_path: str = "/workspace/data/micro_features.duckdb",
        session_indices_path: str = "/workspace/micro/cache/valid_session_indices.pkl",
        results_dir: str = "/workspace/micro/validation_results",
        num_episodes: int = 100,
        monte_carlo_runs: int = 1000
    ):
        """
        Initialize validator with Monte Carlo capabilities.

        Args:
            checkpoint_dir: Directory with checkpoints
            data_path: Path to micro features database
            session_indices_path: Path to pre-calculated session indices
            results_dir: Directory for validation results
            num_episodes: Number of validation episodes
            monte_carlo_runs: Number of Monte Carlo simulations
        """
        self.checkpoint_dir = checkpoint_dir
        self.data_path = data_path
        self.session_indices_path = session_indices_path
        self.results_dir = results_dir
        self.num_episodes = num_episodes
        self.monte_carlo_runs = monte_carlo_runs

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track validated checkpoints
        self.validated_checkpoints = set()

        # Load pre-calculated session indices for proper validation
        self._load_session_indices()

        logger.info(f"Validator initialized")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
        logger.info(f"  Validation sessions: {len(self.val_indices)}")
        logger.info(f"  Monte Carlo runs: {monte_carlo_runs}")
        logger.info(f"  Device: {self.device}")

    def _load_session_indices(self):
        """Load pre-calculated valid session indices."""
        if not os.path.exists(self.session_indices_path):
            logger.warning(f"Session indices not found at {self.session_indices_path}")
            # Fallback to simple split
            self.val_indices = None
            self.test_indices = None
            return

        with open(self.session_indices_path, 'rb') as f:
            indices = pickle.load(f)
            self.val_indices = indices.get('val', [])
            self.test_indices = indices.get('test', [])
            logger.info(f"Loaded {len(self.val_indices)} validation sessions")

    def load_checkpoint(self, checkpoint_path: str) -> Optional[MicroStochasticMuZero]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model or None if failed
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Create model with separated architecture
            model = MicroStochasticMuZero(
                temporal_features=9,  # Market (5) + Time (4)
                static_features=6,  # Position features
                lag_window=32,
                hidden_dim=256,
                action_dim=4,
                support_size=300
            ).to(self.device)

            # Load weights
            model.set_weights(checkpoint['model_state'])
            model.eval()

            return model

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def run_validation_episodes(self, model, num_episodes: int = None) -> List[Episode]:
        """
        Run validation episodes using EpisodeRunner.

        Returns:
            List of Episode objects with full trading metrics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes

        # Create MCTS for validation
        mcts = StochasticMCTS(
            model=model,
            num_actions=4,
            num_simulations=25,
            discount=0.997,
            pb_c_base=19652,
            pb_c_init=1.25,
            dirichlet_alpha=1.0,
            exploration_fraction=0.25
        )

        # Create episode runner
        runner = EpisodeRunner(
            model=model,
            mcts=mcts,
            db_path=self.data_path,
            session_indices_path=self.session_indices_path,
            device=self.device
        )

        episodes = []
        for i in range(num_episodes):
            # Use deterministic settings for validation
            episode = runner.run_episode(
                split='val',
                session_idx=i % len(self.val_indices) if self.val_indices else None,
                temperature=1.0,  # Deterministic
                add_noise=False   # No exploration
            )
            episodes.append(episode)

            if (i + 1) % 10 == 0:
                logger.info(f"Validated {i+1}/{num_episodes} episodes")

        return episodes

    def calculate_bandy_metrics(self, episodes: List[Episode]) -> Dict:
        """
        Calculate Dr. Howard Bandy metrics including CAR, drawdown, Safe-f.

        Reference: https://www.blueowlpress.com/
        """
        # Extract equity curve from episodes
        equity = [60000.0]  # Starting capital
        returns = []
        trades = []

        for episode in episodes:
            for exp in episode.experiences:
                if exp.action == 3:  # CLOSE action
                    trades.append(exp.reward)
                    equity.append(equity[-1] + exp.reward * 100)  # Convert pips to dollars
                    if equity[-2] != 0:
                        returns.append((equity[-1] - equity[-2]) / equity[-2])

        equity = np.array(equity)

        # Calculate metrics
        metrics = {}

        # Basic stats
        metrics['total_trades'] = len(trades)
        metrics['win_rate'] = len([t for t in trades if t > 0]) / max(1, len(trades))
        metrics['expectancy'] = np.mean(trades) if trades else 0

        # Drawdown analysis
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100
        metrics['max_drawdown_dollar'] = abs((equity - running_max).min())

        # CAR (Compound Annual Return)
        if len(equity) > 1:
            years = len(episodes) / (252 * 6)  # Assuming 6 hours per session, 252 trading days
            final_return = (equity[-1] / equity[0])
            metrics['car'] = (final_return ** (1/max(0.1, years)) - 1) * 100 if years > 0 else 0
        else:
            metrics['car'] = 0

        # Sharpe Ratio
        if returns:
            metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 6)
        else:
            metrics['sharpe_ratio'] = 0

        # Safe-f calculation (position sizing)
        if trades:
            # Using Ralph Vince's Optimal f formula approximation
            wins = [t for t in trades if t > 0]
            losses = [abs(t) for t in trades if t < 0]
            if wins and losses:
                avg_win = np.mean(wins)
                avg_loss = np.mean(losses)
                p_win = len(wins) / len(trades)

                # Kelly Criterion as basis for Safe-f
                kelly = (p_win * avg_win - (1 - p_win) * avg_loss) / avg_win
                metrics['safe_f'] = max(0, min(0.25, kelly * 0.25))  # Conservative Safe-f
            else:
                metrics['safe_f'] = 0.01
        else:
            metrics['safe_f'] = 0.01

        return metrics, equity

    def monte_carlo_simulation(self, episodes: List[Episode]) -> Dict:
        """
        Run Monte Carlo simulation to generate confidence intervals.
        """
        # Extract all trades
        all_trades = []
        for episode in episodes:
            episode_trades = [exp.reward for exp in episode.experiences if exp.action == 3]
            all_trades.extend(episode_trades)

        if not all_trades:
            return {}

        # Run Monte Carlo
        mc_results = []
        for _ in range(self.monte_carlo_runs):
            # Bootstrap sample trades with replacement
            sample_trades = np.random.choice(all_trades, size=len(all_trades), replace=True)

            # Calculate equity curve
            equity = 60000.0
            curve = [equity]
            for trade in sample_trades:
                equity += trade * 100
                curve.append(equity)

            mc_results.append(curve[-1])  # Final equity

        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        mc_stats = {}

        for p in percentiles:
            mc_stats[f'p{p}'] = np.percentile(mc_results, p)

        # Additional MC stats
        mc_stats['mean'] = np.mean(mc_results)
        mc_stats['std'] = np.std(mc_results)
        mc_stats['min'] = np.min(mc_results)
        mc_stats['max'] = np.max(mc_results)

        return mc_stats, mc_results

    def get_validation_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch of validation data with separated temporal and static features.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (temporal_batch, static_batch):
                - temporal_batch: (batch, 32, 9) - market + time features
                - static_batch: (batch, 6) - position features
        """
        temporal_observations = []
        static_observations = []

        for _ in range(batch_size):
            # Random index in validation range
            idx = np.random.randint(
                self.validation_start,
                self.total_rows - 100
            )

            # Fetch data
            query = f"""
            SELECT *
            FROM micro_features
            WHERE bar_index = {idx}
            """

            row = self.conn.execute(query).fetchone()

            if row:
                # Extract features (simplified - should match training)
                features = []
                col_idx = 3  # Skip metadata columns

                # Technical indicators (5 x 32)
                for _ in range(5 * 32):
                    features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Cyclical features (4 x 32)
                for _ in range(4 * 32):
                    features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Position features (6)
                position_features = []
                for _ in range(6):
                    position_features.append(row[col_idx] if row[col_idx] is not None else 0.0)
                    col_idx += 1

                # Reshape temporal to (32, 9)
                temporal = np.zeros((32, 9))
                for t in range(32):
                    # Technical (5)
                    for f in range(5):
                        temporal[t, f] = features[f * 32 + t]
                    # Cyclical (4)
                    for f in range(4):
                        temporal[t, 5 + f] = features[160 + f * 32 + t]

                # Static features (6,)
                static = np.array(position_features)

                temporal_observations.append(temporal)
                static_observations.append(static)

        temporal_batch = torch.tensor(np.array(temporal_observations), dtype=torch.float32, device=self.device)
        static_batch = torch.tensor(np.array(static_observations), dtype=torch.float32, device=self.device)

        return temporal_batch, static_batch

    def generate_pdf_report(self, checkpoint_path: str, episodes: List[Episode],
                            metrics: Dict, mc_stats: Dict, mc_results: List, equity: np.ndarray):
        """
        Generate comprehensive PDF report with Dr. Bandy style visualizations.
        """
        pdf_path = os.path.join(
            self.results_dir,
            f"validation_{os.path.basename(checkpoint_path)}.pdf"
        )

        with pdf.PdfPages(pdf_path) as pdf_file:
            # Page 1: Monte Carlo Results
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))

            # MC Min/Max Equity
            ax1 = axes[0]
            mc_curves = []
            for _ in range(min(100, self.monte_carlo_runs)):
                trades = np.random.choice(
                    [exp.reward for ep in episodes for exp in ep.experiences if exp.action == 3],
                    size=len([exp for ep in episodes for exp in ep.experiences if exp.action == 3]),
                    replace=True
                )
                curve = [60000]
                for t in trades:
                    curve.append(curve[-1] + t * 100)
                mc_curves.append(curve)

            # Plot percentile bands
            lengths = [len(c) for c in mc_curves]
            max_len = max(lengths) if lengths else 0

            if max_len > 0:
                padded_curves = []
                for curve in mc_curves:
                    padded = curve + [curve[-1]] * (max_len - len(curve))
                    padded_curves.append(padded)

                curves_array = np.array(padded_curves)
                percentiles = [5, 25, 50, 75, 95]
                colors = ['red', 'orange', 'green', 'orange', 'red']

                for p, color in zip(percentiles, colors):
                    p_curve = np.percentile(curves_array, p, axis=0)
                    ax1.plot(p_curve, color=color, alpha=0.7, label=f'{p}th percentile')

            ax1.set_title('Monte Carlo Min/Max Equity')
            ax1.set_xlabel('Trade #')
            ax1.set_ylabel('Equity ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # MC Final Equity Distribution
            ax2 = axes[1]
            ax2.hist(mc_results, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(np.percentile(mc_results, 50), color='green', linestyle='--', label='Median')
            ax2.axvline(np.percentile(mc_results, 5), color='red', linestyle='--', label='5th percentile')
            ax2.axvline(np.percentile(mc_results, 95), color='red', linestyle='--', label='95th percentile')
            ax2.set_title('MC Final Equity Distribution')
            ax2.set_xlabel('Final Equity ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()

            # CAR Distribution
            ax3 = axes[2]
            car_values = []
            for _ in range(min(1000, self.monte_carlo_runs)):
                sample_trades = np.random.choice(
                    [exp.reward for ep in episodes for exp in ep.experiences if exp.action == 3],
                    size=len([exp for ep in episodes for exp in ep.experiences if exp.action == 3]),
                    replace=True
                )
                sample_equity = 60000
                for t in sample_trades:
                    sample_equity += t * 100
                years = len(episodes) / (252 * 6)
                if years > 0:
                    car = ((sample_equity / 60000) ** (1/years) - 1) * 100
                    car_values.append(car)

            if car_values:
                ax3.hist(car_values, bins=30, edgecolor='black', alpha=0.7)
                ax3.set_title('Annual Return % Distribution')
                ax3.set_xlabel('CAR (%)')
                ax3.set_ylabel('Frequency')

            plt.suptitle(f'Monte Carlo Analysis - {os.path.basename(checkpoint_path)}', fontsize=14)
            plt.tight_layout()
            pdf_file.savefig()
            plt.close()

            # Page 2: Performance Metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Equity Curve
            ax1 = axes[0, 0]
            ax1.plot(equity, 'b-', linewidth=1)
            ax1.fill_between(range(len(equity)), equity[0], equity, alpha=0.3)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Trades')
            ax1.set_ylabel('Equity ($)')
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[0, 1]
            running_max = np.maximum.accumulate(equity)
            dd = (equity - running_max) / running_max * 100
            ax2.fill_between(range(len(dd)), 0, dd, color='red', alpha=0.5)
            ax2.set_title('Drawdown %')
            ax2.set_xlabel('Trades')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # Trade Distribution
            ax3 = axes[1, 0]
            trades = [exp.reward for ep in episodes for exp in ep.experiences if exp.action == 3]
            if trades:
                ax3.hist(trades, bins=30, edgecolor='black', alpha=0.7)
                ax3.axvline(0, color='red', linestyle='--')
                ax3.set_title('Trade Distribution')
                ax3.set_xlabel('Profit/Loss (pips)')
                ax3.set_ylabel('Frequency')

            # Metrics Table
            ax4 = axes[1, 1]
            ax4.axis('tight')
            ax4.axis('off')

            table_data = [
                ['Metric', 'Value'],
                ['Total Episodes', f"{len(episodes)}"],
                ['Total Trades', f"{metrics['total_trades']}"],
                ['Win Rate', f"{metrics['win_rate']:.1%}"],
                ['Expectancy', f"{metrics['expectancy']:.2f} pips"],
                ['Max DD %', f"{metrics['max_drawdown_pct']:.1f}%"],
                ['Max DD $', f"${metrics['max_drawdown_dollar']:.0f}"],
                ['CAR', f"{metrics['car']:.1f}%"],
                ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
                ['Safe-f', f"{metrics['safe_f']:.3f}"],
                ['MC P5', f"${mc_stats.get('p5', 0):.0f}"],
                ['MC P50', f"${mc_stats.get('p50', 0):.0f}"],
                ['MC P95', f"${mc_stats.get('p95', 0):.0f}"],
            ]

            table = ax4.table(cellText=table_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            plt.suptitle(f'Performance Analysis - {os.path.basename(checkpoint_path)}', fontsize=14)
            plt.tight_layout()
            pdf_file.savefig()
            plt.close()

        logger.info(f"PDF report saved: {pdf_path}")
        return pdf_path

    def validate_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Validate a single checkpoint with full episode runs and Monte Carlo.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating checkpoint: {checkpoint_path}")

        # Load model
        model = self.load_checkpoint(checkpoint_path)
        if model is None:
            return None

        # Run validation episodes with proper EpisodeRunner
        logger.info("Running validation episodes...")
        episodes = self.run_validation_episodes(model, self.num_episodes)

        # Calculate Dr. Bandy metrics
        logger.info("Calculating performance metrics...")
        metrics, equity = self.calculate_bandy_metrics(episodes)

        # Run Monte Carlo simulation
        logger.info(f"Running {self.monte_carlo_runs} Monte Carlo simulations...")
        mc_stats, mc_results = self.monte_carlo_simulation(episodes)

        # Generate PDF report
        logger.info("Generating PDF report...")
        pdf_path = self.generate_pdf_report(checkpoint_path, episodes, metrics, mc_stats, mc_results, equity)

        # Compile results
        results = {
            'checkpoint': os.path.basename(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'episodes': len(episodes),
            'metrics': metrics,
            'monte_carlo': mc_stats,
            'pdf_report': pdf_path,

            # Summary stats
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'expectancy': metrics['expectancy'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'car': metrics['car'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'safe_f': metrics['safe_f'],

            # MC confidence intervals
            'mc_p5': mc_stats.get('p5', 0),
            'mc_p50': mc_stats.get('p50', 0),
            'mc_p95': mc_stats.get('p95', 0),

            # Quality score (Dr. Bandy inspired)
            'quality_score': self._calculate_quality_score(metrics, mc_stats)
        }

        logger.info(f"Validation complete:")
        logger.info(f"  Total trades: {results['total_trades']}")
        logger.info(f"  Win rate: {results['win_rate']:.1%}")
        logger.info(f"  Expectancy: {results['expectancy']:.2f} pips")
        logger.info(f"  Max DD: {results['max_drawdown_pct']:.1f}%")
        logger.info(f"  CAR: {results['car']:.1f}%")
        logger.info(f"  Sharpe: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Safe-f: {results['safe_f']:.3f}")
        logger.info(f"  MC P5-P95: ${results['mc_p5']:.0f} - ${results['mc_p95']:.0f}")
        logger.info(f"  Quality Score: {results['quality_score']:.2f}")

        return results

    def _calculate_quality_score(self, metrics: Dict, mc_stats: Dict) -> float:
        """
        Calculate overall quality score based on Dr. Bandy principles.
        """
        # Base score from expectancy
        score = metrics['expectancy'] * 10

        # Add CAR bonus (scaled)
        score += max(0, metrics['car']) * 0.5

        # Add Sharpe ratio bonus
        score += max(0, metrics['sharpe_ratio']) * 5

        # Penalize for drawdown
        score -= metrics['max_drawdown_pct'] * 0.5

        # Add consistency bonus (MC P5 close to P50)
        if mc_stats.get('p50', 0) > 0:
            consistency = mc_stats.get('p5', 0) / mc_stats.get('p50', 1)
            score += consistency * 10

        # Safe-f bonus (higher is better for position sizing)
        score += metrics['safe_f'] * 100

        return max(0, score)

    def save_results(self, results: Dict):
        """Save validation results."""
        # Save to JSON
        results_file = os.path.join(
            self.results_dir,
            f"validation_{results['checkpoint']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved: {results_file}")

        # Update best checkpoint if needed
        best_file = os.path.join(self.results_dir, "best_checkpoint.json")
        is_new_best = False

        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                best = json.load(f)
            if results['quality_score'] > best.get('quality_score', -999):
                is_new_best = True
                with open(best_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"🎯 New best checkpoint! Quality: {results['quality_score']:.2f}, Expectancy: {results['expectancy']:.3f}")
        else:
            is_new_best = True
            with open(best_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"First checkpoint validated")

        # Copy best checkpoint to validation directory to preserve it
        if is_new_best:
            self.preserve_best_checkpoint(results['checkpoint'])

    def preserve_best_checkpoint(self, checkpoint_path: str):
        """Copy best checkpoint to validation directory to prevent overwrite."""
        try:
            import shutil

            # Create preserved checkpoints directory
            preserved_dir = os.path.join(self.results_dir, "preserved_checkpoints")
            os.makedirs(preserved_dir, exist_ok=True)

            # Copy checkpoint to preserved location
            checkpoint_name = os.path.basename(checkpoint_path)
            preserved_path = os.path.join(preserved_dir, f"best_{checkpoint_name}")

            # Also save with timestamp for history
            timestamped_path = os.path.join(
                preserved_dir,
                f"best_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{checkpoint_name}"
            )

            shutil.copy2(checkpoint_path, preserved_path)
            shutil.copy2(checkpoint_path, timestamped_path)

            logger.info(f"✅ Best checkpoint preserved: {preserved_path}")
            logger.info(f"📁 History saved: {timestamped_path}")

            # Also copy to a fixed "best_validated.pth" for easy access
            best_validated_path = os.path.join(preserved_dir, "best_validated.pth")
            shutil.copy2(checkpoint_path, best_validated_path)
            logger.info(f"🏆 Best validated checkpoint: {best_validated_path}")

        except Exception as e:
            logger.error(f"Failed to preserve best checkpoint: {e}")

    def monitor_checkpoints(self):
        """Continuously monitor for new checkpoints."""
        logger.info("Starting checkpoint monitor...")

        while True:
            try:
                # List checkpoint files
                if os.path.exists(self.checkpoint_dir):
                    checkpoints = [
                        f for f in os.listdir(self.checkpoint_dir)
                        if f.endswith('.pth') and f != 'latest.pth'
                    ]

                    for checkpoint_name in checkpoints:
                        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

                        # Skip if already validated
                        if checkpoint_path in self.validated_checkpoints:
                            continue

                        # Validate checkpoint
                        results = self.validate_checkpoint(checkpoint_path)

                        if results:
                            self.save_results(results)
                            self.validated_checkpoints.add(checkpoint_path)

                # Wait before next check
                time.sleep(60)  # Check every minute

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(60)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    validator = MicroValidator()

    try:
        validator.monitor_checkpoints()
    except KeyboardInterrupt:
        logger.info("Validation stopped by user")
    finally:
        validator.cleanup()