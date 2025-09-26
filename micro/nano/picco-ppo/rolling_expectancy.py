#!/usr/bin/env python3
"""
Rolling Expectancy Tracker for Multi-Year Trading Sessions.

Implements Van Tharp's expectancy with rolling windows to track
performance evolution over time, similar to peoplesfintech approach.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import pandas as pd


class RollingExpectancyTracker:
    """
    Tracks expectancy_R using rolling windows for long-term sessions.

    Features:
    - Multiple window sizes (100, 500, 1000 trades)
    - Van Tharp R-multiple expectancy
    - Tracks performance evolution over time
    - Handles multi-year continuous trading
    """

    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize rolling expectancy tracker.

        Args:
            window_sizes: List of rolling window sizes (default: [100, 500, 1000])
        """
        self.window_sizes = window_sizes or [100, 500, 1000]
        self.windows = {size: deque(maxlen=size) for size in self.window_sizes}
        self.all_trades = []  # Keep all trades for full history
        self.trade_count = 0
        self.history = []  # Track expectancy over time

    def add_trade(self, pips: float) -> Dict[str, float]:
        """
        Add a new trade and update rolling expectancies.

        Args:
            pips: P&L in pips for the trade

        Returns:
            Dict of current expectancies for each window size
        """
        self.trade_count += 1
        self.all_trades.append(pips)

        # Add to all windows
        for window in self.windows.values():
            window.append(pips)

        # Calculate current expectancies
        expectancies = self.calculate_expectancies()

        # Store in history
        self.history.append({
            'trade_num': self.trade_count,
            'pips': pips,
            **expectancies
        })

        return expectancies

    def calculate_expectancies(self) -> Dict[str, float]:
        """
        Calculate expectancy_R for all window sizes.

        Returns:
            Dict with expectancy_R for each window size
        """
        results = {}

        for size in self.window_sizes:
            window_trades = list(self.windows[size])

            if len(window_trades) < 20:  # Need minimum trades
                results[f'expectancy_R_{size}'] = 0.0
                results[f'sample_size_{size}'] = len(window_trades)
                continue

            # Calculate Van Tharp expectancy_R
            trades_array = np.array(window_trades)
            losses = trades_array[trades_array < 0]

            if len(losses) == 0:
                # No losses - very good but use small R
                avg_loss = 1.0
            else:
                avg_loss = abs(np.mean(losses))  # R value

            expectancy_pips = np.mean(trades_array)
            expectancy_R = expectancy_pips / avg_loss

            results[f'expectancy_R_{size}'] = expectancy_R
            results[f'sample_size_{size}'] = len(window_trades)
            results[f'win_rate_{size}'] = np.mean(trades_array > 0) * 100
            results[f'avg_pips_{size}'] = expectancy_pips
            results[f'R_value_{size}'] = avg_loss

        # Also calculate lifetime expectancy
        if len(self.all_trades) >= 20:
            all_array = np.array(self.all_trades)
            losses = all_array[all_array < 0]
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
            lifetime_expectancy = np.mean(all_array) / avg_loss

            results['lifetime_expectancy_R'] = lifetime_expectancy
            results['lifetime_trades'] = len(self.all_trades)
            results['lifetime_win_rate'] = np.mean(all_array > 0) * 100

        return results

    def get_quality_assessment(self, expectancy_R: float) -> Tuple[str, str]:
        """
        Assess system quality based on expectancy_R.

        Args:
            expectancy_R: Expectancy in R-multiples

        Returns:
            Tuple of (quality_level, emoji)
        """
        if expectancy_R > 0.5:
            return "EXCELLENT", "🏆"
        elif expectancy_R > 0.25:
            return "GOOD", "✅"
        elif expectancy_R > 0:
            return "ACCEPTABLE", "⚠️"
        else:
            return "NEEDS IMPROVEMENT", "🔴"

    def get_summary(self) -> str:
        """
        Get formatted summary of current expectancies.

        Returns:
            Formatted string with all expectancy metrics
        """
        expectancies = self.calculate_expectancies()

        summary = ["=" * 60]
        summary.append("📊 ROLLING EXPECTANCY SUMMARY")
        summary.append("=" * 60)

        for size in self.window_sizes:
            exp_key = f'expectancy_R_{size}'
            if exp_key in expectancies:
                exp_R = expectancies[exp_key]
                sample = expectancies[f'sample_size_{size}']
                win_rate = expectancies.get(f'win_rate_{size}', 0)
                avg_pips = expectancies.get(f'avg_pips_{size}', 0)

                quality, emoji = self.get_quality_assessment(exp_R)

                summary.append(f"\n📈 {size}-Trade Window:")
                summary.append(f"  Expectancy: {exp_R:.3f}R {emoji} ({quality})")
                summary.append(f"  Win Rate: {win_rate:.1f}%")
                summary.append(f"  Avg Pips: {avg_pips:.2f}")
                summary.append(f"  Sample: {sample}/{size} trades")

        # Lifetime stats
        if 'lifetime_expectancy_R' in expectancies:
            lifetime_R = expectancies['lifetime_expectancy_R']
            lifetime_trades = expectancies['lifetime_trades']
            lifetime_wr = expectancies['lifetime_win_rate']
            quality, emoji = self.get_quality_assessment(lifetime_R)

            summary.append(f"\n🎯 LIFETIME PERFORMANCE:")
            summary.append(f"  Expectancy: {lifetime_R:.3f}R {emoji} ({quality})")
            summary.append(f"  Win Rate: {lifetime_wr:.1f}%")
            summary.append(f"  Total Trades: {lifetime_trades}")

        summary.append("=" * 60)

        return "\n".join(summary)

    def plot_evolution(self, save_path: str = None):
        """
        Plot expectancy evolution over time.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.history:
            print("No data to plot yet")
            return

        try:
            import matplotlib.pyplot as plt

            df = pd.DataFrame(self.history)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot expectancy evolution
            ax1 = axes[0]
            for size in self.window_sizes:
                col = f'expectancy_R_{size}'
                if col in df.columns:
                    ax1.plot(df.index, df[col], label=f'{size}-trade window')

            ax1.axhline(y=0.25, color='g', linestyle='--', alpha=0.5, label='Good threshold')
            ax1.axhline(y=0.5, color='b', linestyle='--', alpha=0.5, label='Excellent threshold')
            ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)

            ax1.set_ylabel('Expectancy (R-multiples)')
            ax1.set_title('Rolling Expectancy Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot cumulative P&L
            ax2 = axes[1]
            cumulative_pips = df['pips'].cumsum()
            ax2.plot(df.index, cumulative_pips, color='navy')
            ax2.fill_between(df.index, 0, cumulative_pips, alpha=0.3)
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Cumulative Pips')
            ax2.set_title('Cumulative Performance')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 Plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")


def test_rolling_expectancy():
    """Test the rolling expectancy tracker."""
    print("Testing Rolling Expectancy Tracker")
    print("=" * 60)

    # Create tracker
    tracker = RollingExpectancyTracker(window_sizes=[10, 50, 100])

    # Simulate trading session with varying performance
    np.random.seed(42)

    # Phase 1: Learning phase (poor performance)
    for i in range(30):
        pips = np.random.normal(-2, 5)  # Negative expectancy
        tracker.add_trade(pips)

    # Phase 2: Improving phase
    for i in range(50):
        pips = np.random.normal(1, 4)  # Slightly positive
        tracker.add_trade(pips)

    # Phase 3: Good performance
    for i in range(70):
        pips = np.random.normal(3, 3)  # Good expectancy
        tracker.add_trade(pips)

    # Print summary
    print(tracker.get_summary())

    # Save plot
    tracker.plot_evolution("rolling_expectancy_test.png")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_rolling_expectancy()