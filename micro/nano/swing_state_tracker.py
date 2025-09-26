#!/usr/bin/env python3
"""
Swing State Tracker - Four Market Structure States
Tracks H-L swing points as one of four states: HHHL, LHLL, HHLL, LHHL

Key Rules:
- A swing low is not established unless previous swing high is exceeded
- A swing high is not established unless previous swing low is broken
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from enum import Enum


class SwingState(Enum):
    """Four possible market structure states."""
    HHHL = "HHHL"  # Higher High, Higher Low (Strong Uptrend)
    LHLL = "LHLL"  # Lower High, Lower Low (Strong Downtrend)
    HHLL = "HHLL"  # Higher High, Lower Low (Expansion/Volatile)
    LHHL = "LHHL"  # Lower High, Higher Low (Contraction/Range)
    UNDEFINED = "UNDEFINED"  # Not enough data


class SwingPoint:
    """Represents a single swing point."""
    def __init__(self, index: int, price: float, swing_type: str, confirmed: bool = False):
        self.index = index
        self.price = price
        self.type = swing_type  # 'high' or 'low'
        self.confirmed = confirmed

    def __repr__(self):
        status = "✓" if self.confirmed else "?"
        return f"Swing({self.type[0].upper()}, {self.price:.5f}, idx={self.index}, {status})"


class SwingStateTracker:
    """
    Tracks swing highs/lows and determines market structure state.

    A swing is only confirmed when opposite swing is exceeded:
    - Swing Low confirmed when price exceeds previous Swing High
    - Swing High confirmed when price breaks below previous Swing Low
    """

    def __init__(self, k: int = 3):
        """
        Args:
            k: Number of bars on each side to identify swing points
        """
        self.k = k
        self.swings: List[SwingPoint] = []
        self.confirmed_swings: List[SwingPoint] = []
        self.current_state = SwingState.UNDEFINED
        self.state_history: List[Tuple[int, SwingState]] = []

    def find_potential_swings(self, df: pd.DataFrame) -> List[SwingPoint]:
        """
        Find potential swing points (not yet confirmed).
        These are local highs/lows based on k-bar lookback.
        """
        highs = df['high'].values
        lows = df['low'].values
        indices = df.index.tolist()  # Get actual bar_index values
        potential_swings = []
        n = len(df)

        for i in range(self.k, n - self.k):
            # Check for swing high
            win_h = highs[i - self.k:i + self.k + 1]
            if highs[i] == win_h.max() and np.sum(win_h == highs[i]) == 1:
                # Use actual bar_index, not sequential position
                potential_swings.append(SwingPoint(indices[i], float(highs[i]), 'high', False))

            # Check for swing low
            win_l = lows[i - self.k:i + self.k + 1]
            if lows[i] == win_l.min() and np.sum(win_l == lows[i]) == 1:
                # Use actual bar_index, not sequential position
                potential_swings.append(SwingPoint(indices[i], float(lows[i]), 'low', False))

        # Sort by index
        potential_swings.sort(key=lambda x: x.index)
        return potential_swings

    def update_confirmations(self, df: pd.DataFrame, potential_swings: List[SwingPoint]) -> List[SwingPoint]:
        """
        Confirm swings based on price exceeding opposite swings.
        A swing low is confirmed when price exceeds the previous swing high.
        A swing high is confirmed when price breaks the previous swing low.
        """
        confirmed = []

        # Create a mapping from bar_index to position in dataframe
        index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        for i, swing in enumerate(potential_swings):
            # Get position of this swing in the dataframe
            if swing.index not in index_to_pos:
                continue

            swing_pos = index_to_pos[swing.index]

            # Look for confirmation after this swing
            if swing_pos + 1 >= len(df):
                continue

            if swing.type == 'low':
                # Find previous swing high to exceed
                prev_highs = [s for s in potential_swings[:i] if s.type == 'high']
                if prev_highs:
                    prev_high = prev_highs[-1]
                    # Check if price exceeded previous high after this low
                    for j in range(swing_pos + 1, len(df)):
                        if df.iloc[j]['high'] > prev_high.price:
                            swing.confirmed = True
                            confirmed.append(swing)
                            break

            elif swing.type == 'high':
                # Find previous swing low to break
                prev_lows = [s for s in potential_swings[:i] if s.type == 'low']
                if prev_lows:
                    prev_low = prev_lows[-1]
                    # Check if price broke previous low after this high
                    for j in range(swing_pos + 1, len(df)):
                        if df.iloc[j]['low'] < prev_low.price:
                            swing.confirmed = True
                            confirmed.append(swing)
                            break

        return confirmed

    def determine_state(self, confirmed_swings: List[SwingPoint]) -> SwingState:
        """
        Determine current market structure state from confirmed swings.
        Requires at least 2 confirmed highs and 2 confirmed lows.
        """
        highs = [s for s in confirmed_swings if s.type == 'high']
        lows = [s for s in confirmed_swings if s.type == 'low']

        if len(highs) < 2 or len(lows) < 2:
            return SwingState.UNDEFINED

        # Get last two of each
        h1, h2 = highs[-2], highs[-1]
        l1, l2 = lows[-2], lows[-1]

        # Determine state
        higher_high = h2.price > h1.price
        higher_low = l2.price > l1.price

        if higher_high and higher_low:
            return SwingState.HHHL
        elif not higher_high and not higher_low:
            return SwingState.LHLL
        elif higher_high and not higher_low:
            return SwingState.HHLL
        else:  # not higher_high and higher_low
            return SwingState.LHHL

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete analysis of swing states.

        Returns:
            Dictionary with swings, state, and analysis results
        """
        # Find potential swings
        potential_swings = self.find_potential_swings(df)

        # Confirm swings based on price action
        confirmed_swings = self.update_confirmations(df, potential_swings)

        # Determine current state
        current_state = self.determine_state(confirmed_swings)

        # Track state changes over time
        state_timeline = []
        for i in range(4, len(confirmed_swings) + 1):
            subset = confirmed_swings[:i]
            state = self.determine_state(subset)
            if state != SwingState.UNDEFINED:
                last_swing = subset[-1]
                state_timeline.append((last_swing.index, state))

        # Store results
        self.swings = potential_swings
        self.confirmed_swings = confirmed_swings
        self.current_state = current_state
        self.state_history = state_timeline

        return {
            'potential_swings': potential_swings,
            'confirmed_swings': confirmed_swings,
            'current_state': current_state,
            'state_history': state_timeline,
            'stats': self.get_stats()
        }

    def get_stats(self) -> Dict:
        """Get statistics about swings and states."""
        return {
            'total_potential_swings': len(self.swings),
            'confirmed_swings': len(self.confirmed_swings),
            'confirmation_rate': len(self.confirmed_swings) / len(self.swings) if self.swings else 0,
            'confirmed_highs': len([s for s in self.confirmed_swings if s.type == 'high']),
            'confirmed_lows': len([s for s in self.confirmed_swings if s.type == 'low']),
            'current_state': self.current_state.value,
            'state_changes': len(set(s[1] for s in self.state_history))
        }

    def plot_analysis(self, df: pd.DataFrame, save_path: str = 'swing_states.png'):
        """Visualize swing points and state transitions."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

        # Reset index to use sequential integers for plotting
        df_plot = df.reset_index(drop=True)

        # Plot price and swings
        ax1.plot(df_plot.index, df_plot['close'], 'k-', linewidth=0.5, alpha=0.7, label='Close')

        # Create index mapping for swings
        # Map original bar_index to sequential plot index
        original_index = df.index.tolist()
        index_map = {orig: i for i, orig in enumerate(original_index)}

        # Plot potential swings (unconfirmed)
        for swing in self.swings:
            if not swing.confirmed:
                if swing.index in index_map:
                    plot_idx = index_map[swing.index]
                    marker = 'v' if swing.type == 'high' else '^'
                    color = 'lightcoral' if swing.type == 'high' else 'lightgreen'
                    ax1.plot(plot_idx, swing.price, marker, color=color, alpha=0.3, markersize=6)

        # Plot confirmed swings
        for swing in self.confirmed_swings:
            if swing.index in index_map:
                plot_idx = index_map[swing.index]
                marker = 'v' if swing.type == 'high' else '^'
                color = 'red' if swing.type == 'high' else 'green'
                ax1.plot(plot_idx, swing.price, marker, color=color, markersize=10, alpha=0.8)
                ax1.text(plot_idx, swing.price, f"{swing.type[0].upper()}", fontsize=8, ha='center')

        # Add state regions
        if self.state_history:
            colors = {
                SwingState.HHHL: 'lightgreen',
                SwingState.LHLL: 'lightcoral',
                SwingState.HHLL: 'lightyellow',
                SwingState.LHHL: 'lightblue'
            }

            for i, (idx, state) in enumerate(self.state_history):
                if idx in index_map:
                    plot_start = index_map[idx]
                    if i+1 < len(self.state_history):
                        next_idx = self.state_history[i+1][0]
                        plot_end = index_map.get(next_idx, len(df_plot))
                    else:
                        plot_end = len(df_plot)
                    ax1.axvspan(plot_start, plot_end, alpha=0.2, color=colors.get(state, 'gray'))

        ax1.set_title(f'Swing State Analysis - Current: {self.current_state.value}')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        # Create legend
        legend_elements = [
            mpatches.Patch(color='lightgreen', alpha=0.2, label='HHHL (Uptrend)'),
            mpatches.Patch(color='lightcoral', alpha=0.2, label='LHLL (Downtrend)'),
            mpatches.Patch(color='lightyellow', alpha=0.2, label='HHLL (Expansion)'),
            mpatches.Patch(color='lightblue', alpha=0.2, label='LHHL (Contraction)')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')

        # Plot state timeline
        if self.state_history:
            states_numeric = {'HHHL': 2, 'LHLL': -2, 'HHLL': 1, 'LHHL': -1}
            x = [index_map.get(s[0], 0) for s in self.state_history if s[0] in index_map]
            y = [states_numeric.get(s[1].value, 0) for s in self.state_history if s[0] in index_map]
            if x and y:
                ax2.step(x, y, where='post', linewidth=2, color='blue')
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_ylim(-3, 3)
                ax2.set_yticks([-2, -1, 0, 1, 2])
                ax2.set_yticklabels(['LHLL', 'LHHL', '-', 'HHLL', 'HHHL'])
                ax2.set_xlabel('Bar Index')
                ax2.set_title('Market Structure State Timeline')
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        print(f"📊 Swing state chart saved to {save_path}")

        return fig


def test_swing_states():
    """Test the swing state tracker with real OHLC data."""
    import duckdb

    print("🔍 TESTING SWING STATE TRACKER WITH REAL OHLC DATA")
    print("="*70)

    # Load real OHLC data from master table
    conn = duckdb.connect("../../data/master.duckdb", read_only=True)
    query = """
    SELECT
        bar_index,
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM master
    WHERE bar_index > 10000
    LIMIT 2000
    """

    df = conn.execute(query).df()

    # Set bar_index as index for the tracker
    df.set_index('bar_index', inplace=True)

    print(f"Loaded {len(df)} bars of real OHLC data")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    print(f"Average candle range: {(df['high'] - df['low']).mean():.5f}")

    # Run analysis
    tracker = SwingStateTracker(k=5)
    results = tracker.analyze(df)

    # Print results
    print("\n📊 ANALYSIS RESULTS:")
    print("-"*50)
    stats = results['stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n🎯 CONFIRMED SWINGS (last 10):")
    print("-"*50)
    for swing in results['confirmed_swings'][-10:]:
        print(f"  {swing}")

    print("\n📈 STATE HISTORY (last 5 transitions):")
    print("-"*50)
    for idx, state in results['state_history'][-5:]:
        print(f"  Bar {idx}: {state.value}")

    # Create visualization
    tracker.plot_analysis(df)

    print("\n✅ Swing state tracking complete!")
    return tracker, results


if __name__ == "__main__":
    tracker, results = test_swing_states()