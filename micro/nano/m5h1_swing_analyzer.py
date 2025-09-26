#!/usr/bin/env python3
"""
Multi-timeframe swing analyzer using M5 (5-minute) and H1 (60-minute) timeframes.
M5 bars are constructed from underlying M1 data for better signal quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
# import mplfinance as mpf  # Not needed for custom candlestick plotting
from datetime import datetime, timedelta
import duckdb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from swing_state_tracker import SwingState, SwingPoint, SwingStateTracker


class M5H1SwingAnalyzer:
    """Analyzes swing states across M5 and H1 timeframes"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.m1_data = None
        self.m5_data = None
        self.h1_data = None
        self.m5_tracker = SwingStateTracker(k=3)  # k=3 for M5 swing detection
        self.h1_tracker = SwingStateTracker(k=3)  # k=3 for H1 swing detection

    def load_data(self, start_idx: int = 10000, end_idx: int = 14000):
        """Load M1 data and construct M5 and H1 bars"""
        conn = duckdb.connect(self.db_path, read_only=True)

        # Load M1 data from master table (GBPJPY data)
        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {end_idx}
        ORDER BY bar_index
        """

        self.m1_data = pd.read_sql(query, conn)
        self.m1_data['timestamp'] = pd.to_datetime(self.m1_data['timestamp'])
        self.m1_data.set_index('timestamp', inplace=True)

        conn.close()

        # Construct M5 bars (5-minute aggregation)
        self.m5_data = self.aggregate_bars(self.m1_data, 5)

        # Construct H1 bars (60-minute aggregation)
        self.h1_data = self.aggregate_bars(self.m1_data, 60)

        print(f"Loaded {len(self.m1_data)} M1 bars")
        print(f"Created {len(self.m5_data)} M5 bars")
        print(f"Created {len(self.h1_data)} H1 bars")

    def aggregate_bars(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Aggregate M1 bars into higher timeframe bars"""
        # Create period groups based on bar_index
        data_copy = data.copy()
        data_copy['period_group'] = data_copy['bar_index'] // period

        # Aggregate OHLC data
        agg_data = data_copy.groupby('period_group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_index': 'first'  # Keep first bar_index for reference
        })

        # Set proper timestamp (using the first timestamp of each period)
        timestamps = data_copy.groupby('period_group').first().index
        agg_data.index = timestamps

        # Add synthetic bar numbers for tracking
        agg_data['bar_num'] = range(len(agg_data))

        return agg_data

    def analyze_swings(self):
        """Analyze swing states for both timeframes"""
        # Analyze M5 swings
        m5_results = self.m5_tracker.analyze(self.m5_data)
        self.m5_swings = m5_results['confirmed_swings']
        self.m5_potential = m5_results['potential_swings']

        # Create state timeline for M5
        self.m5_states = self._create_state_timeline(self.m5_data, m5_results['state_history'])

        # Analyze H1 swings
        h1_results = self.h1_tracker.analyze(self.h1_data)
        self.h1_swings = h1_results['confirmed_swings']
        self.h1_potential = h1_results['potential_swings']

        # Create state timeline for H1
        self.h1_states = self._create_state_timeline(self.h1_data, h1_results['state_history'])

        print(f"Found {len(self.m5_swings)} confirmed M5 swings")
        print(f"Found {len(self.h1_swings)} confirmed H1 swings")

    def _create_state_timeline(self, data, state_history):
        """Create a state array for each bar in the data"""
        states = [None] * len(data)

        for i, (bar_idx, state) in enumerate(state_history):
            # Find the position in data where this state starts
            if i < len(state_history) - 1:
                next_idx = state_history[i + 1][0]
                # Apply state to all bars between current and next state change
                for j in range(len(data)):
                    if data.iloc[j]['bar_index'] >= bar_idx and data.iloc[j]['bar_index'] < next_idx:
                        states[j] = state
            else:
                # Last state applies to all remaining bars
                for j in range(len(data)):
                    if data.iloc[j]['bar_index'] >= bar_idx:
                        states[j] = state

        return states

    def check_alignment(self, m5_state: SwingState, h1_state: SwingState) -> str:
        """Check alignment between M5 and H1 states"""
        # Strong alignment
        if h1_state == SwingState.HHHL and m5_state == SwingState.HHHL:
            return 'STRONG_BULLISH'
        if h1_state == SwingState.LHLL and m5_state == SwingState.LHLL:
            return 'STRONG_BEARISH'

        # Pullback/Rally opportunities
        if h1_state == SwingState.HHHL and m5_state in [SwingState.LHLL, SwingState.LHHL]:
            return 'BULLISH_PULLBACK'
        if h1_state == SwingState.LHLL and m5_state in [SwingState.HHHL, SwingState.LHHL]:
            return 'BEARISH_RALLY'

        # Neutral/Consolidation
        if h1_state in [SwingState.HHLL, SwingState.LHHL]:
            return 'CONSOLIDATION'

        return 'NEUTRAL'

    def plot_analysis(self, save_path: str = 'm5h1_swing_analysis.png'):
        """Create comprehensive plot with M5 and H1 analysis"""
        fig = plt.figure(figsize=(16, 12))

        # Create subplots
        ax1 = plt.subplot(3, 1, 1)  # H1 chart
        ax2 = plt.subplot(3, 1, 2)  # M5 chart
        ax3 = plt.subplot(3, 1, 3)  # Alignment indicator

        # Plot H1 candlesticks
        self.plot_candlesticks(ax1, self.h1_data, self.h1_swings, self.h1_states, 'H1 (60-min)')

        # Plot M5 candlesticks
        self.plot_candlesticks(ax2, self.m5_data, self.m5_swings, self.m5_states, 'M5 (5-min)')

        # Plot alignment zones
        self.plot_alignment(ax3)

        # Add legend for background colors
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='lightgreen', alpha=0.3, label='HHHL (Uptrend)'),
            plt.Rectangle((0, 0), 1, 1, fc='lightcoral', alpha=0.3, label='LHLL (Downtrend)'),
            plt.Rectangle((0, 0), 1, 1, fc='lightyellow', alpha=0.3, label='HHLL (Expansion)'),
            plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.3, label='LHHL (Contraction)')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=8)

        plt.suptitle('Multi-Timeframe Swing Analysis: M5 and H1 with Candlesticks', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Analysis saved to {save_path}")

    def plot_candlesticks(self, ax, data, swings, states, title):
        """Plot candlestick chart with swing markers"""
        # Create candlestick data
        x = range(len(data))

        # Plot candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            ax.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                  facecolor=color, edgecolor='black', alpha=0.8))
            # Wicks
            ax.plot([i, i], [row['low'], row['high']], color='black', linewidth=0.5)

        # Add background colors for states
        if states is not None:
            state_colors = {
                SwingState.HHHL: 'lightgreen',
                SwingState.LHLL: 'lightcoral',
                SwingState.HHLL: 'lightyellow',
                SwingState.LHHL: 'lightgray'
            }

            for i in range(len(data)):
                if i < len(states) and states[i] is not None:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.3, color=state_colors[states[i]])

        # Mark confirmed swings
        for swing in swings:
            # Find the x position for this swing using bar_index
            swing_bar_idx = swing.index
            # Find position in data where bar_index matches
            matching_rows = data[data['bar_index'] == swing_bar_idx]
            if not matching_rows.empty:
                x_pos = data.index.get_loc(matching_rows.index[0])
                if swing.type == 'high':
                    ax.scatter(x_pos, swing.price, marker='v', color='red', s=100, zorder=5)
                    ax.annotate(f'H', (x_pos, swing.price), xytext=(0, 5),
                              textcoords='offset points', ha='center', fontsize=8)
                else:
                    ax.scatter(x_pos, swing.price, marker='^', color='green', s=100, zorder=5)
                    ax.annotate(f'L', (x_pos, swing.price), xytext=(0, -15),
                              textcoords='offset points', ha='center', fontsize=8)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Price', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Set x-axis labels (show every Nth label to avoid crowding)
        step = max(1, len(data) // 10)
        ax.set_xticks(range(0, len(data), step))
        ax.set_xticklabels([str(data.index[i])[:16] for i in range(0, len(data), step)],
                          rotation=45, ha='right', fontsize=8)

    def plot_alignment(self, ax):
        """Plot alignment indicator between M5 and H1"""
        # Map M5 bars to H1 alignment
        alignments = []
        alignment_colors = {
            'STRONG_BULLISH': 'darkgreen',
            'BULLISH_PULLBACK': 'lightgreen',
            'STRONG_BEARISH': 'darkred',
            'BEARISH_RALLY': 'lightcoral',
            'CONSOLIDATION': 'yellow',
            'NEUTRAL': 'gray'
        }

        for i in range(len(self.m5_data)):
            # Find corresponding H1 state
            m5_timestamp = self.m5_data.index[i]
            h1_idx = self.h1_data.index.get_indexer([m5_timestamp], method='ffill')[0]

            if h1_idx >= 0 and h1_idx < len(self.h1_states) and i < len(self.m5_states):
                m5_state = self.m5_states[i]
                h1_state = self.h1_states[h1_idx]

                if m5_state is not None and h1_state is not None:
                    alignment = self.check_alignment(m5_state, h1_state)
                    alignments.append(alignment)
                else:
                    alignments.append('NEUTRAL')
            else:
                alignments.append('NEUTRAL')

        # Plot alignment as colored bars
        x = range(len(alignments))
        for i, alignment in enumerate(alignments):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.6, color=alignment_colors[alignment])

        ax.set_title('M5/H1 Alignment Zones', fontsize=10)
        ax.set_ylabel('Alignment', fontsize=9)
        ax.set_ylim(0, 1)

        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6, label=label)
            for label, color in alignment_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        # Set x-axis labels
        step = max(1, len(alignments) // 10)
        ax.set_xticks(range(0, len(alignments), step))
        ax.set_xticklabels([str(self.m5_data.index[i])[:16] for i in range(0, len(alignments), step)],
                          rotation=45, ha='right', fontsize=8)

        ax.grid(True, alpha=0.3)


def main():
    """Run M5/H1 multi-timeframe analysis"""
    analyzer = M5H1SwingAnalyzer()

    # Load and process data
    print("Loading data and constructing M5/H1 bars...")
    analyzer.load_data(start_idx=10000, end_idx=14000)

    # Analyze swings
    print("\nAnalyzing swing states...")
    analyzer.analyze_swings()

    # Create visualization
    print("\nCreating visualization...")
    analyzer.plot_analysis()

    # Print summary statistics
    print("\n=== M5 State Distribution ===")
    state_counts = pd.Series(analyzer.m5_states).value_counts()
    for state, count in state_counts.items():
        if state is not None:
            print(f"{state.name}: {count} bars ({count/len(analyzer.m5_states)*100:.1f}%)")

    print("\n=== H1 State Distribution ===")
    state_counts = pd.Series(analyzer.h1_states).value_counts()
    for state, count in state_counts.items():
        if state is not None:
            print(f"{state.name}: {count} bars ({count/len(analyzer.h1_states)*100:.1f}%)")


if __name__ == "__main__":
    main()