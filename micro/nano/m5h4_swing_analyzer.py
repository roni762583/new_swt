#!/usr/bin/env python3
"""
Multi-timeframe swing analyzer using M5 (5-minute) and H4 (240-minute) timeframes.
Tests if longer timeframe (H4) provides better trend context than H1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from swing_state_tracker import SwingState, SwingPoint, SwingStateTracker


class M5H4SwingAnalyzer:
    """Analyzes swing states across M5 and H4 timeframes"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.m1_data = None
        self.m5_data = None
        self.h4_data = None
        self.m5_tracker = SwingStateTracker(k=3)  # k=3 for M5 swing detection
        self.h4_tracker = SwingStateTracker(k=3)  # k=3 for H4 swing detection

    def load_data(self, start_idx: int = 10000, end_idx: int = 14000):
        """Load M1 data and construct M5 and H4 bars
        Note: Using wider range for H4 to get enough bars"""
        conn = duckdb.connect(self.db_path, read_only=True)

        # Load M1 data from master table
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

        # Construct H4 bars (240-minute aggregation)
        self.h4_data = self.aggregate_bars(self.m1_data, 240)

        print(f"Loaded {len(self.m1_data)} M1 bars")
        print(f"Created {len(self.m5_data)} M5 bars")
        print(f"Created {len(self.h4_data)} H4 bars")

    def aggregate_bars(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Aggregate M1 bars into higher timeframe bars"""
        data_copy = data.copy()
        data_copy['period_group'] = data_copy['bar_index'] // period

        # Aggregate OHLC data
        agg_data = data_copy.groupby('period_group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_index': 'first'
        })

        # Set proper timestamp
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
        self.m5_states = self._create_state_timeline(self.m5_data, m5_results['state_history'])

        # Analyze H4 swings
        h4_results = self.h4_tracker.analyze(self.h4_data)
        self.h4_swings = h4_results['confirmed_swings']
        self.h4_potential = h4_results['potential_swings']
        self.h4_states = self._create_state_timeline(self.h4_data, h4_results['state_history'])

        print(f"Found {len(self.m5_swings)} confirmed M5 swings")
        print(f"Found {len(self.h4_swings)} confirmed H4 swings")

    def _create_state_timeline(self, data, state_history):
        """Create a state array for each bar in the data"""
        states = [None] * len(data)

        for i, (bar_idx, state) in enumerate(state_history):
            if i < len(state_history) - 1:
                next_idx = state_history[i + 1][0]
                for j in range(len(data)):
                    if data.iloc[j]['bar_index'] >= bar_idx and data.iloc[j]['bar_index'] < next_idx:
                        states[j] = state
            else:
                for j in range(len(data)):
                    if data.iloc[j]['bar_index'] >= bar_idx:
                        states[j] = state

        return states

    def check_alignment(self, m5_state: SwingState, h4_state: SwingState) -> str:
        """Check alignment between M5 and H4 states"""
        # Strong alignment
        if h4_state == SwingState.HHHL and m5_state == SwingState.HHHL:
            return 'STRONG_BULLISH'
        if h4_state == SwingState.LHLL and m5_state == SwingState.LHLL:
            return 'STRONG_BEARISH'

        # Pullback/Rally opportunities (H4 trend with M5 counter-move)
        if h4_state == SwingState.HHHL and m5_state in [SwingState.LHLL, SwingState.LHHL]:
            return 'BULLISH_PULLBACK'
        if h4_state == SwingState.LHLL and m5_state in [SwingState.HHHL, SwingState.LHHL]:
            return 'BEARISH_RALLY'

        # H4 Consolidation
        if h4_state in [SwingState.HHLL, SwingState.LHHL]:
            return 'CONSOLIDATION'

        return 'NEUTRAL'

    def find_trade_opportunities(self) -> List[Dict]:
        """Find potential trade setups based on M5/H4 alignment"""
        opportunities = []

        for i in range(20, len(self.m5_data) - 1):  # Start after some bars for context
            m5_state = self.m5_states[i] if i < len(self.m5_states) else None

            # Find corresponding H4 state
            m5_timestamp = self.m5_data.index[i]
            h4_idx = self.h4_data.index.get_indexer([m5_timestamp], method='ffill')[0]
            h4_state = self.h4_states[h4_idx] if h4_idx >= 0 and h4_idx < len(self.h4_states) else None

            if m5_state is None or h4_state is None:
                continue

            alignment = self.check_alignment(m5_state, h4_state)

            # Track high-probability setups
            if alignment in ['STRONG_BULLISH', 'BULLISH_PULLBACK', 'STRONG_BEARISH', 'BEARISH_RALLY']:
                opportunities.append({
                    'bar_index': self.m5_data.iloc[i]['bar_index'],
                    'timestamp': m5_timestamp,
                    'alignment': alignment,
                    'm5_state': m5_state.value,
                    'h4_state': h4_state.value,
                    'price': self.m5_data.iloc[i]['close']
                })

        return opportunities

    def plot_analysis(self, save_path: str = 'm5h4_swing_analysis.png'):
        """Create comprehensive plot with M5 and H4 analysis"""
        fig = plt.figure(figsize=(16, 12))

        # Create subplots
        ax1 = plt.subplot(3, 1, 1)  # H4 chart
        ax2 = plt.subplot(3, 1, 2)  # M5 chart
        ax3 = plt.subplot(3, 1, 3)  # Alignment indicator

        # Plot H4 candlesticks
        self.plot_candlesticks(ax1, self.h4_data, self.h4_swings, self.h4_states, 'H4 (4-hour)')

        # Plot M5 candlesticks (sample for visibility)
        m5_sample = self.m5_data.iloc[-200:]  # Last 200 M5 bars
        m5_swings_sample = [s for s in self.m5_swings if s.index >= m5_sample.iloc[0]['bar_index']]
        m5_states_sample = self.m5_states[-200:] if len(self.m5_states) >= 200 else self.m5_states
        self.plot_candlesticks(ax2, m5_sample, m5_swings_sample, m5_states_sample, 'M5 (5-min) - Last 200 bars')

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

        plt.suptitle('Multi-Timeframe Swing Analysis: M5 and H4', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Analysis saved to {save_path}")

    def plot_candlesticks(self, ax, data, swings, states, title):
        """Plot candlestick chart with swing markers"""
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
            swing_bar_idx = swing.index
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

        # Set x-axis labels
        step = max(1, len(data) // 10)
        ax.set_xticks(range(0, len(data), step))
        ax.set_xticklabels([str(data.index[i])[:16] for i in range(0, len(data), step)],
                          rotation=45, ha='right', fontsize=8)

    def plot_alignment(self, ax):
        """Plot alignment indicator between M5 and H4"""
        alignments = []
        alignment_colors = {
            'STRONG_BULLISH': 'darkgreen',
            'BULLISH_PULLBACK': 'lightgreen',
            'STRONG_BEARISH': 'darkred',
            'BEARISH_RALLY': 'lightcoral',
            'CONSOLIDATION': 'yellow',
            'NEUTRAL': 'gray'
        }

        # Sample last 200 M5 bars for visibility
        m5_sample = self.m5_data.iloc[-200:]

        for i in range(len(m5_sample)):
            m5_timestamp = m5_sample.index[i]
            h4_idx = self.h4_data.index.get_indexer([m5_timestamp], method='ffill')[0]

            sample_offset = len(self.m5_data) - 200
            m5_state_idx = sample_offset + i

            if (h4_idx >= 0 and h4_idx < len(self.h4_states) and
                m5_state_idx < len(self.m5_states)):
                m5_state = self.m5_states[m5_state_idx]
                h4_state = self.h4_states[h4_idx]

                if m5_state is not None and h4_state is not None:
                    alignment = self.check_alignment(m5_state, h4_state)
                    alignments.append(alignment)
                else:
                    alignments.append('NEUTRAL')
            else:
                alignments.append('NEUTRAL')

        # Plot alignment as colored bars
        x = range(len(alignments))
        for i, alignment in enumerate(alignments):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.6, color=alignment_colors[alignment])

        ax.set_title('M5/H4 Alignment Zones (Last 200 M5 bars)', fontsize=10)
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
        ax.set_xticklabels([str(m5_sample.index[i])[:16] for i in range(0, len(alignments), step)],
                          rotation=45, ha='right', fontsize=8)

        ax.grid(True, alpha=0.3)

    def generate_statistics(self):
        """Generate and print statistics about the analysis"""
        print("\n=== M5 State Distribution ===")
        if self.m5_states:
            state_counts = pd.Series([s for s in self.m5_states if s is not None]).value_counts()
            for state, count in state_counts.items():
                print(f"{state.name}: {count} bars ({count/len(self.m5_states)*100:.1f}%)")

        print("\n=== H4 State Distribution ===")
        if self.h4_states:
            state_counts = pd.Series([s for s in self.h4_states if s is not None]).value_counts()
            for state, count in state_counts.items():
                print(f"{state.name}: {count} bars ({count/len(self.h4_states)*100:.1f}%)")

        # Find trade opportunities
        opportunities = self.find_trade_opportunities()

        print(f"\n=== Trade Opportunities Found ===")
        print(f"Total opportunities: {len(opportunities)}")

        # Group by alignment type
        alignment_counts = {}
        for opp in opportunities:
            alignment = opp['alignment']
            if alignment not in alignment_counts:
                alignment_counts[alignment] = 0
            alignment_counts[alignment] += 1

        for alignment, count in alignment_counts.items():
            print(f"{alignment}: {count} opportunities")

        print("\n=== Comparison: M5/H1 vs M5/H4 ===")
        print("M5/H4 Advantages:")
        print("• H4 provides stronger trend context (less noise)")
        print("• Fewer false signals from H4 stability")
        print("• Better for position/swing trading")
        print("• Clearer major support/resistance levels")

        print("\nM5/H4 Disadvantages:")
        print("• H4 swings take much longer to confirm")
        print("• Fewer H4 state changes = less context updates")
        print("• May miss shorter-term opportunities")
        print("• Requires more historical data for H4 analysis")


def main():
    """Run M5/H4 multi-timeframe analysis"""
    analyzer = M5H4SwingAnalyzer()

    # Load and process data
    print("Loading data and constructing M5/H4 bars...")
    analyzer.load_data(start_idx=10000, end_idx=14000)

    # Analyze swings
    print("\nAnalyzing swing states...")
    analyzer.analyze_swings()

    # Create visualization
    print("\nCreating visualization...")
    analyzer.plot_analysis()

    # Generate statistics
    analyzer.generate_statistics()


if __name__ == "__main__":
    main()