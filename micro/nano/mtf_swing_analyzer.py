#!/usr/bin/env python3
"""
Multi-Timeframe Swing State Analyzer
Uses H1 (60-minute) bars for major trend direction
Uses M1 (1-minute) bars for timing entries
"""

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from swing_state_tracker import SwingStateTracker, SwingState
from typing import Dict, List, Tuple


class MTFSwingAnalyzer:
    """Multi-timeframe swing state analyzer."""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def load_m1_data(self, start_bar: int = 10000, limit: int = 3600):
        """Load M1 (1-minute) OHLC data."""
        query = f"""
        SELECT
            bar_index,
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM master
        WHERE bar_index > {start_bar}
        ORDER BY bar_index
        LIMIT {limit}
        """

        df = self.conn.execute(query).df()
        df.set_index('bar_index', inplace=True)
        return df

    def aggregate_to_h1(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """Aggregate M1 data to H1 (60-minute) timeframe."""
        # Group every 60 bars
        df_m1['h1_group'] = (df_m1.index - df_m1.index[0]) // 60

        # Aggregate OHLCV
        df_h1 = df_m1.groupby('h1_group').agg({
            'timestamp': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Create H1 bar indices
        df_h1.index = df_h1.index * 60 + df_m1.index[0]
        df_h1.index.name = 'h1_bar_index'

        return df_h1

    def analyze_both_timeframes(self, df_m1: pd.DataFrame, df_h1: pd.DataFrame) -> Dict:
        """Run swing analysis on both timeframes."""

        # M1 Analysis (k=3 for faster swings)
        tracker_m1 = SwingStateTracker(k=3)
        results_m1 = tracker_m1.analyze(df_m1)

        # H1 Analysis (k=2 for H1 since fewer bars)
        tracker_h1 = SwingStateTracker(k=2)
        results_h1 = tracker_h1.analyze(df_h1)

        return {
            'm1': {
                'tracker': tracker_m1,
                'results': results_m1,
                'df': df_m1
            },
            'h1': {
                'tracker': tracker_h1,
                'results': results_h1,
                'df': df_h1
            }
        }

    def identify_trading_zones(self, mtf_data: Dict) -> List[Dict]:
        """Identify high-probability trading zones based on MTF alignment."""
        zones = []

        h1_state_history = mtf_data['h1']['results']['state_history']
        m1_state_history = mtf_data['m1']['results']['state_history']

        # Create state maps for quick lookup
        h1_states = {}
        for i, (idx, state) in enumerate(h1_state_history):
            next_idx = h1_state_history[i+1][0] if i+1 < len(h1_state_history) else float('inf')
            h1_states[idx] = (state, next_idx)

        # Find aligned zones
        for m1_idx, m1_state in m1_state_history:
            # Find corresponding H1 state
            h1_state = None
            for h1_idx, (state, next_h1) in h1_states.items():
                if h1_idx <= m1_idx < next_h1:
                    h1_state = state
                    break

            if h1_state:
                # Check for alignment
                alignment = self.check_alignment(m1_state, h1_state)
                if alignment:
                    zones.append({
                        'bar_index': m1_idx,
                        'm1_state': m1_state.value,
                        'h1_state': h1_state.value,
                        'alignment': alignment
                    })

        return zones

    def check_alignment(self, m1_state: SwingState, h1_state: SwingState) -> str:
        """Check if M1 and H1 states are aligned for trading."""
        # Strong bullish alignment
        if h1_state == SwingState.HHHL and m1_state == SwingState.HHHL:
            return 'STRONG_BULLISH'

        # Strong bearish alignment
        if h1_state == SwingState.LHLL and m1_state == SwingState.LHLL:
            return 'STRONG_BEARISH'

        # H1 uptrend with M1 pullback (buy opportunity)
        if h1_state == SwingState.HHHL and m1_state in [SwingState.LHLL, SwingState.LHHL]:
            return 'BULLISH_PULLBACK'

        # H1 downtrend with M1 rally (sell opportunity)
        if h1_state == SwingState.LHLL and m1_state in [SwingState.HHHL, SwingState.LHHL]:
            return 'BEARISH_RALLY'

        return None

    def plot_mtf_analysis(self, mtf_data: Dict, save_path: str = 'mtf_swing_analysis.png'):
        """Create comprehensive multi-timeframe visualization with candlestick charts."""

        # Import additional libraries for candlesticks
        from matplotlib.patches import Rectangle

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(5, 1, height_ratios=[2, 1, 2, 1, 1], hspace=0.3)

        # H1 Price and Swings
        ax_h1_price = fig.add_subplot(gs[0])
        # H1 State Timeline
        ax_h1_state = fig.add_subplot(gs[1], sharex=ax_h1_price)
        # M1 Price and Swings
        ax_m1_price = fig.add_subplot(gs[2])
        # M1 State Timeline
        ax_m1_state = fig.add_subplot(gs[3], sharex=ax_m1_price)
        # Alignment Zones
        ax_align = fig.add_subplot(gs[4], sharex=ax_m1_price)

        # Plot H1 data with candlesticks
        self.plot_timeframe_with_candles(ax_h1_price, ax_h1_state, mtf_data['h1'], 'H1 (60-min) Candlestick Chart')

        # Plot M1 data with candlesticks (simplified for performance)
        self.plot_timeframe_with_candles(ax_m1_price, ax_m1_state, mtf_data['m1'], 'M1 (1-min) Candlestick Chart', simplify=True)

        # Plot alignment zones
        zones = self.identify_trading_zones(mtf_data)
        self.plot_alignment_zones(ax_align, zones, mtf_data['m1']['df'])

        # Add title and save
        fig.suptitle('Multi-Timeframe Swing State Analysis with Candlestick Charts\nH1 for Direction, M1 for Timing',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"📊 MTF analysis saved to {save_path}")

        return zones

    def plot_candlesticks(self, ax, df_plot, width=0.8, simplify=False):
        """Draw candlestick chart."""
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D

        # Simplify by showing every nth candle for M1
        step = 10 if simplify else 1

        for i in range(0, len(df_plot), step):
            row = df_plot.iloc[i]

            # Determine color
            color = 'green' if row['close'] >= row['open'] else 'red'
            edge_color = 'darkgreen' if row['close'] >= row['open'] else 'darkred'

            # Draw high-low line (wick)
            ax.plot([i, i], [row['low'], row['high']],
                   color=edge_color, linewidth=0.5, alpha=0.7)

            # Draw open-close rectangle (body)
            height = abs(row['close'] - row['open'])
            bottom = min(row['open'], row['close'])

            rect = Rectangle((i - width/2, bottom), width, height,
                           facecolor=color, edgecolor=edge_color,
                           alpha=0.7 if color == 'green' else 0.6,
                           linewidth=0.5)
            ax.add_patch(rect)

    def plot_timeframe_with_candles(self, ax_price, ax_state, tf_data, title, simplify=False):
        """Plot candlestick chart with swing points and state background."""
        df = tf_data['df']
        tracker = tf_data['tracker']
        results = tf_data['results']

        # Reset index for plotting
        df_plot = df.reset_index(drop=True)

        # Draw candlesticks
        self.plot_candlesticks(ax_price, df_plot, width=0.6 if simplify else 0.8, simplify=simplify)

        # Create index mapping
        original_index = df.index.tolist()
        index_map = {orig: i for i, orig in enumerate(original_index)}

        # Plot confirmed swings
        for swing in tracker.confirmed_swings:
            if swing.index in index_map:
                plot_idx = index_map[swing.index]
                marker = 'v' if swing.type == 'high' else '^'
                color = 'darkred' if swing.type == 'high' else 'darkgreen'
                ax_price.scatter(plot_idx, swing.price, marker=marker,
                               color=color, s=100, alpha=0.9, zorder=5,
                               edgecolors='black', linewidth=1)

        # Add state backgrounds with proper legend
        if tracker.state_history:
            # Define regime colors and meanings
            regime_colors = {
                SwingState.HHHL: ('lightgreen', 'HHHL: Uptrend (Higher High, Higher Low)'),
                SwingState.LHLL: ('lightcoral', 'LHLL: Downtrend (Lower High, Lower Low)'),
                SwingState.HHLL: ('lightyellow', 'HHLL: Expansion (Higher High, Lower Low)'),
                SwingState.LHHL: ('lightblue', 'LHHL: Contraction (Lower High, Higher Low)')
            }

            for i, (idx, state) in enumerate(tracker.state_history):
                if idx in index_map:
                    plot_start = index_map[idx]
                    if i+1 < len(tracker.state_history):
                        next_idx = tracker.state_history[i+1][0]
                        plot_end = index_map.get(next_idx, len(df_plot))
                    else:
                        plot_end = len(df_plot)

                    color, _ = regime_colors.get(state, ('gray', 'Unknown'))
                    ax_price.axvspan(plot_start, plot_end, alpha=0.2, color=color)

            # Create custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            # Regime patches
            regime_patches = [Patch(facecolor=color, alpha=0.2, label=label)
                            for state, (color, label) in regime_colors.items()]

            # Candlestick elements
            candle_elements = [
                Patch(facecolor='green', alpha=0.7, edgecolor='darkgreen', label='Bullish Candle'),
                Patch(facecolor='red', alpha=0.6, edgecolor='darkred', label='Bearish Candle')
            ]

            # Swing markers
            swing_markers = [
                Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred',
                      markersize=10, label='Swing High', markeredgecolor='black'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen',
                      markersize=10, label='Swing Low', markeredgecolor='black')
            ]

            # Combine all legend elements
            all_handles = regime_patches + candle_elements + swing_markers
            ax_price.legend(handles=all_handles, loc='upper left', ncol=2,
                          framealpha=0.9, fontsize=8)

        ax_price.set_title(f'{title} - Current State: {tracker.current_state.value}')
        ax_price.set_ylabel('Price')
        ax_price.grid(True, alpha=0.3, linestyle='--')
        ax_price.set_xlim(-1, len(df_plot))

        # Plot state timeline
        self.plot_state_timeline(ax_state, tracker, index_map)

    def plot_state_timeline(self, ax_state, tracker, index_map):
        """Plot the state timeline."""
        if tracker.state_history:
            states_numeric = {'HHHL': 2, 'LHLL': -2, 'HHLL': 1, 'LHHL': -1}
            x = [index_map.get(s[0], 0) for s in tracker.state_history if s[0] in index_map]
            y = [states_numeric.get(s[1].value, 0) for s in tracker.state_history if s[0] in index_map]

            if x and y:
                ax_state.step(x, y, where='post', linewidth=2, color='blue')
                ax_state.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_state.set_ylim(-3, 3)
                ax_state.set_yticks([-2, -1, 0, 1, 2])
                ax_state.set_yticklabels(['LHLL', 'LHHL', '-', 'HHLL', 'HHHL'])
                ax_state.set_ylabel('State')
                ax_state.grid(True, alpha=0.3)

    def plot_timeframe(self, ax_price, ax_state, tf_data, title):
        """Plot price and state for a single timeframe."""
        df = tf_data['df']
        tracker = tf_data['tracker']
        results = tf_data['results']

        # Reset index for plotting
        df_plot = df.reset_index(drop=True)

        # Plot price
        ax_price.plot(df_plot.index, df_plot['close'], 'k-', linewidth=0.5, alpha=0.7)

        # Create index mapping
        original_index = df.index.tolist()
        index_map = {orig: i for i, orig in enumerate(original_index)}

        # Plot confirmed swings
        for swing in tracker.confirmed_swings:
            if swing.index in index_map:
                plot_idx = index_map[swing.index]
                marker = 'v' if swing.type == 'high' else '^'
                color = 'red' if swing.type == 'high' else 'green'
                ax_price.plot(plot_idx, swing.price, marker, color=color, markersize=8, alpha=0.8)

        # Add state backgrounds
        if tracker.state_history:
            colors = {
                SwingState.HHHL: 'lightgreen',
                SwingState.LHLL: 'lightcoral',
                SwingState.HHLL: 'lightyellow',
                SwingState.LHHL: 'lightblue'
            }

            for i, (idx, state) in enumerate(tracker.state_history):
                if idx in index_map:
                    plot_start = index_map[idx]
                    if i+1 < len(tracker.state_history):
                        next_idx = tracker.state_history[i+1][0]
                        plot_end = index_map.get(next_idx, len(df_plot))
                    else:
                        plot_end = len(df_plot)
                    ax_price.axvspan(plot_start, plot_end, alpha=0.2, color=colors.get(state, 'gray'))

        ax_price.set_title(f'{title} - Current: {tracker.current_state.value}')
        ax_price.set_ylabel('Price')
        ax_price.grid(True, alpha=0.3)

        # Plot state timeline
        if tracker.state_history:
            states_numeric = {'HHHL': 2, 'LHLL': -2, 'HHLL': 1, 'LHHL': -1}
            x = [index_map.get(s[0], 0) for s in tracker.state_history if s[0] in index_map]
            y = [states_numeric.get(s[1].value, 0) for s in tracker.state_history if s[0] in index_map]

            if x and y:
                ax_state.step(x, y, where='post', linewidth=2, color='blue')
                ax_state.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax_state.set_ylim(-3, 3)
                ax_state.set_yticks([-2, -1, 0, 1, 2])
                ax_state.set_yticklabels(['LHLL', 'LHHL', '-', 'HHLL', 'HHHL'])
                ax_state.set_ylabel('State')
                ax_state.grid(True, alpha=0.3)

    def plot_alignment_zones(self, ax, zones, df_m1):
        """Plot trading alignment zones."""
        # Create index mapping for M1
        original_index = df_m1.index.tolist()
        index_map = {orig: i for i, orig in enumerate(original_index)}

        # Color map for alignment types
        colors = {
            'STRONG_BULLISH': 'darkgreen',
            'STRONG_BEARISH': 'darkred',
            'BULLISH_PULLBACK': 'lightgreen',
            'BEARISH_RALLY': 'lightcoral'
        }

        # Plot zones
        zone_data = []
        for zone in zones:
            if zone['bar_index'] in index_map:
                x = index_map[zone['bar_index']]
                color = colors.get(zone['alignment'], 'gray')
                ax.axvline(x, color=color, alpha=0.3, linewidth=2)
                zone_data.append((x, zone['alignment']))

        # Add legend
        legend_elements = [
            mpatches.Patch(color='darkgreen', alpha=0.5, label='Strong Bullish'),
            mpatches.Patch(color='darkred', alpha=0.5, label='Strong Bearish'),
            mpatches.Patch(color='lightgreen', alpha=0.5, label='Bullish Pullback'),
            mpatches.Patch(color='lightcoral', alpha=0.5, label='Bearish Rally')
        ]
        ax.legend(handles=legend_elements, loc='upper left', ncol=4)

        ax.set_title('Trading Alignment Zones (H1 + M1)')
        ax.set_xlabel('Bar Index')
        ax.set_ylabel('Alignment')
        ax.grid(True, alpha=0.3)

        # Add text summary
        if zones:
            strong_bull = sum(1 for z in zones if z['alignment'] == 'STRONG_BULLISH')
            strong_bear = sum(1 for z in zones if z['alignment'] == 'STRONG_BEARISH')
            bull_pb = sum(1 for z in zones if z['alignment'] == 'BULLISH_PULLBACK')
            bear_rally = sum(1 for z in zones if z['alignment'] == 'BEARISH_RALLY')

            text = f"Zones: Bull:{strong_bull} Bear:{strong_bear} BullPB:{bull_pb} BearRally:{bear_rally}"
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def main():
    """Run multi-timeframe analysis."""
    print("🔍 MULTI-TIMEFRAME SWING ANALYSIS")
    print("="*70)

    # Create analyzer
    analyzer = MTFSwingAnalyzer()

    # Load M1 data (60 hours = 3600 minutes)
    print("Loading M1 data...")
    df_m1 = analyzer.load_m1_data(start_bar=10000, limit=3600)
    print(f"  Loaded {len(df_m1)} M1 bars")
    print(f"  Time range: {df_m1['timestamp'].min()} to {df_m1['timestamp'].max()}")

    # Aggregate to H1
    print("\nAggregating to H1...")
    df_h1 = analyzer.aggregate_to_h1(df_m1)
    print(f"  Created {len(df_h1)} H1 bars")

    # Analyze both timeframes
    print("\nAnalyzing timeframes...")

    # Update todo
    mtf_data = analyzer.analyze_both_timeframes(df_m1, df_h1)

    # Print results
    print("\n📊 M1 ANALYSIS:")
    m1_stats = mtf_data['m1']['results']['stats']
    print(f"  Confirmed swings: {m1_stats['confirmed_swings']}")
    print(f"  Current state: {m1_stats['current_state']}")
    print(f"  State changes: {m1_stats['state_changes']}")

    print("\n📊 H1 ANALYSIS:")
    h1_stats = mtf_data['h1']['results']['stats']
    print(f"  Confirmed swings: {h1_stats['confirmed_swings']}")
    print(f"  Current state: {h1_stats['current_state']}")
    print(f"  State changes: {h1_stats['state_changes']}")

    # Create visualization and identify zones
    print("\n🎨 Creating visualization...")
    zones = analyzer.plot_mtf_analysis(mtf_data)

    # Print trading zones summary
    if zones:
        print(f"\n🎯 TRADING ZONES IDENTIFIED: {len(zones)}")
        alignment_counts = {}
        for zone in zones:
            alignment_counts[zone['alignment']] = alignment_counts.get(zone['alignment'], 0) + 1

        for alignment, count in alignment_counts.items():
            print(f"  {alignment}: {count} zones")
    else:
        print("\n⚠️ No aligned trading zones found")

    print("\n✅ Multi-timeframe analysis complete!")
    return mtf_data, zones


if __name__ == "__main__":
    mtf_data, zones = main()