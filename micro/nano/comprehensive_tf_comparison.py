#!/usr/bin/env python3
"""
Comprehensive Timeframe Comparison using entire dataset
Memory-efficient implementation using chunking and incremental processing
"""

import pandas as pd
import numpy as np
import duckdb
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass
from swing_state_tracker import SwingStateTracker, SwingState


@dataclass
class TFResults:
    """Results for a specific timeframe combination"""
    tf_pair: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pips: float = 0
    avg_win_pips: float = 0
    avg_loss_pips: float = 0
    win_rate: float = 0
    profit_factor: float = 0
    avg_bars_held: float = 0
    total_opportunities: int = 0
    execution_time: float = 0


class EfficientTFComparison:
    """Memory-efficient timeframe comparison analyzer"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.pip_multiplier = 100  # JPY pairs
        self.results = {}

    def get_data_range(self) -> Tuple[int, int]:
        """Get the full range of available data"""
        conn = duckdb.connect(self.db_path, read_only=True)

        query = """
        SELECT MIN(bar_index) as min_idx,
               MAX(bar_index) as max_idx,
               COUNT(*) as total_bars
        FROM master
        """

        result = conn.execute(query).fetchone()
        conn.close()

        print(f"Dataset range: {result[0]} to {result[1]} ({result[2]:,} total bars)")
        return result[0], result[1]

    def process_timeframe_pair(self, exec_tf: int, context_tf: int,
                               chunk_size: int = 50000) -> TFResults:
        """Process a single timeframe pair efficiently using chunks"""

        tf_name = f"M{exec_tf}/M{context_tf}" if context_tf < 60 else f"M{exec_tf}/H{context_tf//60}"
        print(f"\nProcessing {tf_name}...")

        results = TFResults(tf_pair=tf_name)

        # Get data range
        min_idx, max_idx = self.get_data_range()

        # Process in chunks to manage memory
        all_trades = []
        all_opportunities = []

        # We need enough data for the context timeframe
        min_required = context_tf * 10  # At least 10 context bars

        for chunk_start in range(min_idx, max_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size + context_tf * 20, max_idx)

            if chunk_end - chunk_start < min_required:
                continue

            # Process chunk
            trades, opportunities = self.analyze_chunk(
                chunk_start, chunk_end, exec_tf, context_tf
            )

            all_trades.extend(trades)
            all_opportunities.extend(opportunities)

            # Free memory
            gc.collect()

            # Progress indicator
            progress = (chunk_start - min_idx) / (max_idx - min_idx) * 100
            print(f"  Progress: {progress:.1f}%", end='\r')

        print(f"  Complete! Found {len(all_trades)} trades")

        # Calculate statistics
        if all_trades:
            results.total_trades = len(all_trades)
            results.winning_trades = len([t for t in all_trades if t['pips'] > 0])
            results.total_pips = sum(t['pips'] for t in all_trades)

            wins = [t['pips'] for t in all_trades if t['pips'] > 0]
            losses = [t['pips'] for t in all_trades if t['pips'] <= 0]

            results.avg_win_pips = np.mean(wins) if wins else 0
            results.avg_loss_pips = np.mean(losses) if losses else 0
            results.win_rate = len(wins) / len(all_trades) * 100

            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            results.profit_factor = total_wins / total_losses

            results.avg_bars_held = np.mean([t['bars_held'] for t in all_trades])

        results.total_opportunities = len(all_opportunities)

        return results

    def analyze_chunk(self, start_idx: int, end_idx: int,
                     exec_tf: int, context_tf: int) -> Tuple[List, List]:
        """Analyze a single chunk of data"""

        conn = duckdb.connect(self.db_path, read_only=True)

        # Load chunk
        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {end_idx}
        ORDER BY bar_index
        """

        m1_data = pd.read_sql(query, conn)
        conn.close()

        if len(m1_data) < context_tf * 4:
            return [], []

        # Aggregate to required timeframes
        exec_data = self.aggregate_data(m1_data, exec_tf)
        context_data = self.aggregate_data(m1_data, context_tf)

        if len(exec_data) < 20 or len(context_data) < 4:
            return [], []

        # Analyze swings
        exec_tracker = SwingStateTracker(k=3)
        context_tracker = SwingStateTracker(k=3)

        exec_results = exec_tracker.analyze(exec_data)
        context_results = context_tracker.analyze(context_data)

        # Find alignments and simulate trades
        trades = self.simulate_trades_for_chunk(
            exec_data, exec_results,
            context_data, context_results
        )

        opportunities = self.find_opportunities(
            exec_data, exec_results['state_history'],
            context_data, context_results['state_history']
        )

        return trades, opportunities

    def aggregate_data(self, m1_data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Aggregate M1 data to higher timeframe"""
        if period == 1:
            return m1_data.copy()

        m1_data['group'] = m1_data['bar_index'] // period

        agg = m1_data.groupby('group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_index': 'first'
        }).reset_index(drop=True)

        return agg

    def simulate_trades_for_chunk(self, exec_data, exec_results,
                                  context_data, context_results) -> List[Dict]:
        """Simulate trades for this chunk"""
        trades = []

        exec_states = self.create_state_array(exec_data, exec_results['state_history'])
        context_states = self.create_state_array(context_data, context_results['state_history'])

        in_position = False
        current_trade = None

        for i in range(20, len(exec_data) - 1):
            if in_position:
                # Check exit
                if current_trade:
                    # Simple exit after 20 bars or 10% of data
                    if i - current_trade['entry_idx'] > min(20, len(exec_data) * 0.1):
                        exit_price = exec_data.iloc[i]['close']

                        if current_trade['direction'] == 'long':
                            pips = (exit_price - current_trade['entry_price']) * self.pip_multiplier
                        else:
                            pips = (current_trade['entry_price'] - exit_price) * self.pip_multiplier

                        current_trade['pips'] = pips
                        current_trade['bars_held'] = i - current_trade['entry_idx']
                        trades.append(current_trade)
                        in_position = False
                        current_trade = None
                continue

            # Check entry conditions
            exec_state = exec_states[i] if i < len(exec_states) else None

            # Map to context timeframe
            exec_bar = exec_data.iloc[i]['bar_index']
            context_idx = None
            for j, ctx_bar in enumerate(context_data['bar_index']):
                if ctx_bar <= exec_bar:
                    context_idx = j
                else:
                    break

            if context_idx is None or context_idx >= len(context_states):
                continue

            context_state = context_states[context_idx]

            if exec_state is None or context_state is None:
                continue

            # Entry logic
            alignment = self.check_alignment(exec_state, context_state)

            if alignment in ['STRONG_BULLISH', 'BULLISH_PULLBACK']:
                current_trade = {
                    'entry_idx': i,
                    'entry_price': exec_data.iloc[i]['close'],
                    'direction': 'long',
                    'alignment': alignment
                }
                in_position = True

            elif alignment in ['STRONG_BEARISH', 'BEARISH_RALLY']:
                current_trade = {
                    'entry_idx': i,
                    'entry_price': exec_data.iloc[i]['close'],
                    'direction': 'short',
                    'alignment': alignment
                }
                in_position = True

        return trades

    def find_opportunities(self, exec_data, exec_history,
                          context_data, context_history) -> List:
        """Count trading opportunities"""
        opportunities = []

        exec_states = self.create_state_array(exec_data, exec_history)
        context_states = self.create_state_array(context_data, context_history)

        for i in range(len(exec_states)):
            exec_state = exec_states[i]

            # Map to context
            exec_bar = exec_data.iloc[i]['bar_index']
            context_idx = None
            for j, ctx_bar in enumerate(context_data['bar_index']):
                if ctx_bar <= exec_bar:
                    context_idx = j
                else:
                    break

            if context_idx is None or context_idx >= len(context_states):
                continue

            context_state = context_states[context_idx]

            if exec_state and context_state:
                alignment = self.check_alignment(exec_state, context_state)
                if alignment in ['STRONG_BULLISH', 'BULLISH_PULLBACK',
                               'STRONG_BEARISH', 'BEARISH_RALLY']:
                    opportunities.append(alignment)

        return opportunities

    def create_state_array(self, data, state_history):
        """Create state array from history"""
        states = [None] * len(data)

        for i, (bar_idx, state) in enumerate(state_history):
            start_idx = None
            for j, row_idx in enumerate(data['bar_index']):
                if row_idx >= bar_idx:
                    start_idx = j
                    break

            if start_idx is not None:
                if i < len(state_history) - 1:
                    next_bar = state_history[i + 1][0]
                    for j in range(start_idx, len(data)):
                        if data.iloc[j]['bar_index'] < next_bar:
                            states[j] = state
                        else:
                            break
                else:
                    for j in range(start_idx, len(data)):
                        states[j] = state

        return states

    def check_alignment(self, exec_state, context_state):
        """Check alignment between execution and context timeframes"""
        if context_state == SwingState.HHHL and exec_state == SwingState.HHHL:
            return 'STRONG_BULLISH'
        if context_state == SwingState.LHLL and exec_state == SwingState.LHLL:
            return 'STRONG_BEARISH'
        if context_state == SwingState.HHHL and exec_state in [SwingState.LHLL, SwingState.LHHL]:
            return 'BULLISH_PULLBACK'
        if context_state == SwingState.LHLL and exec_state in [SwingState.HHHL, SwingState.LHHL]:
            return 'BEARISH_RALLY'
        return 'NEUTRAL'

    def run_all_comparisons(self):
        """Run all timeframe combinations"""

        # Define timeframe combinations to test
        # (execution_tf, context_tf) in minutes
        tf_combinations = [
            (1, 60),    # M1/H1 - Original best
            (1, 240),   # M1/H4
            (5, 60),    # M5/H1
            (5, 240),   # M5/H4
            (15, 60),   # M15/H1
            (15, 240),  # M15/H4
            (30, 240),  # M30/H4
        ]

        print("="*60)
        print("COMPREHENSIVE TIMEFRAME COMPARISON")
        print("Using entire master dataset")
        print("="*60)

        all_results = []

        for exec_tf, context_tf in tf_combinations:
            result = self.process_timeframe_pair(exec_tf, context_tf)
            all_results.append(result)
            self.results[result.tf_pair] = result

        # Display summary table
        self.display_results_table(all_results)

        # Find best combination
        best = max(all_results, key=lambda x: x.total_pips)

        print("\n" + "="*60)
        print("WINNER: " + best.tf_pair)
        print("="*60)
        print(f"Total Pips: {best.total_pips:.1f}")
        print(f"Win Rate: {best.win_rate:.1f}%")
        print(f"Profit Factor: {best.profit_factor:.2f}")

        return all_results

    def display_results_table(self, results: List[TFResults]):
        """Display results in a formatted table"""

        print("\n" + "="*60)
        print("RESULTS SUMMARY TABLE")
        print("="*60)

        # Header
        print(f"{'TF Pair':<12} {'Trades':<8} {'Win%':<8} {'Pips':<10} "
              f"{'PF':<6} {'Avg Hold':<10} {'Opps':<8}")
        print("-"*60)

        # Sort by total pips
        results.sort(key=lambda x: x.total_pips, reverse=True)

        for r in results:
            print(f"{r.tf_pair:<12} {r.total_trades:<8} "
                  f"{r.win_rate:<7.1f}% {r.total_pips:<9.1f} "
                  f"{r.profit_factor:<6.2f} {r.avg_bars_held:<9.0f} "
                  f"{r.total_opportunities:<8}")

        print("-"*60)


def main():
    """Run comprehensive timeframe comparison"""

    analyzer = EfficientTFComparison()

    print("Starting comprehensive analysis...")
    print("This may take several minutes due to large dataset...")

    results = analyzer.run_all_comparisons()

    # Save results
    import json
    with open('tf_comparison_results.json', 'w') as f:
        json.dump([{
            'tf_pair': r.tf_pair,
            'total_trades': r.total_trades,
            'win_rate': r.win_rate,
            'total_pips': r.total_pips,
            'profit_factor': r.profit_factor,
            'avg_bars_held': r.avg_bars_held
        } for r in results], f, indent=2)

    print("\nResults saved to tf_comparison_results.json")


if __name__ == "__main__":
    main()