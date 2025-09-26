#!/usr/bin/env python3
"""
M5/H1 Trade Simulator - Simulates trades using M5 timing with H1 trend alignment.
Uses swing points for stop loss and take profit levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import duckdb
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from m5h1_swing_analyzer import M5H1SwingAnalyzer
from swing_state_tracker import SwingState


@dataclass
class Trade:
    """Represents a single trade"""
    entry_bar: int
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: str  # 'long' or 'short'
    alignment: str  # Type of alignment when trade was entered
    exit_bar: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # 'win', 'loss', 'open'
    pips: Optional[float] = None
    bars_held: Optional[int] = None


class M5H1TradeSimulator:
    """Simulates trades using M5/H1 swing alignment"""

    def __init__(self):
        self.analyzer = M5H1SwingAnalyzer()
        self.trades: List[Trade] = []
        self.pip_multiplier = 100  # For JPY pairs (2 decimal places)

    def run_simulation(self, start_idx: int = 10000, end_idx: int = 14000):
        """Run complete trade simulation"""
        # Load and analyze data
        print("Loading data and analyzing swings...")
        self.analyzer.load_data(start_idx, end_idx)
        self.analyzer.analyze_swings()

        # Find trade opportunities
        print("\nFinding trade opportunities...")
        self.find_trades()

        # Simulate trade execution
        print("Simulating trade execution...")
        self.execute_trades()

        # Calculate and display results
        print("\nGenerating results...")
        self.calculate_results()

    def find_trades(self):
        """Find trade opportunities based on M5/H1 alignment"""
        m5_data = self.analyzer.m5_data
        h1_data = self.analyzer.h1_data
        m5_swings = self.analyzer.m5_swings
        h1_states = self.analyzer.h1_states
        m5_states = self.analyzer.m5_states

        # Track open positions to avoid overlapping trades
        open_trade = None

        for i in range(20, len(m5_data) - 1):  # Start after some bars for swing context
            # Skip if we have an open trade
            if open_trade is not None:
                # Check if open trade should be closed
                bar = m5_data.iloc[i]
                if open_trade.direction == 'long':
                    if bar['low'] <= open_trade.stop_loss or bar['high'] >= open_trade.take_profit:
                        open_trade = None
                        continue
                else:  # short
                    if bar['high'] >= open_trade.stop_loss or bar['low'] <= open_trade.take_profit:
                        open_trade = None
                        continue
                continue

            # Get current states
            m5_state = m5_states[i] if i < len(m5_states) else None

            # Find corresponding H1 state
            m5_timestamp = m5_data.index[i]
            h1_idx = h1_data.index.get_indexer([m5_timestamp], method='ffill')[0]
            h1_state = h1_states[h1_idx] if h1_idx >= 0 and h1_idx < len(h1_states) else None

            if m5_state is None or h1_state is None:
                continue

            # Check alignment
            alignment = self.analyzer.check_alignment(m5_state, h1_state)

            # Only trade on strong alignments or pullback/rally opportunities
            if alignment not in ['STRONG_BULLISH', 'BULLISH_PULLBACK', 'STRONG_BEARISH', 'BEARISH_RALLY']:
                continue

            # Find recent M5 swings for stop/target levels
            recent_swings = [s for s in m5_swings if s.index <= m5_data.iloc[i]['bar_index']]
            if len(recent_swings) < 2:
                continue

            # Determine trade direction and levels
            if alignment in ['STRONG_BULLISH', 'BULLISH_PULLBACK']:
                # Long trade
                direction = 'long'
                entry_price = m5_data.iloc[i]['close']

                # Find most recent swing low for stop loss
                recent_lows = [s for s in recent_swings if s.type == 'low']
                if not recent_lows:
                    continue
                stop_loss = recent_lows[-1].price - 0.05  # Small buffer

                # Find previous swing high for take profit
                recent_highs = [s for s in recent_swings if s.type == 'high']
                if not recent_highs:
                    continue

                # Use 1.5-2x risk for reward
                risk = entry_price - stop_loss
                if risk <= 0:
                    continue
                take_profit = entry_price + (risk * 2.0)

            elif alignment in ['STRONG_BEARISH', 'BEARISH_RALLY']:
                # Short trade
                direction = 'short'
                entry_price = m5_data.iloc[i]['close']

                # Find most recent swing high for stop loss
                recent_highs = [s for s in recent_swings if s.type == 'high']
                if not recent_highs:
                    continue
                stop_loss = recent_highs[-1].price + 0.05  # Small buffer

                # Find previous swing low for take profit
                recent_lows = [s for s in recent_swings if s.type == 'low']
                if not recent_lows:
                    continue

                # Use 1.5-2x risk for reward
                risk = stop_loss - entry_price
                if risk <= 0:
                    continue
                take_profit = entry_price - (risk * 2.0)
            else:
                continue

            # Create trade
            trade = Trade(
                entry_bar=m5_data.iloc[i]['bar_index'],
                entry_time=m5_data.index[i],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                direction=direction,
                alignment=alignment
            )

            self.trades.append(trade)
            open_trade = trade

    def execute_trades(self):
        """Simulate trade execution with M5 bar data"""
        m5_data = self.analyzer.m5_data

        for trade in self.trades:
            # Find entry bar position in data
            entry_positions = m5_data[m5_data['bar_index'] == trade.entry_bar].index
            if len(entry_positions) == 0:
                continue

            entry_pos = m5_data.index.get_loc(entry_positions[0])

            # Simulate trade from next bar
            for j in range(entry_pos + 1, len(m5_data)):
                bar = m5_data.iloc[j]

                if trade.direction == 'long':
                    # Check stop loss
                    if bar['low'] <= trade.stop_loss:
                        trade.exit_bar = bar['bar_index']
                        trade.exit_time = m5_data.index[j]
                        trade.exit_price = trade.stop_loss
                        trade.outcome = 'loss'
                        trade.pips = (trade.stop_loss - trade.entry_price) * self.pip_multiplier
                        trade.bars_held = j - entry_pos
                        break

                    # Check take profit
                    if bar['high'] >= trade.take_profit:
                        trade.exit_bar = bar['bar_index']
                        trade.exit_time = m5_data.index[j]
                        trade.exit_price = trade.take_profit
                        trade.outcome = 'win'
                        trade.pips = (trade.take_profit - trade.entry_price) * self.pip_multiplier
                        trade.bars_held = j - entry_pos
                        break

                else:  # short
                    # Check stop loss
                    if bar['high'] >= trade.stop_loss:
                        trade.exit_bar = bar['bar_index']
                        trade.exit_time = m5_data.index[j]
                        trade.exit_price = trade.stop_loss
                        trade.outcome = 'loss'
                        trade.pips = (trade.entry_price - trade.stop_loss) * self.pip_multiplier
                        trade.bars_held = j - entry_pos
                        break

                    # Check take profit
                    if bar['low'] <= trade.take_profit:
                        trade.exit_bar = bar['bar_index']
                        trade.exit_time = m5_data.index[j]
                        trade.exit_price = trade.take_profit
                        trade.outcome = 'win'
                        trade.pips = (trade.entry_price - trade.take_profit) * self.pip_multiplier
                        trade.bars_held = j - entry_pos
                        break

            # Mark as open if not closed
            if trade.outcome is None:
                trade.outcome = 'open'

    def calculate_results(self):
        """Calculate and display trading results"""
        if not self.trades:
            print("No trades found!")
            return

        # Filter completed trades
        completed = [t for t in self.trades if t.outcome in ['win', 'loss']]

        if not completed:
            print("No completed trades!")
            return

        # Calculate statistics
        wins = [t for t in completed if t.outcome == 'win']
        losses = [t for t in completed if t.outcome == 'loss']

        win_rate = len(wins) / len(completed) * 100
        total_pips = sum(t.pips for t in completed)
        avg_win = np.mean([t.pips for t in wins]) if wins else 0
        avg_loss = np.mean([t.pips for t in losses]) if losses else 0
        profit_factor = abs(sum(t.pips for t in wins) / sum(t.pips for t in losses)) if losses else 0

        # Average hold time (in M5 bars)
        avg_hold = np.mean([t.bars_held for t in completed if t.bars_held])
        avg_hold_minutes = avg_hold * 5  # Convert to minutes

        print("\n" + "="*60)
        print("M5/H1 TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Total Trades: {len(completed)}")
        print(f"Wins: {len(wins)} ({win_rate:.1f}%)")
        print(f"Losses: {len(losses)} ({100-win_rate:.1f}%)")
        print(f"\nTotal Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"\nAverage Hold Time: {avg_hold:.1f} bars ({avg_hold_minutes:.0f} minutes)")

        # Breakdown by alignment
        print("\n--- Results by Alignment Type ---")
        for alignment in set(t.alignment for t in completed):
            alignment_trades = [t for t in completed if t.alignment == alignment]
            alignment_wins = [t for t in alignment_trades if t.outcome == 'win']
            alignment_pips = sum(t.pips for t in alignment_trades)
            print(f"{alignment}: {len(alignment_trades)} trades, "
                  f"{len(alignment_wins)}/{len(alignment_trades)} wins, "
                  f"{alignment_pips:.1f} pips")

        # Save results to CSV
        self.save_results()

        # Create visualization
        self.plot_results()

    def save_results(self):
        """Save trade results to CSV"""
        if not self.trades:
            return

        # Convert to DataFrame
        data = []
        for t in self.trades:
            if t.outcome in ['win', 'loss']:
                data.append({
                    'entry_bar': t.entry_bar,
                    'alignment': t.alignment,
                    'direction': t.direction,
                    'entry_price': t.entry_price,
                    'stop_loss': t.stop_loss,
                    'take_profit': t.take_profit,
                    'outcome': t.outcome,
                    'pips': round(t.pips, 1),
                    'exit_price': t.exit_price,
                    'exit_bar': t.exit_bar,
                    'bars_held': t.bars_held,
                    'minutes_held': t.bars_held * 5
                })

        df = pd.DataFrame(data)
        df.to_csv('m5h1_trade_results.csv', index=False)
        print(f"\nTrade results saved to m5h1_trade_results.csv")

    def plot_results(self):
        """Create visualization of trading results"""
        if not self.trades:
            return

        completed = [t for t in self.trades if t.outcome in ['win', 'loss']]
        if not completed:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Cumulative P&L
        cumulative_pips = np.cumsum([t.pips for t in completed])
        ax1.plot(cumulative_pips, 'b-', linewidth=2)
        ax1.fill_between(range(len(cumulative_pips)), 0, cumulative_pips,
                         where=(cumulative_pips >= 0), color='green', alpha=0.3)
        ax1.fill_between(range(len(cumulative_pips)), 0, cumulative_pips,
                         where=(cumulative_pips < 0), color='red', alpha=0.3)
        ax1.set_title('Cumulative P&L (Pips)')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Pips')
        ax1.grid(True, alpha=0.3)

        # 2. Win/Loss Distribution
        wins = [t.pips for t in completed if t.outcome == 'win']
        losses = [abs(t.pips) for t in completed if t.outcome == 'loss']
        ax2.hist([wins, losses], label=['Wins', 'Losses'], color=['green', 'red'], alpha=0.7, bins=20)
        ax2.set_title('Win/Loss Distribution')
        ax2.set_xlabel('Pips')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Trade Duration Distribution (in minutes)
        durations = [t.bars_held * 5 for t in completed if t.bars_held]  # Convert to minutes
        ax3.hist(durations, bins=30, color='blue', alpha=0.7)
        ax3.set_title('Trade Duration Distribution')
        ax3.set_xlabel('Duration (minutes)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Results by Alignment Type
        alignments = list(set(t.alignment for t in completed))
        alignment_pips = [sum(t.pips for t in completed if t.alignment == a) for a in alignments]
        colors = ['green' if p > 0 else 'red' for p in alignment_pips]
        ax4.bar(alignments, alignment_pips, color=colors, alpha=0.7)
        ax4.set_title('Performance by Alignment Type')
        ax4.set_xlabel('Alignment')
        ax4.set_ylabel('Total Pips')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('M5/H1 Trade Simulation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('m5h1_trade_results.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Results chart saved to m5h1_trade_results.png")


def main():
    """Run M5/H1 trade simulation"""
    simulator = M5H1TradeSimulator()
    simulator.run_simulation(start_idx=10000, end_idx=14000)


if __name__ == "__main__":
    main()