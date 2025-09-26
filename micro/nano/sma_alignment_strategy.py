#!/usr/bin/env python3
"""
SMA Alignment Strategy Tester
Tests a simple strategy: price > SMA50 > SMA200 for trend
Entry on pullback realignment, exit on SMA200 cross
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SMASignal:
    """Represents a trading signal"""
    bar_index: int
    timestamp: datetime
    signal_type: str  # 'long_entry', 'short_entry', 'exit'
    price: float
    sma50: float
    sma200: float
    alignment: str  # 'bullish', 'bearish', 'neutral'


@dataclass
class SMATrade:
    """Represents a completed trade"""
    entry_bar: int
    entry_time: datetime
    entry_price: float
    exit_bar: int = None
    exit_time: datetime = None
    exit_price: float = None
    direction: str = 'long'
    pips: float = 0
    bars_held: int = 0
    entry_reason: str = ''
    exit_reason: str = ''


class SMAAlignmentStrategy:
    """Test SMA alignment strategy"""

    def __init__(self, db_path: str = "../../data/master.duckdb"):
        self.db_path = db_path
        self.data = None
        self.signals: List[SMASignal] = []
        self.trades: List[SMATrade] = []
        self.pip_multiplier = 100  # JPY pairs

    def load_data(self, start_idx: int = 10000, end_idx: int = 14000):
        """Load price data"""
        conn = duckdb.connect(self.db_path, read_only=True)

        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {start_idx} AND {end_idx}
        ORDER BY bar_index
        """

        self.data = pd.read_sql(query, conn)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        conn.close()

        print(f"Loaded {len(self.data)} bars")

    def compute_smas(self):
        """Calculate SMAs and alignment"""
        # Calculate SMAs
        self.data['sma50'] = self.data['close'].rolling(window=50).mean()
        self.data['sma200'] = self.data['close'].rolling(window=200).mean()

        # Determine alignment for each bar
        self.data['alignment'] = 'neutral'

        for i in range(len(self.data)):
            if pd.isna(self.data.iloc[i]['sma50']) or pd.isna(self.data.iloc[i]['sma200']):
                continue

            close = self.data.iloc[i]['close']
            sma50 = self.data.iloc[i]['sma50']
            sma200 = self.data.iloc[i]['sma200']

            # Bullish: price > sma50 > sma200
            if close > sma50 and sma50 > sma200:
                self.data.loc[i, 'alignment'] = 'bullish'
            # Bearish: price < sma50 < sma200
            elif close < sma50 and sma50 < sma200:
                self.data.loc[i, 'alignment'] = 'bearish'
            # Pullback zones
            elif close < sma50 and sma50 > sma200:
                self.data.loc[i, 'alignment'] = 'bullish_pullback'
            elif close > sma50 and sma50 < sma200:
                self.data.loc[i, 'alignment'] = 'bearish_rally'

        print(f"Calculated SMAs and alignment")

    def find_signals(self):
        """Find entry and exit signals based on alignment changes"""
        self.signals = []
        in_position = None  # None, 'long', or 'short'

        for i in range(201, len(self.data)):  # Start after SMA200 is available
            curr = self.data.iloc[i]
            prev = self.data.iloc[i-1]

            close = curr['close']
            sma50 = curr['sma50']
            sma200 = curr['sma200']

            # Skip if SMAs not ready
            if pd.isna(sma50) or pd.isna(sma200):
                continue

            # LONG ENTRY: Was in pullback, now realigned bullish
            if (prev['alignment'] == 'bullish_pullback' and
                curr['alignment'] == 'bullish' and
                in_position != 'long'):

                signal = SMASignal(
                    bar_index=curr['bar_index'],
                    timestamp=curr['timestamp'],
                    signal_type='long_entry',
                    price=close,
                    sma50=sma50,
                    sma200=sma200,
                    alignment=curr['alignment']
                )
                self.signals.append(signal)
                in_position = 'long'

            # SHORT ENTRY: Was in rally, now realigned bearish
            elif (prev['alignment'] == 'bearish_rally' and
                  curr['alignment'] == 'bearish' and
                  in_position != 'short'):

                signal = SMASignal(
                    bar_index=curr['bar_index'],
                    timestamp=curr['timestamp'],
                    signal_type='short_entry',
                    price=close,
                    sma50=sma50,
                    sma200=sma200,
                    alignment=curr['alignment']
                )
                self.signals.append(signal)
                in_position = 'short'

            # EXIT CONDITIONS
            elif in_position is not None:
                exit_signal = False
                exit_reason = ''

                if in_position == 'long':
                    # Exit long if price crosses below SMA200
                    if close < sma200:
                        exit_signal = True
                        exit_reason = 'price < sma200'
                    # Or if alignment turns bearish
                    elif curr['alignment'] == 'bearish':
                        exit_signal = True
                        exit_reason = 'bearish alignment'

                elif in_position == 'short':
                    # Exit short if price crosses above SMA200
                    if close > sma200:
                        exit_signal = True
                        exit_reason = 'price > sma200'
                    # Or if alignment turns bullish
                    elif curr['alignment'] == 'bullish':
                        exit_signal = True
                        exit_reason = 'bullish alignment'

                if exit_signal:
                    signal = SMASignal(
                        bar_index=curr['bar_index'],
                        timestamp=curr['timestamp'],
                        signal_type='exit',
                        price=close,
                        sma50=sma50,
                        sma200=sma200,
                        alignment=curr['alignment']
                    )
                    self.signals.append(signal)
                    in_position = None

        print(f"Found {len(self.signals)} signals")

    def simulate_trades(self):
        """Convert signals to trades with P&L"""
        self.trades = []
        current_trade = None

        for signal in self.signals:
            if signal.signal_type == 'long_entry':
                if current_trade is None:  # Only if not in position
                    current_trade = SMATrade(
                        entry_bar=signal.bar_index,
                        entry_time=signal.timestamp,
                        entry_price=signal.price,
                        direction='long',
                        entry_reason='pullback_realignment'
                    )

            elif signal.signal_type == 'short_entry':
                if current_trade is None:  # Only if not in position
                    current_trade = SMATrade(
                        entry_bar=signal.bar_index,
                        entry_time=signal.timestamp,
                        entry_price=signal.price,
                        direction='short',
                        entry_reason='rally_realignment'
                    )

            elif signal.signal_type == 'exit' and current_trade is not None:
                # Close current trade
                current_trade.exit_bar = signal.bar_index
                current_trade.exit_time = signal.timestamp
                current_trade.exit_price = signal.price
                current_trade.exit_reason = signal.alignment

                # Calculate P&L
                if current_trade.direction == 'long':
                    current_trade.pips = (current_trade.exit_price - current_trade.entry_price) * self.pip_multiplier
                else:  # short
                    current_trade.pips = (current_trade.entry_price - current_trade.exit_price) * self.pip_multiplier

                # Calculate bars held
                entry_idx = self.data[self.data['bar_index'] == current_trade.entry_bar].index[0]
                exit_idx = self.data[self.data['bar_index'] == current_trade.exit_bar].index[0]
                current_trade.bars_held = exit_idx - entry_idx

                self.trades.append(current_trade)
                current_trade = None

        print(f"Completed {len(self.trades)} trades")

    def analyze_results(self):
        """Analyze trading results"""
        if not self.trades:
            print("No completed trades to analyze")
            return

        # Calculate statistics
        total_pips = sum(t.pips for t in self.trades)
        winning_trades = [t for t in self.trades if t.pips > 0]
        losing_trades = [t for t in self.trades if t.pips <= 0]

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pips for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pips for t in losing_trades]) if losing_trades else 0

        # Profit factor
        total_wins = sum(t.pips for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pips for t in losing_trades)) if losing_trades else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Average hold time
        avg_bars = np.mean([t.bars_held for t in self.trades])

        print("\n" + "="*60)
        print("SMA ALIGNMENT STRATEGY RESULTS")
        print("="*60)
        print(f"Strategy: Price > SMA50 > SMA200 (Bullish)")
        print(f"Entry: Pullback to SMA50 then realignment")
        print(f"Exit: Cross of SMA200 or opposite alignment")
        print("-"*60)
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winners: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losers: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"\nTotal Pips: {total_pips:.1f}")
        print(f"Average Win: {avg_win:.1f} pips")
        print(f"Average Loss: {avg_loss:.1f} pips")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Hold: {avg_bars:.0f} bars ({avg_bars:.0f} minutes)")

        # Breakdown by direction
        long_trades = [t for t in self.trades if t.direction == 'long']
        short_trades = [t for t in self.trades if t.direction == 'short']

        print(f"\nLong Trades: {len(long_trades)} trades, {sum(t.pips for t in long_trades):.1f} pips")
        print(f"Short Trades: {len(short_trades)} trades, {sum(t.pips for t in short_trades):.1f} pips")

        # Save to CSV
        self.save_results()

        # Create visualization
        self.plot_strategy()

    def save_results(self):
        """Save trade results to CSV"""
        if not self.trades:
            return

        data = []
        for t in self.trades:
            data.append({
                'entry_bar': t.entry_bar,
                'entry_time': t.entry_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pips': round(t.pips, 1),
                'bars_held': t.bars_held,
                'exit_reason': t.exit_reason
            })

        df = pd.DataFrame(data)
        df.to_csv('sma_trades.csv', index=False)
        print(f"\nTrade details saved to sma_trades.csv")

    def plot_strategy(self):
        """Visualize the strategy"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])

        # Sample data for visibility
        plot_start = 500
        plot_end = 1500
        plot_data = self.data.iloc[plot_start:plot_end]

        # 1. Price chart with SMAs
        ax = axes[0]
        ax.plot(plot_data.index, plot_data['close'], 'k-', linewidth=0.5, label='Price')
        ax.plot(plot_data.index, plot_data['sma50'], 'b-', linewidth=1, label='SMA50', alpha=0.7)
        ax.plot(plot_data.index, plot_data['sma200'], 'r-', linewidth=1, label='SMA200', alpha=0.7)

        # Mark trades
        for trade in self.trades:
            if trade.entry_bar in plot_data['bar_index'].values:
                entry_idx = plot_data[plot_data['bar_index'] == trade.entry_bar].index[0]
                marker = '^' if trade.direction == 'long' else 'v'
                color = 'g' if trade.direction == 'long' else 'r'
                ax.scatter(entry_idx, trade.entry_price, marker=marker, color=color, s=100, zorder=5)

            if trade.exit_bar and trade.exit_bar in plot_data['bar_index'].values:
                exit_idx = plot_data[plot_data['bar_index'] == trade.exit_bar].index[0]
                ax.scatter(exit_idx, trade.exit_price, marker='x', color='black', s=100, zorder=5)

        ax.set_title('SMA Alignment Strategy - Price Action')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. Alignment indicator
        ax = axes[1]
        alignment_map = {'bullish': 2, 'bullish_pullback': 1, 'neutral': 0,
                        'bearish_rally': -1, 'bearish': -2}
        alignment_values = [alignment_map.get(a, 0) for a in plot_data['alignment']]

        ax.fill_between(plot_data.index, 0, alignment_values,
                        where=[v > 0 for v in alignment_values], color='green', alpha=0.3, label='Bullish')
        ax.fill_between(plot_data.index, 0, alignment_values,
                        where=[v < 0 for v in alignment_values], color='red', alpha=0.3, label='Bearish')
        ax.plot(plot_data.index, alignment_values, 'k-', linewidth=0.5)
        ax.set_title('Market Alignment')
        ax.set_ylabel('Alignment')
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['Bearish', 'Rally', 'Neutral', 'Pullback', 'Bullish'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cumulative P&L
        ax = axes[2]
        if self.trades:
            trade_pnl = []
            cumulative = 0
            for i in range(len(plot_data)):
                bar_idx = plot_data.iloc[i]['bar_index']
                # Check if any trade exits at this bar
                for trade in self.trades:
                    if trade.exit_bar == bar_idx:
                        cumulative += trade.pips
                trade_pnl.append(cumulative)

            ax.plot(plot_data.index, trade_pnl, 'b-', linewidth=2)
            ax.fill_between(plot_data.index, 0, trade_pnl,
                          where=[p > 0 for p in trade_pnl], color='green', alpha=0.3)
            ax.fill_between(plot_data.index, 0, trade_pnl,
                          where=[p < 0 for p in trade_pnl], color='red', alpha=0.3)

        ax.set_title('Cumulative P&L (pips)')
        ax.set_xlabel('Bar Index')
        ax.set_ylabel('Pips')
        ax.grid(True, alpha=0.3)

        plt.suptitle('SMA Alignment Strategy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sma_strategy.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Strategy chart saved to sma_strategy.png")

    def compare_with_swing_approach(self):
        """Compare results with swing-based approach"""
        print("\n" + "="*60)
        print("COMPARISON WITH SWING-BASED APPROACH")
        print("="*60)

        print("\nSMA Approach Advantages:")
        print("✓ Simpler to implement and understand")
        print("✓ No complex state tracking required")
        print("✓ Clear visual signals on chart")
        print("✓ Works well for trend following")

        print("\nSMA Approach Disadvantages:")
        print("✗ Lagging indicator (late entries/exits)")
        print("✗ Whipsaws in ranging markets")
        print("✗ Fixed parameters (50/200) may not be optimal")
        print("✗ No market structure context")

        print("\nBest Use Cases:")
        print("• Strong trending markets")
        print("• Longer timeframes (H1, H4, Daily)")
        print("• Position/swing trading")
        print("• Risk management overlay")


def main():
    """Run SMA alignment strategy test"""
    strategy = SMAAlignmentStrategy()

    # Load data
    print("Testing SMA Alignment Strategy...")
    print("-"*40)
    strategy.load_data(start_idx=10000, end_idx=14000)

    # Calculate indicators
    strategy.compute_smas()

    # Find signals
    strategy.find_signals()

    # Simulate trades
    strategy.simulate_trades()

    # Analyze results
    strategy.analyze_results()

    # Compare approaches
    strategy.compare_with_swing_approach()


if __name__ == "__main__":
    main()