#!/usr/bin/env python3
"""
MTF Trade Simulator - Realistic trading simulation using alignment zones
Tests whether M1 timing with H1 direction can produce profitable trades
"""

import numpy as np
import pandas as pd
import duckdb
from mtf_swing_analyzer import MTFSwingAnalyzer
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MTFTradeSimulator:
    """Simulate realistic trades based on MTF alignment."""

    def __init__(self):
        self.analyzer = MTFSwingAnalyzer()
        self.pip_multiplier = 100  # For JPY pairs (2 decimal places)
        self.spread = 0.01  # 1 pip spread for GBPJPY (0.01)
        self.commission = 0.005  # 0.5 pip commission per side
        self.slippage = 0.005  # 0.5 pip slippage
        self.min_rr = 1.5  # Minimum risk:reward ratio
        self.risk_per_trade = 0.01  # 1% risk per trade

    def find_entry_exit_levels(self, df, entry_idx, direction, swings):
        """
        Find entry, stop loss, and take profit levels based on swings.
        """
        # Get position in dataframe
        try:
            entry_pos = df.index.get_loc(entry_idx)
        except KeyError:
            return None

        entry_price = df.iloc[entry_pos]['close']

        # Find recent swings for stop loss and take profit
        recent_swings = [s for s in swings if s.index <= entry_idx]
        if not recent_swings:
            return None

        # Separate highs and lows
        recent_highs = [s for s in recent_swings if s.type == 'high'][-3:]  # Last 3 highs
        recent_lows = [s for s in recent_swings if s.type == 'low'][-3:]  # Last 3 lows

        if not recent_highs or not recent_lows:
            return None

        if direction == 'long':
            # Stop loss below recent swing low
            stop_loss = min([s.price for s in recent_lows]) - 0.01  # 1 pip buffer

            # Take profit at recent swing high
            take_profit = max([s.price for s in recent_highs])

            # Adjust entry for spread and slippage
            entry_price = entry_price + self.spread + self.slippage

        else:  # short
            # Stop loss above recent swing high
            stop_loss = max([s.price for s in recent_highs]) + 0.01  # 1 pip buffer

            # Take profit at recent swing low
            take_profit = min([s.price for s in recent_lows])

            # Adjust entry for spread and slippage
            entry_price = entry_price - self.slippage

        # Calculate risk reward
        if direction == 'long':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        if risk <= 0 or reward <= 0:
            return None

        rr_ratio = reward / risk

        # Only take trades with good R:R
        if rr_ratio < self.min_rr:
            return None

        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk,
            'reward': reward,
            'rr_ratio': rr_ratio,
            'direction': direction
        }

    def simulate_trade(self, df, trade_setup, entry_idx):
        """
        Simulate trade execution and determine outcome.
        """
        entry_pos = df.index.get_loc(entry_idx)

        # Can't trade if too close to end
        if entry_pos >= len(df) - 10:
            return None

        # Simulate price movement after entry
        max_bars = min(200, len(df) - entry_pos - 1)  # Max 200 M1 bars (3.3 hours)

        for i in range(1, max_bars):
            bar = df.iloc[entry_pos + i]

            if trade_setup['direction'] == 'long':
                # Check if stop hit
                if bar['low'] <= trade_setup['stop_loss']:
                    pips = (trade_setup['stop_loss'] - trade_setup['entry']) * self.pip_multiplier
                    return {
                        'outcome': 'loss',
                        'pips': pips,
                        'exit_price': trade_setup['stop_loss'],
                        'exit_bar': entry_pos + i,
                        'bars_held': i,
                        'rr_achieved': 0
                    }

                # Check if target hit
                if bar['high'] >= trade_setup['take_profit']:
                    pips = (trade_setup['take_profit'] - trade_setup['entry']) * self.pip_multiplier
                    return {
                        'outcome': 'win',
                        'pips': pips,
                        'exit_price': trade_setup['take_profit'],
                        'exit_bar': entry_pos + i,
                        'bars_held': i,
                        'rr_achieved': trade_setup['rr_ratio']
                    }

            else:  # short
                # Check if stop hit
                if bar['high'] >= trade_setup['stop_loss']:
                    pips = (trade_setup['entry'] - trade_setup['stop_loss']) * self.pip_multiplier
                    return {
                        'outcome': 'loss',
                        'pips': pips,
                        'exit_price': trade_setup['stop_loss'],
                        'exit_bar': entry_pos + i,
                        'bars_held': i,
                        'rr_achieved': 0
                    }

                # Check if target hit
                if bar['low'] <= trade_setup['take_profit']:
                    pips = (trade_setup['entry'] - trade_setup['take_profit']) * self.pip_multiplier
                    return {
                        'outcome': 'win',
                        'pips': pips,
                        'exit_price': trade_setup['take_profit'],
                        'exit_bar': entry_pos + i,
                        'bars_held': i,
                        'rr_achieved': trade_setup['rr_ratio']
                    }

        # Trade still open after max bars - close at market
        close_price = df.iloc[entry_pos + max_bars - 1]['close']
        if trade_setup['direction'] == 'long':
            pips = (close_price - trade_setup['entry']) * self.pip_multiplier
        else:
            pips = (trade_setup['entry'] - close_price) * self.pip_multiplier

        return {
            'outcome': 'timeout',
            'pips': pips,
            'exit_price': close_price,
            'exit_bar': entry_pos + max_bars - 1,
            'bars_held': max_bars - 1,
            'rr_achieved': pips / (abs(trade_setup['risk']) * self.pip_multiplier)
        }

    def run_simulation(self):
        """
        Run complete trade simulation on alignment zones.
        """
        print("🚀 MTF TRADE SIMULATION")
        print("="*70)

        # Load and analyze data
        print("Loading data...")
        df_m1 = self.analyzer.load_m1_data(start_bar=10000, limit=3600)
        df_h1 = self.analyzer.aggregate_to_h1(df_m1)

        print("Analyzing timeframes...")
        mtf_data = self.analyzer.analyze_both_timeframes(df_m1, df_h1)

        # Get alignment zones
        zones = self.analyzer.identify_trading_zones(mtf_data)

        if not zones:
            print("No alignment zones found!")
            return None

        print(f"\nFound {len(zones)} alignment zones")

        # Get swings for both timeframes
        m1_swings = mtf_data['m1']['tracker'].confirmed_swings
        h1_swings = mtf_data['h1']['tracker'].confirmed_swings

        # Simulate trades
        trades = []

        for zone in zones:
            # Determine trade direction based on alignment
            if zone['alignment'] in ['STRONG_BULLISH', 'BULLISH_PULLBACK']:
                direction = 'long'
            elif zone['alignment'] in ['STRONG_BEARISH', 'BEARISH_RALLY']:
                direction = 'short'
            else:
                continue

            # Find entry and exit levels using M1 swings
            trade_setup = self.find_entry_exit_levels(
                df_m1, zone['bar_index'], direction, m1_swings
            )

            if not trade_setup:
                continue

            # Simulate trade
            result = self.simulate_trade(df_m1, trade_setup, zone['bar_index'])

            if result:
                trades.append({
                    'entry_bar': zone['bar_index'],
                    'alignment': zone['alignment'],
                    'direction': direction,
                    'entry_price': trade_setup['entry'],
                    'stop_loss': trade_setup['stop_loss'],
                    'take_profit': trade_setup['take_profit'],
                    'planned_rr': trade_setup['rr_ratio'],
                    'risk_pips': abs(trade_setup['risk']) * self.pip_multiplier,
                    **result
                })

        if not trades:
            print("No valid trades found!")
            return None

        # Create DataFrame and calculate statistics
        trades_df = pd.DataFrame(trades)

        # Calculate statistics
        stats = self.calculate_statistics(trades_df)

        # Print results
        self.print_results(trades_df, stats)

        # Create visualization
        self.visualize_trades(df_m1, trades_df, stats)

        # Save results
        trades_df.to_csv('mtf_trade_results.csv', index=False)
        print("\n📁 Results saved to mtf_trade_results.csv")

        return trades_df, stats

    def calculate_statistics(self, trades_df):
        """Calculate comprehensive trading statistics."""

        total_trades = len(trades_df)
        wins = trades_df[trades_df['outcome'] == 'win']
        losses = trades_df[trades_df['outcome'] == 'loss']
        timeouts = trades_df[trades_df['outcome'] == 'timeout']

        stats = {
            'total_trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'timeouts': len(timeouts),
            'win_rate': len(wins) / total_trades * 100 if total_trades > 0 else 0,
            'avg_win_pips': wins['pips'].mean() if len(wins) > 0 else 0,
            'avg_loss_pips': losses['pips'].mean() if len(losses) > 0 else 0,
            'total_pips': trades_df['pips'].sum(),
            'avg_rr_planned': trades_df['planned_rr'].mean(),
            'avg_rr_achieved': trades_df['rr_achieved'].mean(),
            'avg_bars_held': trades_df['bars_held'].mean(),
            'profit_factor': abs(wins['pips'].sum() / losses['pips'].sum()) if len(losses) > 0 and losses['pips'].sum() != 0 else 0
        }

        # Break down by alignment type
        for alignment in trades_df['alignment'].unique():
            alignment_trades = trades_df[trades_df['alignment'] == alignment]
            alignment_wins = alignment_trades[alignment_trades['outcome'] == 'win']

            stats[f'{alignment}_trades'] = len(alignment_trades)
            stats[f'{alignment}_win_rate'] = len(alignment_wins) / len(alignment_trades) * 100 if len(alignment_trades) > 0 else 0
            stats[f'{alignment}_pips'] = alignment_trades['pips'].sum()

        return stats

    def print_results(self, trades_df, stats):
        """Print comprehensive results."""

        print("\n" + "="*70)
        print("📊 TRADING RESULTS")
        print("="*70)

        print(f"\n📈 Overall Performance:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"  Losses: {stats['losses']}")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Total Pips: {stats['total_pips']:.1f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")

        print(f"\n💰 Risk/Reward:")
        print(f"  Avg Planned R:R: {stats['avg_rr_planned']:.2f}")
        print(f"  Avg Achieved R:R: {stats['avg_rr_achieved']:.2f}")
        print(f"  Avg Win: {stats['avg_win_pips']:.1f} pips")
        print(f"  Avg Loss: {stats['avg_loss_pips']:.1f} pips")
        print(f"  Avg Hold Time: {stats['avg_bars_held']:.0f} bars ({stats['avg_bars_held']:.0f} minutes)")

        print(f"\n🎯 By Alignment Type:")
        for alignment in trades_df['alignment'].unique():
            if f'{alignment}_trades' in stats:
                print(f"  {alignment}:")
                print(f"    Trades: {stats[f'{alignment}_trades']}")
                print(f"    Win Rate: {stats[f'{alignment}_win_rate']:.1f}%")
                print(f"    Total Pips: {stats[f'{alignment}_pips']:.1f}")

        # Sample trades
        print(f"\n📝 Sample Trades (First 5):")
        print(trades_df[['alignment', 'direction', 'planned_rr', 'outcome', 'pips', 'bars_held']].head())

    def visualize_trades(self, df_m1, trades_df, stats, save_path='mtf_trade_results.png'):
        """Create comprehensive visualization of trading results."""

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # 1. Win/Loss Distribution
        ax = axes[0, 0]
        outcomes = trades_df['outcome'].value_counts()
        colors = ['green' if x == 'win' else 'red' if x == 'loss' else 'gray' for x in outcomes.index]
        ax.bar(outcomes.index, outcomes.values, color=colors, alpha=0.7)
        ax.set_title('Trade Outcomes')
        ax.set_ylabel('Count')

        # 2. Pips Distribution
        ax = axes[0, 1]
        ax.hist(trades_df['pips'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'Pips Distribution (Total: {stats["total_pips"]:.1f})')
        ax.set_xlabel('Pips')
        ax.set_ylabel('Frequency')

        # 3. R:R Comparison
        ax = axes[1, 0]
        x = range(len(trades_df))
        ax.scatter(x, trades_df['planned_rr'], alpha=0.5, label='Planned R:R', color='blue')
        ax.scatter(x, trades_df['rr_achieved'], alpha=0.5, label='Achieved R:R', color='green')
        ax.axhline(self.min_rr, color='red', linestyle='--', alpha=0.5, label=f'Min R:R ({self.min_rr})')
        ax.set_title('Risk:Reward Analysis')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('R:R Ratio')
        ax.legend()

        # 4. Cumulative Pips
        ax = axes[1, 1]
        cumulative_pips = trades_df['pips'].cumsum()
        ax.plot(cumulative_pips, linewidth=2)
        ax.fill_between(range(len(cumulative_pips)), 0, cumulative_pips,
                        where=(cumulative_pips >= 0), color='green', alpha=0.3, label='Profit')
        ax.fill_between(range(len(cumulative_pips)), 0, cumulative_pips,
                        where=(cumulative_pips < 0), color='red', alpha=0.3, label='Loss')
        ax.set_title('Cumulative Performance')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative Pips')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 5. Performance by Alignment
        ax = axes[2, 0]
        alignment_stats = []
        for alignment in trades_df['alignment'].unique():
            alignment_trades = trades_df[trades_df['alignment'] == alignment]
            alignment_stats.append({
                'alignment': alignment,
                'win_rate': (alignment_trades['outcome'] == 'win').mean() * 100,
                'total_pips': alignment_trades['pips'].sum()
            })

        if alignment_stats:
            alignment_df = pd.DataFrame(alignment_stats)
            x_pos = range(len(alignment_df))
            ax.bar(x_pos, alignment_df['win_rate'], alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(alignment_df['alignment'], rotation=45, ha='right')
            ax.set_title('Win Rate by Alignment Type')
            ax.set_ylabel('Win Rate %')

        # 6. Hold Time Distribution
        ax = axes[2, 1]
        ax.hist(trades_df['bars_held'], bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.set_title('Trade Duration (M1 Bars)')
        ax.set_xlabel('Bars Held')
        ax.set_ylabel('Frequency')
        ax.axvline(trades_df['bars_held'].mean(), color='red', linestyle='--',
                  label=f'Avg: {trades_df["bars_held"].mean():.0f} bars')
        ax.legend()

        plt.suptitle(f'MTF Trade Simulation Results\n'
                    f'Total: {stats["total_trades"]} trades | '
                    f'Win Rate: {stats["win_rate"]:.1f}% | '
                    f'Total Pips: {stats["total_pips"]:.1f}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\n📊 Results chart saved to {save_path}")

        return fig


def main():
    """Run the trade simulation."""
    simulator = MTFTradeSimulator()
    trades_df, stats = simulator.run_simulation()

    if trades_df is not None:
        print("\n" + "="*70)
        print("🎯 KEY FINDINGS")
        print("="*70)

        # Determine if M1 is viable
        if stats['win_rate'] > 40 and stats['profit_factor'] > 1.2:
            print("✅ M1 timeframe appears VIABLE for this strategy!")
            print("   - Sufficient win rate and profit factor")
            print("   - Alignment zones provide edge")
        else:
            print("⚠️ M1 timeframe may be TOO NOISY for this strategy")
            print("   - Consider using M5 or M15 for entry timing")
            print("   - Or tighter stop losses with trailing stops")

        if stats['avg_bars_held'] < 30:
            print("✅ Quick trades (< 30 minutes average)")
            print("   - Good for scalping approach")
        else:
            print("📊 Longer hold times suggest position trading")
            print("   - Consider higher timeframe entries")

    return trades_df, stats


if __name__ == "__main__":
    trades_df, stats = main()