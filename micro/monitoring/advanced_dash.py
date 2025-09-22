#!/usr/bin/env python3
"""
Advanced terminal dashboard with comprehensive trade statistics.
"""

import subprocess
import time
import os
import re
from datetime import datetime
from collections import deque

class AdvancedDashboard:
    def __init__(self, container_name="micro_training"):
        self.container_name = container_name
        self.episode_history = deque(maxlen=100)
        self.last_episode = 0

    def get_logs(self, lines=3000):
        """Get recent container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name, "--tail", str(lines)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5
            )
            return result.stdout
        except Exception as e:
            print(f"Error getting logs: {e}")
            return ""

    def parse_metrics(self, logs):
        """Parse all metrics from logs."""
        metrics = {}

        # Find latest episode line - handle both INFO and plain format
        episode_pattern = r'Episode (\d+) \| Steps (\d+) \| EPS: ([\d.]+) \| Loss: ([\d.]+) \| TD: ([\d.]+) \| Exp: ([-\d.]+) \| WR: ([\d.]+)% \| TradeRatio: ([\d.]+)%'
        episode_matches = re.findall(episode_pattern, logs)

        if episode_matches:
            latest = episode_matches[-1]
            metrics['episode'] = int(latest[0])
            metrics['steps'] = int(latest[1])
            metrics['eps'] = float(latest[2])
            metrics['loss'] = float(latest[3])
            metrics['expectancy'] = float(latest[5])
            metrics['win_rate'] = float(latest[6])
            metrics['trade_ratio'] = float(latest[7])

            # Track episode progression
            if metrics['episode'] != self.last_episode:
                self.episode_history.append({
                    'episode': metrics['episode'],
                    'expectancy': metrics['expectancy'],
                    'win_rate': metrics['win_rate'],
                    'time': datetime.now()
                })
                self.last_episode = metrics['episode']

        # Find validation trade counts
        validation_pattern = r'Validation.*Expectancy: ([-\d.]+).*Win Rate: ([\d.]+)%.*Trades: (\d+)'
        val_matches = re.findall(validation_pattern, logs)
        if val_matches:
            latest_val = val_matches[-1]
            metrics['val_trades'] = int(latest_val[2])
            metrics['val_expectancy'] = float(latest_val[0])
            metrics['val_win_rate'] = float(latest_val[1])

        # Find action distribution
        action_pattern = r'Action distribution - HOLD: ([\d.]+)%, BUY: ([\d.]+)%, SELL: ([\d.]+)%, CLOSE: ([\d.]+)%'
        action_matches = re.findall(action_pattern, logs)
        if action_matches:
            latest_action = action_matches[-1]
            metrics['actions'] = {
                'HOLD': float(latest_action[0]),
                'BUY': float(latest_action[1]),
                'SELL': float(latest_action[2]),
                'CLOSE': float(latest_action[3])
            }

        # Find collection stats
        collection_pattern = r'Collected (\d+) episodes, (\d+) experiences'
        collection_matches = re.findall(collection_pattern, logs)
        if collection_matches:
            latest_col = collection_matches[-1]
            metrics['episodes_collected'] = int(latest_col[0])
            metrics['experiences_collected'] = int(latest_col[1])

        return metrics

    def calculate_trade_stats(self, metrics):
        """Calculate detailed trade statistics."""
        stats = {}

        if 'experiences_collected' in metrics and 'episodes_collected' in metrics:
            if metrics['episodes_collected'] > 0:
                exp_per_ep = metrics['experiences_collected'] / metrics['episodes_collected']
                stats['exp_per_episode'] = exp_per_ep

                # Each episode is 360 bars
                # Trade ratio tells us what % of experiences are trades
                if 'trade_ratio' in metrics:
                    # Estimate trades per episode
                    trades_per_ep = exp_per_ep * (metrics['trade_ratio'] / 100)
                    stats['trades_per_episode'] = trades_per_ep

                    # Estimate average trade duration
                    if trades_per_ep > 0:
                        avg_duration = 360 / trades_per_ep
                        stats['avg_trade_duration'] = avg_duration

        # Calculate improvement trend
        if len(self.episode_history) > 10:
            old_exp = sum(h['expectancy'] for h in list(self.episode_history)[:10]) / 10
            new_exp = sum(h['expectancy'] for h in list(self.episode_history)[-10:]) / 10
            stats['expectancy_trend'] = new_exp - old_exp

            old_wr = sum(h['win_rate'] for h in list(self.episode_history)[:10]) / 10
            new_wr = sum(h['win_rate'] for h in list(self.episode_history)[-10:]) / 10
            stats['winrate_trend'] = new_wr - old_wr

        return stats

    def display(self, metrics, stats):
        """Display formatted dashboard."""
        os.system('clear')

        print("=" * 80)
        print(" " * 25 + "🚀 ADVANCED MUZERO DASHBOARD 🚀")
        print("=" * 80)
        print()

        if not metrics:
            print("Waiting for training data...")
            return

        # Progress section
        if 'episode' in metrics:
            progress = (metrics['episode'] / 1000000) * 100
            print("📊 TRAINING PROGRESS")
            print("─" * 40)
            print(f"  Episode:      {metrics['episode']:,} / 1,000,000 ({progress:.3f}%)")
            print(f"  Total Steps:  {metrics.get('steps', 0):,}")
            print(f"  Speed:        {metrics.get('eps', 0):.2f} episodes/sec")

            # ETA calculation
            if metrics.get('eps', 0) > 0:
                remaining = 1000000 - metrics['episode']
                eta_hours = remaining / (metrics['eps'] * 3600)
                print(f"  ETA:          {eta_hours/24:.1f} days ({eta_hours:.0f} hours)")

            # Progress bar
            bar_len = 50
            filled = int(bar_len * progress / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {progress:.2f}%")
            print()

        # Performance section
        print("💰 TRADING PERFORMANCE")
        print("─" * 40)
        print(f"  Expectancy:   {metrics.get('expectancy', 0):.2f} pips/trade")
        print(f"  Win Rate:     {metrics.get('win_rate', 0):.1f}%")
        print(f"  Trade Ratio:  {metrics.get('trade_ratio', 0):.1f}% (actions that open/close)")
        print(f"  Loss:         {metrics.get('loss', 0):.2f}")

        # Show trends
        if 'expectancy_trend' in stats:
            trend_symbol = "📈" if stats['expectancy_trend'] > 0 else "📉"
            print(f"  Exp. Trend:   {trend_symbol} {stats['expectancy_trend']:+.3f} pips")
        if 'winrate_trend' in stats:
            trend_symbol = "📈" if stats['winrate_trend'] > 0 else "📉"
            print(f"  WR Trend:     {trend_symbol} {stats['winrate_trend']:+.1f}%")
        print()

        # Trade Statistics section
        print("📊 TRADE STATISTICS")
        print("─" * 40)
        if 'val_trades' in metrics:
            print(f"  Validation Trades:     {metrics['val_trades']}")
            print(f"  Validation Expectancy: {metrics.get('val_expectancy', 0):.2f} pips")
            print(f"  Validation Win Rate:   {metrics.get('val_win_rate', 0):.1f}%")

        if 'trades_per_episode' in stats:
            print(f"  Trades/Episode:        {stats['trades_per_episode']:.1f}")
            print(f"  Avg Trade Duration:    {stats.get('avg_trade_duration', 0):.0f} bars")
            print(f"  Experiences/Episode:   {stats.get('exp_per_episode', 0):.0f}")

        # Estimate daily trades at current speed
        if 'eps' in metrics and 'trades_per_episode' in stats:
            trades_per_day = metrics['eps'] * stats['trades_per_episode'] * 86400
            print(f"  Training Rate:         {trades_per_day:.0f} trades/day")
        print()

        # Action Distribution
        if 'actions' in metrics:
            print("🎮 ACTION DISTRIBUTION")
            print("─" * 40)
            actions = metrics['actions']

            # Create visual bars
            for action, pct in actions.items():
                bar_len = int(pct / 2)  # Scale to 50 chars max
                bar = "█" * bar_len
                print(f"  {action:5} {pct:5.1f}% {bar}")

            # Check balance
            trade_actions = actions['BUY'] + actions['SELL'] + actions['CLOSE']
            print(f"  Trading Actions: {trade_actions:.1f}% | Hold: {actions['HOLD']:.1f}%")
            print()

        print("─" * 80)
        print(f"Updated: {datetime.now().strftime('%H:%M:%S')} | Refresh: 2s | Press Ctrl+C to exit")

    def run(self):
        """Main dashboard loop."""
        print("Starting advanced dashboard...")

        while True:
            try:
                logs = self.get_logs()
                metrics = self.parse_metrics(logs)
                stats = self.calculate_trade_stats(metrics)
                self.display(metrics, stats)
                time.sleep(2)

            except KeyboardInterrupt:
                print("\nDashboard stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)

if __name__ == "__main__":
    dashboard = AdvancedDashboard()
    dashboard.run()