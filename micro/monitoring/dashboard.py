#!/usr/bin/env python3
"""
Simple text-based terminal dashboard for monitoring MuZero training.
Updates every 2 seconds with all key metrics.
"""

import os
import sys
import time
import json
import pickle
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional
import curses

class TrainingDashboard:
    """Real-time training metrics dashboard."""

    def __init__(self, container_name: str = "micro_training"):
        self.container_name = container_name
        self.start_time = datetime.now()
        self.last_episode = 0
        self.episode_times = []

    def get_container_logs(self, lines: int = 100) -> str:
        """Get recent container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name, "--tail", str(lines)],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout + result.stderr
        except:
            return ""

    def parse_latest_metrics(self, logs: str) -> Dict:
        """Parse metrics from logs."""
        metrics = {
            'episode': 0,
            'steps': 0,
            'expectancy': 0.0,
            'win_rate': 0.0,
            'trade_ratio': 0.0,
            'loss': 0.0,
            'buffer_size': 0,
            'action_dist': {},
            'eps': 0.0  # episodes per second
        }

        # Find latest episode line
        for line in reversed(logs.split('\n')):
            if 'Episode' in line and '|' in line and 'Exp:' in line:
                try:
                    # Parse: Episode 1900 | Steps 684000 | EPS: 7.0 | Loss: 83.7152 | TD: 0.000 | Exp: -4.01 | WR: 7.4% | TradeRatio: 74.7%
                    parts = line.split('|')
                    for part in parts:
                        if 'Episode' in part:
                            metrics['episode'] = int(part.split()[-1])
                        elif 'Steps' in part:
                            metrics['steps'] = int(part.split()[-1])
                        elif 'EPS:' in part:
                            metrics['eps'] = float(part.split()[-1])
                        elif 'Loss:' in part:
                            metrics['loss'] = float(part.split()[-1])
                        elif 'Exp:' in part:
                            metrics['expectancy'] = float(part.split()[-1])
                        elif 'WR:' in part:
                            metrics['win_rate'] = float(part.split()[-1].rstrip('%'))
                        elif 'TradeRatio:' in part:
                            metrics['trade_ratio'] = float(part.split()[-1].rstrip('%'))
                    break
                except:
                    continue

        # Find buffer size
        for line in reversed(logs.split('\n')):
            if 'buffer_size' in line:
                try:
                    import re
                    match = re.search(r'buffer_size.*?(\d+)', line)
                    if match:
                        metrics['buffer_size'] = int(match.group(1))
                        break
                except:
                    pass

        # Find action distribution
        for line in reversed(logs.split('\n')):
            if 'Action distribution' in line:
                try:
                    # Parse: Action distribution - HOLD: 25.8%, BUY: 28.1%, SELL: 22.6%, CLOSE: 23.5%
                    parts = line.split('-')[1].strip()
                    for action_part in parts.split(','):
                        action, pct = action_part.strip().split(':')
                        metrics['action_dist'][action] = float(pct.rstrip('%'))
                    break
                except:
                    continue

        return metrics

    def get_container_status(self) -> str:
        """Check if container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip() if result.stdout else "Not running"
        except:
            return "Unknown"

    def calculate_eta(self, current_episode: int, target: int = 1000000) -> str:
        """Calculate ETA to target episodes."""
        if current_episode == 0 or current_episode == self.last_episode:
            return "Calculating..."

        # Track episode progress
        if current_episode > self.last_episode:
            self.episode_times.append((datetime.now(), current_episode))
            self.last_episode = current_episode

            # Keep only last 10 measurements
            if len(self.episode_times) > 10:
                self.episode_times.pop(0)

        # Calculate rate
        if len(self.episode_times) < 2:
            return "Calculating..."

        time_diff = (self.episode_times[-1][0] - self.episode_times[0][0]).total_seconds()
        episode_diff = self.episode_times[-1][1] - self.episode_times[0][1]

        if episode_diff > 0:
            episodes_per_sec = episode_diff / time_diff
            remaining = target - current_episode
            eta_seconds = remaining / episodes_per_sec

            # Format as days, hours, minutes
            days = int(eta_seconds // 86400)
            hours = int((eta_seconds % 86400) // 3600)
            minutes = int((eta_seconds % 3600) // 60)

            if days > 0:
                return f"{days}d {hours}h {minutes}m"
            elif hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"

        return "Calculating..."

    def format_dashboard(self, metrics: Dict, status: str) -> list:
        """Format dashboard display."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(" " * 25 + "🚀 MUZERO TRAINING DASHBOARD 🚀")
        lines.append("=" * 80)
        lines.append("")

        # Container status
        lines.append(f"📦 Container: {self.container_name} | Status: {status}")
        lines.append(f"⏱️  Uptime: {str(datetime.now() - self.start_time).split('.')[0]}")
        lines.append("")

        # Progress
        episode = metrics['episode']
        target = 1000000
        progress = (episode / target) * 100
        eta = self.calculate_eta(episode, target)

        lines.append("📊 TRAINING PROGRESS")
        lines.append("─" * 40)
        lines.append(f"Episode:     {episode:,} / {target:,} ({progress:.2f}%)")
        lines.append(f"Steps:       {metrics['steps']:,}")
        lines.append(f"Speed:       {metrics['eps']:.1f} episodes/sec")
        lines.append(f"ETA:         {eta}")

        # Progress bar
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        lines.append(f"Progress:    [{bar}]")
        lines.append("")

        # Trading Performance
        lines.append("💰 TRADING PERFORMANCE")
        lines.append("─" * 40)
        lines.append(f"Expectancy:  {metrics['expectancy']:.2f} pips/trade")
        lines.append(f"Win Rate:    {metrics['win_rate']:.1f}%")
        lines.append(f"Trade Ratio: {metrics['trade_ratio']:.1f}% (min 30%)")

        # Visual indicator for expectancy
        exp_bar = "🟢" if metrics['expectancy'] > 0 else "🔴"
        lines.append(f"Trend:       {exp_bar} {'Profitable' if metrics['expectancy'] > 0 else 'Learning'}")
        lines.append("")

        # Model Metrics
        lines.append("🧠 MODEL METRICS")
        lines.append("─" * 40)
        lines.append(f"Loss:        {metrics['loss']:.2f}")
        lines.append(f"Buffer:      {metrics['buffer_size']:,} / 10,000")
        lines.append("")

        # Action Distribution
        lines.append("🎯 ACTION DISTRIBUTION")
        lines.append("─" * 40)
        if metrics['action_dist']:
            for action, pct in metrics['action_dist'].items():
                bar_len = int(pct / 2)  # Scale to 50 chars max
                bar = "█" * bar_len
                lines.append(f"{action:6} {pct:5.1f}% {bar}")
        else:
            lines.append("No action data available")
        lines.append("")

        # Reward Analysis
        lines.append("🏆 REWARD ANALYSIS")
        lines.append("─" * 40)

        # Estimate from win rate and expectancy
        if metrics['win_rate'] > 0:
            avg_win = 10.0  # Estimate
            avg_loss = (metrics['win_rate'] * avg_win / 100 - metrics['expectancy']) / (1 - metrics['win_rate'] / 100)
            lines.append(f"Est. Avg Win:  +{avg_win:.1f} pips")
            lines.append(f"Est. Avg Loss: -{avg_loss:.1f} pips")
            lines.append(f"Risk/Reward:   1:{avg_win/avg_loss:.1f}")
        else:
            lines.append("Insufficient data")
        lines.append("")

        # Footer
        lines.append("─" * 80)
        lines.append(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refresh: 2s | Press Ctrl+C to exit")

        return lines

    def run_curses(self, stdscr):
        """Run dashboard with curses for clean updates."""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)    # Non-blocking input
        stdscr.timeout(2000) # 2 second refresh

        while True:
            try:
                # Get latest data
                logs = self.get_container_logs(200)
                metrics = self.parse_latest_metrics(logs)
                status = self.get_container_status()

                # Format dashboard
                lines = self.format_dashboard(metrics, status)

                # Clear and redraw
                stdscr.clear()

                # Display lines
                max_y, max_x = stdscr.getmaxyx()
                for i, line in enumerate(lines[:max_y-1]):
                    if i < max_y - 1:
                        # Truncate line if too long
                        if len(line) > max_x - 1:
                            line = line[:max_x-1]
                        try:
                            # Color coding
                            if "🚀" in line or "DASHBOARD" in line:
                                stdscr.addstr(i, 0, line, curses.A_BOLD)
                            elif "█" in line:
                                stdscr.addstr(i, 0, line, curses.A_DIM)
                            else:
                                stdscr.addstr(i, 0, line)
                        except:
                            pass

                stdscr.refresh()

                # Check for quit
                key = stdscr.getch()
                if key == ord('q') or key == 27:  # q or ESC
                    break

            except KeyboardInterrupt:
                break
            except Exception as e:
                # Show error
                stdscr.clear()
                stdscr.addstr(0, 0, f"Error: {e}")
                stdscr.refresh()
                time.sleep(2)

    def run_simple(self):
        """Run dashboard with simple print (no curses)."""
        while True:
            try:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')

                # Get latest data
                logs = self.get_container_logs(200)
                metrics = self.parse_latest_metrics(logs)
                status = self.get_container_status()

                # Format and print
                lines = self.format_dashboard(metrics, status)
                for line in lines:
                    print(line)

                # Wait
                time.sleep(2)

            except KeyboardInterrupt:
                print("\nDashboard stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)


def main():
    """Main entry point."""
    dashboard = TrainingDashboard()

    # Try curses first, fall back to simple
    try:
        curses.wrapper(dashboard.run_curses)
    except:
        # Fallback to simple print
        dashboard.run_simple()


if __name__ == "__main__":
    main()