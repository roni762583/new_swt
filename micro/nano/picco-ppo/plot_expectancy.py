#!/usr/bin/env python3
"""
Live plot of rolling expectancy during training.
Reads from results/rolling_expectancy.json and displays graph.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import numpy as np

def load_expectancy_data(filepath="results/rolling_expectancy.json"):
    """Load expectancy data from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def create_expectancy_plot():
    """Create live updating plot of rolling expectancy."""

    # Setup figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('PPO Trading Performance - Live Monitoring', fontsize=16)

    # Data storage for plotting
    episodes = []
    expectancy_100 = []
    expectancy_500 = []
    expectancy_1000 = []
    lifetime_exp = []
    cumulative_pips = []
    win_rates = []

    def update_plot(frame):
        """Update plot with latest data."""
        data = load_expectancy_data()

        if data is None:
            return

        # Clear previous data
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Extract current values
        current_episode = data.get('episode', 0)

        # Store data points
        if current_episode not in episodes:
            episodes.append(current_episode)
            expectancy_100.append(data.get('expectancy_R_100', 0))
            expectancy_500.append(data.get('expectancy_R_500', 0))
            expectancy_1000.append(data.get('expectancy_R_1000', 0))
            lifetime_exp.append(data.get('lifetime_expectancy_R', 0))
            cumulative_pips.append(data.get('cumulative_pips', 0))
            win_rates.append(data.get('lifetime_win_rate', 0))

        # Plot 1: Rolling Expectancy
        ax1.plot(episodes, expectancy_100, 'b-', label='100-trade', linewidth=1, alpha=0.7)
        ax1.plot(episodes, expectancy_500, 'g-', label='500-trade', linewidth=2, alpha=0.8)
        ax1.plot(episodes, expectancy_1000, 'r-', label='1000-trade', linewidth=2)
        ax1.plot(episodes, lifetime_exp, 'k--', label='Lifetime', linewidth=2)

        # Add quality zones
        ax1.axhline(y=0.5, color='gold', linestyle='--', alpha=0.3, label='Excellent (>0.5R)')
        ax1.axhline(y=0.25, color='green', linestyle='--', alpha=0.3, label='Good (>0.25R)')
        ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3)

        ax1.set_ylabel('Expectancy (R-multiples)')
        ax1.set_title('Rolling Expectancy Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # Plot 2: Cumulative P&L
        ax2.plot(episodes, cumulative_pips, 'navy', linewidth=2)
        ax2.fill_between(episodes, 0, cumulative_pips, alpha=0.3, color='blue')
        ax2.set_ylabel('Cumulative Pips')
        ax2.set_title('Total Performance')
        ax2.grid(True, alpha=0.3)

        # Add current value annotation
        if len(cumulative_pips) > 0:
            latest_pips = cumulative_pips[-1]
            ax2.text(0.98, 0.95, f'Total: {latest_pips:+.1f} pips',
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 3: Win Rate
        ax3.plot(episodes, win_rates, 'green', linewidth=2)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(episodes, 50, win_rates, where=np.array(win_rates)>50,
                         alpha=0.3, color='green', label='Above 50%')
        ax3.fill_between(episodes, 50, win_rates, where=np.array(win_rates)<=50,
                         alpha=0.3, color='red', label='Below 50%')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')

        # Add statistics box
        if data:
            stats_text = (
                f"Episode: {data.get('episode', 0)}\n"
                f"Total Trades: {data.get('total_trades', 0):,}\n"
                f"Latest Expectancy: {data.get('lifetime_expectancy_R', 0):+.3f}R\n"
                f"Win Rate: {data.get('lifetime_win_rate', 0):.1f}%"
            )
            fig.text(0.15, 0.89, stats_text, transform=fig.transFigure,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)

        # Add timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        fig.text(0.99, 0.01, f'Updated: {timestamp}', transform=fig.transFigure,
                ha='right', fontsize=9, color='gray')

        plt.tight_layout()

    # Animation
    ani = animation.FuncAnimation(fig, update_plot, interval=5000, cache_frame_data=False)

    plt.show()

def generate_static_plot():
    """Generate a static plot and save to file."""
    data = load_expectancy_data()

    if data is None:
        print("No expectancy data found")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Get window data
    windows = [100, 500, 1000]
    expectancies = []
    win_rates = []

    for window in windows:
        exp_key = f'expectancy_R_{window}'
        wr_key = f'win_rate_{window}'
        expectancies.append(data.get(exp_key, 0))
        win_rates.append(data.get(wr_key, 0))

    # Plot expectancy bars
    colors = ['red' if e < 0 else 'yellow' if e < 0.25 else 'lightgreen' if e < 0.5 else 'green'
              for e in expectancies]
    bars1 = ax1.bar(windows, expectancies, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.25, color='green', linestyle='--', alpha=0.5, label='Good (0.25R)')
    ax1.axhline(y=0.5, color='gold', linestyle='--', alpha=0.5, label='Excellent (0.5R)')
    ax1.set_xlabel('Window Size (trades)')
    ax1.set_ylabel('Expectancy (R-multiples)')
    ax1.set_title(f'Rolling Expectancy - Episode {data.get("episode", 0)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, expectancies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.3f}R', ha='center', va='bottom' if val >= 0 else 'top')

    # Plot win rates
    bars2 = ax2.bar(windows, win_rates, color='steelblue', alpha=0.7)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_xlabel('Window Size (trades)')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom')

    # Add summary text
    summary = (
        f"Total Trades: {data.get('total_trades', 0):,}\n"
        f"Cumulative P&L: {data.get('cumulative_pips', 0):+.1f} pips\n"
        f"Lifetime Expectancy: {data.get('lifetime_expectancy_R', 0):+.3f}R\n"
        f"Lifetime Win Rate: {data.get('lifetime_win_rate', 0):.1f}%"
    )
    fig.text(0.72, 0.45, summary, transform=fig.transFigure,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            fontsize=11)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/expectancy_plot_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to {filename}")

    return fig

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--static':
        # Generate static plot
        fig = generate_static_plot()
        plt.show()
    else:
        # Live updating plot
        print("📊 Starting live expectancy monitoring...")
        print("   Updates every 5 seconds")
        print("   Press Ctrl+C to stop")
        create_expectancy_plot()