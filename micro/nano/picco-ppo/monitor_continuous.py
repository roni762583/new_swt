#!/usr/bin/env python3
"""
Continuous monitoring script for PPO training with auto-restart on failures.
"""

import subprocess
import time
import re
from datetime import datetime
import sys

def get_container_status():
    """Check if container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ppo_training_improved", "--format", "{{.Status}}"],
            capture_output=True, text=True
        )
        return "Up" in result.stdout
    except:
        return False

def get_latest_metrics():
    """Get latest training metrics from logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "ppo_training_improved", "--tail", "200"],
            capture_output=True, text=True, stderr=subprocess.STDOUT
        )

        metrics = {}

        # Extract timesteps
        timestep_match = re.search(r'total_timesteps\s+\|\s+(\d+)', result.stdout)
        if timestep_match:
            metrics['timesteps'] = int(timestep_match.group(1))

        # Extract win rate
        winrate_match = re.search(r'win_rate\s+\|\s+([\d.]+)', result.stdout)
        if winrate_match:
            metrics['win_rate'] = float(winrate_match.group(1))

        # Count recent trades
        trade_count = len(re.findall(r'Trade #\d+:', result.stdout))
        metrics['recent_trades'] = trade_count

        # Check for errors
        error_count = len(re.findall(r'ERROR|Error|error', result.stdout))
        metrics['errors'] = error_count

        return metrics
    except Exception as e:
        return {'error': str(e)}

def restart_container():
    """Restart the training container."""
    print("⚠️  Restarting container...")
    subprocess.run(["docker", "restart", "ppo_training_improved"])
    time.sleep(10)

def monitor_loop():
    """Main monitoring loop."""
    last_timestep = 0
    stuck_counter = 0

    print("🚀 PPO Training Monitor - Continuous")
    print("=====================================\n")

    while True:
        now = datetime.now().strftime("%H:%M:%S")

        # Check container status
        if not get_container_status():
            print(f"[{now}] ❌ Container not running! Restarting...")
            restart_container()
            continue

        # Get metrics
        metrics = get_latest_metrics()

        if 'error' in metrics:
            print(f"[{now}] ⚠️  Error getting metrics: {metrics['error']}")
        else:
            timesteps = metrics.get('timesteps', 0)
            win_rate = metrics.get('win_rate', 0)
            trades = metrics.get('recent_trades', 0)
            errors = metrics.get('errors', 0)

            # Display status
            print(f"[{now}] 📊 Timesteps: {timesteps:,} / 1,000,000 ({timesteps/10000:.1f}%)")
            print(f"         📈 Win Rate: {win_rate*100:.1f}%")
            print(f"         💹 Recent Trades: {trades}")

            if errors > 0:
                print(f"         ⚠️  Errors detected: {errors}")

            # Check if training is stuck
            if timesteps == last_timestep and timesteps > 0:
                stuck_counter += 1
                if stuck_counter >= 3:  # Stuck for 3 checks (3 minutes)
                    print(f"[{now}] ⚠️  Training appears stuck. Restarting...")
                    restart_container()
                    stuck_counter = 0
            else:
                stuck_counter = 0

            last_timestep = timesteps

            # Check if training complete
            if timesteps >= 1000000:
                print(f"\n[{now}] ✅ Training complete!")
                break

        print("")  # Blank line between updates
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        sys.exit(0)