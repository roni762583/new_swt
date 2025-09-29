#!/bin/bash
# Monitor improved PPO training progress

echo "🚀 PPO Training Monitor - Improved Version"
echo "=========================================="
echo ""

while true; do
    clear
    echo "🚀 PPO Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Check container status
    STATUS=$(docker ps --filter "name=ppo_training_improved" --format "{{.Status}}")
    echo "📦 Container Status: $STATUS"
    echo ""

    # Get latest metrics from logs
    echo "📊 Latest Metrics:"
    docker logs ppo_training_improved 2>&1 | grep -E "total_timesteps|win_rate|expectancy|gate_rate" | tail -5
    echo ""

    # Get recent trades
    echo "💹 Recent Trades:"
    docker logs ppo_training_improved 2>&1 | grep "Trade #" | tail -5
    echo ""

    # Get gate statistics
    echo "🚪 Gating Statistics:"
    docker logs ppo_training_improved 2>&1 | grep -E "rolling_std|threshold|Gate" | tail -3
    echo ""

    # Check for errors
    ERROR_COUNT=$(docker logs ppo_training_improved 2>&1 | grep -c ERROR)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "⚠️  Errors found: $ERROR_COUNT"
        docker logs ppo_training_improved 2>&1 | grep ERROR | tail -3
    fi

    echo ""
    echo "Press Ctrl+C to exit | Refreshing in 30s..."
    sleep 30
done