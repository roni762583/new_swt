#!/bin/bash
# Live monitoring script for PPO training with auto-restart capability

CONTAINER_NAME="ppo_training_improved"
TARGET_TIMESTEPS=1000000
CHECK_INTERVAL=60  # seconds
STUCK_THRESHOLD=5  # restart if no progress for 5 checks

last_timesteps=0
stuck_count=0

echo "🚀 PPO Training Monitor - Live Tracking"
echo "======================================="
echo "Target: $TARGET_TIMESTEPS timesteps"
echo ""

while true; do
    # Get current timestamp
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if container is running
    if ! docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
        echo "[$NOW] ❌ Container not running! Attempting restart..."
        docker restart $CONTAINER_NAME
        sleep 30
        continue
    fi

    # Get current metrics
    LOGS=$(docker logs $CONTAINER_NAME 2>&1 | tail -500)

    # Extract timesteps
    TIMESTEPS=$(echo "$LOGS" | grep "total_timesteps" | tail -1 | awk '{print $3}' | tr -d '|')

    # Extract win rate
    WIN_RATE=$(echo "$LOGS" | grep "win_rate" | tail -1 | awk '{print $3}')

    # Extract latest trade info
    LATEST_TRADE=$(echo "$LOGS" | grep "Trade #" | tail -1)

    # Check for errors
    ERROR_COUNT=$(echo "$LOGS" | grep -c "ERROR\|Error\|error" || true)

    # Display status
    clear
    echo "🚀 PPO Training Monitor - Live Tracking"
    echo "======================================="
    echo "[$NOW]"
    echo ""

    if [ -n "$TIMESTEPS" ]; then
        PROGRESS=$(awk "BEGIN {printf \"%.2f\", $TIMESTEPS / $TARGET_TIMESTEPS * 100}")
        echo "📊 Progress: $TIMESTEPS / $TARGET_TIMESTEPS ($PROGRESS%)"
        echo "📈 Win Rate: $WIN_RATE"

        # Check if training is stuck
        if [ "$TIMESTEPS" = "$last_timesteps" ] && [ "$TIMESTEPS" -gt 0 ]; then
            stuck_count=$((stuck_count + 1))
            echo "⚠️  No progress for $stuck_count checks"

            if [ $stuck_count -ge $STUCK_THRESHOLD ]; then
                echo "🔄 Training appears stuck. Restarting container..."
                docker restart $CONTAINER_NAME
                stuck_count=0
                sleep 30
            fi
        else
            stuck_count=0
        fi

        last_timesteps=$TIMESTEPS

        # Check if complete
        if [ "$TIMESTEPS" -ge "$TARGET_TIMESTEPS" ]; then
            echo ""
            echo "✅ Training Complete!"
            break
        fi
    else
        echo "⏳ Waiting for metrics..."
    fi

    if [ -n "$LATEST_TRADE" ]; then
        echo ""
        echo "💹 Latest: $LATEST_TRADE"
    fi

    if [ $ERROR_COUNT -gt 0 ]; then
        echo ""
        echo "⚠️  Errors detected: $ERROR_COUNT"
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Next check in ${CHECK_INTERVAL}s..."

    sleep $CHECK_INTERVAL
done