#!/bin/bash
# Simple MuZero training monitor - non-flickering version

LAST_EPISODE=0
FIRST_RUN=true

while true; do
    # Get latest episode info (fetch once per loop)
    # Note: Search all logs for Episode lines (training logs may be sparse)
    EPISODE_LINE=$(docker logs micro_training 2>&1 | grep -E "Episode [0-9]+ \|" | tail -1)

    if [ ! -z "$EPISODE_LINE" ]; then
        # Parse episode number
        EPISODE=$(echo "$EPISODE_LINE" | grep -oP 'Episode \K[0-9]+')

        # Only update display if we have new data or first run
        if [ "$EPISODE" != "$LAST_EPISODE" ] || [ "$FIRST_RUN" = true ]; then
            FIRST_RUN=false
            LAST_EPISODE=$EPISODE

            # Parse all metrics
            EXPECTANCY=$(echo "$EPISODE_LINE" | grep -oP 'Exp: \K[-0-9.]+')
            WIN_RATE=$(echo "$EPISODE_LINE" | grep -oP 'WR: \K[0-9.]+')
            TRADE_RATIO=$(echo "$EPISODE_LINE" | grep -oP 'TradeRatio: \K[0-9.]+')
            EPS=$(echo "$EPISODE_LINE" | grep -oP 'EPS: \K[0-9.]+')
            LOSS=$(echo "$EPISODE_LINE" | grep -oP 'Loss: \K[0-9.]+')

            # Calculate progress
            PROGRESS=$(echo "scale=4; $EPISODE / 1000000 * 100" | bc)

            # Get action distribution
            ACTION_LINE=$(docker logs micro_training 2>&1 | grep "Action distribution" | tail -1)
            if [ ! -z "$ACTION_LINE" ]; then
                ACTIONS=$(echo "$ACTION_LINE" | cut -d'-' -f2)
            fi

            # Clear and redraw only when we have new data
            clear
            echo "============================================================"
            echo "                    MUZERO TRAINING MONITOR"
            echo "============================================================"
            echo ""

            echo "📊 PROGRESS"
            echo "  Episode:     $EPISODE / 1,000,000 ($PROGRESS%)"
            echo "  Speed:       $EPS episodes/sec"

            # Calculate ETA
            if [ ! -z "$EPS" ] && [ "$EPS" != "0" ]; then
                REMAINING=$((1000000 - EPISODE))
                ETA_SECONDS=$(echo "scale=0; $REMAINING / $EPS" | bc)
                ETA_HOURS=$(echo "scale=1; $ETA_SECONDS / 3600" | bc)
                ETA_DAYS=$(echo "scale=1; $ETA_HOURS / 24" | bc)
                echo "  ETA:         $ETA_DAYS days ($ETA_HOURS hours)"
            fi
            echo ""

            echo "💰 PERFORMANCE"
            echo "  Expectancy:  $EXPECTANCY pips"
            echo "  Win Rate:    $WIN_RATE%"
            echo "  Trade Ratio: $TRADE_RATIO%"
            echo "  Loss:        $LOSS"
            echo ""

            if [ ! -z "$ACTIONS" ]; then
                echo "🎮 ACTIONS: $ACTIONS"
                echo ""
            fi

            # Progress bar
            BAR_LEN=50
            FILLED=$(echo "scale=0; $BAR_LEN * $PROGRESS / 100" | bc)
            EMPTY=$((BAR_LEN - FILLED))
            printf "Progress: ["
            printf '=%.0s' $(seq 1 $FILLED)
            printf ' %.0s' $(seq 1 $EMPTY)
            printf "] %.3f%%\n" $PROGRESS
            echo ""

            echo "🐳 CONTAINERS"
            docker ps --format "table {{.Names}}\t{{.Status}}" | grep micro
            echo ""
            echo "Updated: $(date '+%H:%M:%S') | Refreshing every 3s | Press Ctrl+C to exit"
            echo "------------------------------------------------------------"
        fi
    else
        # Only show waiting message once
        if [ "$FIRST_RUN" = true ]; then
            echo "Waiting for training data..."
            FIRST_RUN=false
        fi
    fi

    sleep 3
done
