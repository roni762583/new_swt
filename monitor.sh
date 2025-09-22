#!/bin/bash
# Simple MuZero training monitor

while true; do
    clear
    echo "============================================================"
    echo "                    MUZERO TRAINING MONITOR"  
    echo "============================================================"
    echo ""
    
    # Get latest episode info
    EPISODE_LINE=$(docker logs micro_training 2>&1 | grep "Episode [0-9]" | tail -1)
    
    if [ ! -z "$EPISODE_LINE" ]; then
        # Parse the line
        EPISODE=$(echo "$EPISODE_LINE" | grep -oP 'Episode \K[0-9]+')
        EXPECTANCY=$(echo "$EPISODE_LINE" | grep -oP 'Exp: \K[-0-9.]+')
        WIN_RATE=$(echo "$EPISODE_LINE" | grep -oP 'WR: \K[0-9.]+')
        TRADE_RATIO=$(echo "$EPISODE_LINE" | grep -oP 'TradeRatio: \K[0-9.]+')
        EPS=$(echo "$EPISODE_LINE" | grep -oP 'EPS: \K[0-9.]+')
        LOSS=$(echo "$EPISODE_LINE" | grep -oP 'Loss: \K[0-9.]+')
        
        # Calculate progress
        PROGRESS=$(echo "scale=4; $EPISODE / 1000000 * 100" | bc)
        
        echo "📊 PROGRESS"
        echo "  Episode:     $EPISODE / 1,000,000 ($PROGRESS%)"
        echo "  Speed:       $EPS episodes/sec"
        echo ""
        
        echo "💰 PERFORMANCE"
        echo "  Expectancy:  $EXPECTANCY pips"
        echo "  Win Rate:    $WIN_RATE%"
        echo "  Trade Ratio: $TRADE_RATIO%"
        echo "  Loss:        $LOSS"
        echo ""
        
        # Get action distribution
        ACTION_LINE=$(docker logs micro_training 2>&1 | grep "Action distribution" | tail -1)
        if [ ! -z "$ACTION_LINE" ]; then
            ACTIONS=$(echo "$ACTION_LINE" | cut -d'-' -f2)
            echo "🎮 ACTIONS: $ACTIONS"
        fi
        
        echo ""
        echo "🐳 CONTAINERS"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep micro
    else
        echo "Waiting for training data..."
    fi
    
    echo ""
    echo "Updated: $(date '+%H:%M:%S') | Press Ctrl+C to exit"
    echo "------------------------------------------------------------"
    
    sleep 3
done
