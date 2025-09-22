#!/bin/bash
# Cloud monitoring script for Speed Demon instance

INSTANCE_NAME="muzero-monster"
ZONE="us-central1-a"

echo "🔍 MUZERO CLOUD MONITOR"
echo "======================="

while true; do
    clear
    echo "🔍 MUZERO CLOUD MONITOR - $(date)"
    echo "======================================="

    # Check instance status
    STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null)
    if [ "$STATUS" == "RUNNING" ]; then
        echo "✅ Instance: RUNNING"

        # Get latest episode
        echo ""
        echo "📊 Latest Training Info:"
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
            docker logs micro_training_cloud 2>&1 | grep 'Episode' | tail -3
        " 2>/dev/null || echo "  Waiting for episode logs..."

        # Get checkpoint info
        echo ""
        echo "💾 Latest Checkpoint:"
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
            ls -lh micro/checkpoints/*.pth 2>/dev/null | tail -2 | awk '{print \"  \" \$NF \": \" \$5}'
        " 2>/dev/null || echo "  No checkpoints yet..."

        # Get resource usage
        echo ""
        echo "🖥️ Resource Usage:"
        gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
            docker stats --no-stream --format 'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}'
        " 2>/dev/null || echo "  No containers running..."

        # Calculate cost
        UPTIME=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(lastStartTimestamp)" 2>/dev/null)
        if [ ! -z "$UPTIME" ]; then
            UPTIME_SECONDS=$(($(date +%s) - $(date -d "$UPTIME" +%s)))
            UPTIME_HOURS=$(echo "scale=2; $UPTIME_SECONDS / 3600" | bc)
            COST=$(echo "scale=2; $UPTIME_HOURS * 2.00" | bc)
            echo ""
            echo "💰 Session Cost:"
            echo "  Uptime: ${UPTIME_HOURS} hours"
            echo "  Cost: \$${COST} (at \$2.00/hour spot)"
        fi

    elif [ "$STATUS" == "TERMINATED" ]; then
        echo "⚠️ Instance: STOPPED (spot preemption or manual stop)"
        echo ""
        echo "To restart: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    else
        echo "❌ Instance: $STATUS"
    fi

    echo ""
    echo "======================================="
    echo "Press Ctrl+C to exit | Refreshing in 30s..."
    sleep 30
done