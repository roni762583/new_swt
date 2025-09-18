#!/bin/bash
# Start all Micro MuZero containers (training, validation, live)

echo "🚀 Starting Micro MuZero System"
echo "================================"

# Check if training is already running
if docker ps | grep -q micro_training; then
    echo "⚠️ Training container already running"
else
    echo "Starting training container..."
fi

# Build and start all containers
echo "Building and starting all containers..."
docker compose up -d --build micro-training micro-validation micro-live

# Wait for containers to start
sleep 5

# Show status
echo ""
echo "📊 Container Status:"
echo "-------------------"
docker ps | grep micro_ | awk '{print $NF, $7}'

echo ""
echo "📈 Training Status:"
docker logs micro_training 2>&1 | tail -3

echo ""
echo "✅ System Ready!"
echo ""
echo "Monitor with:"
echo "  Training:    docker logs -f micro_training"
echo "  Validation:  docker logs -f micro_validation"
echo "  Live:        docker logs -f micro_live"
echo ""
echo "Stop all with: docker compose down"