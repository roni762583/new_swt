#!/bin/bash

# Startup script for Micro MuZero training system
# Builds and runs all containers: training, validation, and live

echo "=========================================="
echo "   MICRO MUZERO TRAINING SYSTEM"
echo "=========================================="

# Check if micro_features.duckdb exists
if [ ! -f "data/micro_features.duckdb" ]; then
    echo "⚠️  Building micro features database..."
    python3 data/prepare_micro_features.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build micro features database"
        exit 1
    fi
fi

echo "✅ Micro features database ready"

# Create required directories
echo "📁 Creating directories..."
mkdir -p micro/checkpoints
mkdir -p micro/logs
mkdir -p micro/validation_results
mkdir -p micro/live_state

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null

# Build and start all containers
echo "🚀 Building and starting containers..."
docker compose up -d --build

# Check if containers started
sleep 5
echo ""
echo "📊 Container Status:"
docker compose ps

echo ""
echo "=========================================="
echo "✅ TRAINING SYSTEM STARTED"
echo "=========================================="
echo ""
echo "Monitor logs with:"
echo "  docker compose logs -f training     # Training logs"
echo "  docker compose logs -f validation   # Validation logs"
echo "  docker compose logs -f live         # Live trading logs"
echo ""
echo "Stop with:"
echo "  docker compose down"
echo ""