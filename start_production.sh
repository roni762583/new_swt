#!/bin/bash
# Production startup script for SWT Trading System
# All 3 containers run concurrently with automatic checkpoint sharing

set -e

echo "🚀 Starting SWT Production Environment"
echo "======================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p shared_checkpoints
mkdir -p logs/{training,validation,live}
mkdir -p validation_results
mkdir -p sessions
chmod 777 shared_checkpoints

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose -f docker-compose.prod.yml down 2>/dev/null || true

# Build with cache optimization
echo "🔨 Building containers with cache optimization..."
docker compose -f docker-compose.prod.yml build --parallel

# Start all services
echo "🚀 Starting all services..."
docker compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Show status
echo ""
echo "📊 System Status:"
echo "=================="
docker compose -f docker-compose.prod.yml ps

echo ""
echo "📝 Container Logs:"
echo "=================="
echo "Training logs:    docker compose -f docker-compose.prod.yml logs training -f"
echo "Validation logs:  docker compose -f docker-compose.prod.yml logs validation -f"
echo "Live trading logs: docker compose -f docker-compose.prod.yml logs live -f"

echo ""
echo "🔄 Checkpoint Flow:"
echo "==================="
echo "1. Training → saves new checkpoints to /shared/checkpoints"
echo "2. Validation → monitors and validates new checkpoints automatically"
echo "3. Live Trading → uses latest validated checkpoint"

echo ""
echo "📈 Monitoring:"
echo "=============="
echo "Training metrics:   http://localhost:8081"
echo "Live trading health: http://localhost:8080/health"
echo "Container stats:     docker stats"

echo ""
echo "✅ Production environment started successfully!"
echo ""
echo "To stop all services: docker compose -f docker-compose.prod.yml down"