#!/bin/bash
# Start Episode 13475 Live Trading - SIMPLE VERSION
# One command to start live trading with real money

echo "💰 Episode 13475 Live Trading"
echo "⚠️  Trading REAL MONEY on OANDA live account!"

# Clean up any old containers
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# Start ONLY the live trading container (no extras)
echo "🚀 Starting live trading container..."
docker-compose -f docker-compose.episode13475-live.yml up -d --build

echo "📊 Live trading active! Showing logs:"
docker-compose -f docker-compose.episode13475-live.yml logs -f episode-13475-live