# Common Issues & Solutions

## 🚨 Troubleshooting Guide

This guide covers the most frequently encountered issues in the SWT Live Trading System and their solutions.

## 📋 Quick Diagnosis

### System Health Check
```bash
# Check all services
docker-compose ps

# Check health endpoints
curl -f http://localhost:8080/health | jq .

# Check resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=100 swt-live-trader | grep -i error
```

## 🔧 Common Issues

### 1. Container Startup Issues

#### Issue: Container Fails to Start
**Symptoms**:
```
swt-live-trader exited with code 1
Error: Failed to load checkpoint
```

**Diagnosis**:
```bash
# Check container logs
docker-compose logs swt-live-trader

# Check configuration
docker-compose exec swt-live-trader python -c "
from swt_live.config import load_config
config = load_config('/app/config/live.yaml')
print('Config loaded successfully')
"
```

**Solutions**:

1. **Missing Episode 13475 Checkpoint**:
```bash
# Verify checkpoint exists
ls -la checkpoints/episode_13475.pth

# Check checkpoint integrity
python -c "
import torch
checkpoint = torch.load('checkpoints/episode_13475.pth', map_location='cpu')
print(f'Checkpoint valid: {\"episode\" in checkpoint}')
"

# If missing, copy from backup
cp /backup/checkpoints/episode_13475.pth checkpoints/
```

2. **Invalid Configuration**:
```bash
# Validate YAML syntax
python -c "
import yaml
with open('config/live.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('YAML syntax valid')
"

# Check environment variables
docker-compose exec swt-live-trader env | grep SWT_
```

3. **Permission Issues**:
```bash
# Fix ownership
sudo chown -R $(id -u):$(id -g) checkpoints/ config/ logs/

# Fix permissions
chmod 644 config/live.yaml
chmod 600 config/production.env
chmod 755 checkpoints/
```

#### Issue: Out of Memory (OOM) Killed
**Symptoms**:
```
Container swt-live-trader killed (OOMKilled)
```

**Solutions**:
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G  # Increase from 2G

# Check memory usage patterns
docker stats --no-stream

# Enable memory monitoring
docker-compose exec swt-live-trader python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

### 2. OANDA API Issues

#### Issue: Authentication Failures
**Symptoms**:
```
401 Unauthorized: Invalid authorization header
403 Forbidden: Account access denied
```

**Diagnosis**:
```bash
# Test API credentials
curl -H "Authorization: Bearer $SWT_OANDA_API_TOKEN" \
     "https://api-fxpractice.oanda.com/v3/accounts/$SWT_OANDA_ACCOUNT_ID"
```

**Solutions**:

1. **Invalid Credentials**:
```bash
# Verify environment variables
echo "Account: $SWT_OANDA_ACCOUNT_ID"
echo "Token: ${SWT_OANDA_API_TOKEN:0:10}..."  # Show first 10 chars

# Update credentials
export SWT_OANDA_ACCOUNT_ID="your_correct_account_id"
export SWT_OANDA_API_TOKEN="your_correct_token"

# Restart container
docker-compose restart swt-live-trader
```

2. **Wrong Environment (Live vs Practice)**:
```bash
# Check current environment
echo "Environment: $SWT_OANDA_ENVIRONMENT"

# For practice trading
export SWT_OANDA_ENVIRONMENT=practice

# For live trading (CAREFUL!)
export SWT_OANDA_ENVIRONMENT=live
```

#### Issue: Connection Timeouts
**Symptoms**:
```
TimeoutError: Request timed out after 30 seconds
ConnectionError: Failed to connect to OANDA API
```

**Solutions**:
```bash
# Test network connectivity
curl -I https://api-fxpractice.oanda.com

# Check DNS resolution
nslookup api-fxpractice.oanda.com

# Test with increased timeout
docker-compose exec swt-live-trader python -c "
import aiohttp
import asyncio

async def test_connection():
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get('https://api-fxpractice.oanda.com/health') as resp:
            print(f'Status: {resp.status}')

asyncio.run(test_connection())
"
```

#### Issue: Rate Limiting
**Symptoms**:
```
429 Too Many Requests: Rate limit exceeded
```

**Solutions**:
```bash
# Reduce request frequency in config/live.yaml
market_data:
  stream:
    heartbeat_interval: 10  # Increase from 5

oanda:
  api:
    rate_limit: 50  # Reduce from 100
    retry_delay: 2  # Increase delay
```

### 3. Trading System Issues

#### Issue: No Trading Decisions
**Symptoms**:
```
INFO: Market data received, but no trading decisions made
WARN: Confidence below threshold: 0.4 < 0.6
```

**Diagnosis**:
```bash
# Check trading configuration
docker-compose exec swt-live-trader python -c "
from swt_live.config import load_config
config = load_config('/app/config/live.yaml')
print(f'Min confidence: {config.trading.min_confidence}')
print(f'Decision frequency: {config.trading.decision_frequency}')
"

# Check recent decisions
docker-compose logs swt-live-trader | grep -i "trading decision"
```

**Solutions**:

1. **Lower Confidence Threshold**:
```yaml
# In config/live.yaml
trading:
  min_confidence: 0.4  # Reduce from 0.6
```

2. **Check Market Conditions**:
```bash
# Verify market data flow
docker-compose logs swt-live-trader | grep -i "price update"

# Check for market holidays/closures
python -c "
from datetime import datetime
import pytz
now = datetime.now(pytz.timezone('UTC'))
print(f'Current time (UTC): {now}')
print(f'Is weekend: {now.weekday() >= 5}')
"
```

#### Issue: Excessive Trading
**Symptoms**:
```
WARN: Daily trade limit exceeded: 25 > 20
ERROR: Daily loss limit exceeded: -600 > -500
```

**Solutions**:
```bash
# Increase limits in config/live.yaml
trading:
  risk:
    max_daily_trades: 30  # Increase limit
    max_daily_loss: 750.0  # Increase limit

# Or temporarily pause trading
docker-compose exec swt-live-trader python -c "
import redis
r = redis.Redis(host='redis', port=6379, db=0)
r.set('swt:trading:paused', '1', ex=3600)  # Pause for 1 hour
print('Trading paused for 1 hour')
"
```

#### Issue: Position Reconciliation Failures
**Symptoms**:
```
ERROR: Position mismatch detected
Local: {'GBP_JPY': {'units': 1000, 'side': 'long'}}
Broker: {'GBP_JPY': {'units': 1500, 'side': 'long'}}
```

**Solutions**:
```bash
# Force position reconciliation
docker-compose exec swt-live-trader python -c "
from swt_live.position_reconciler import SWTPositionReconciler
import asyncio

async def force_reconcile():
    reconciler = SWTPositionReconciler()
    result = await reconciler.reconcile_positions()
    print(f'Reconciliation result: {result}')

asyncio.run(force_reconcile())
"

# Manual position correction
python -c "
# Get current broker positions
from swt_live.brokers.oanda_client import OANDAClient
import asyncio

async def get_positions():
    client = OANDAClient()
    positions = await client.get_positions()
    for pos in positions:
        print(f'{pos.instrument}: {pos.long.units} long, {pos.short.units} short')

asyncio.run(get_positions())
"
```

### 4. Performance Issues

#### Issue: High Latency
**Symptoms**:
```
WARN: Inference time exceeded threshold: 2.5s > 1.0s
WARN: Trading decision latency: 5.2s
```

**Diagnosis**:
```bash
# Check system resources
docker stats --no-stream

# Profile inference performance
docker-compose exec swt-live-trader python scripts/benchmark_system.py \
  --config config/live.yaml \
  --duration 60
```

**Solutions**:

1. **Resource Optimization**:
```bash
# Increase CPU allocation
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4'  # Increase from 2
```

2. **Model Optimization**:
```yaml
# In config/live.yaml
agent:
  mcts:
    num_simulations: 10  # Reduce from 15
```

3. **Feature Processing Optimization**:
```yaml
features:
  window_size: 50  # Reduce from 100
```

#### Issue: Memory Leaks
**Symptoms**:
```
Memory usage steadily increasing
Container restart required every few hours
```

**Solutions**:
```bash
# Enable memory profiling
docker-compose exec swt-live-trader python -c "
import psutil
import gc
import torch

# Check memory usage
print(f'Memory usage: {psutil.virtual_memory().percent}%')

# Force garbage collection
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f'After GC: {psutil.virtual_memory().percent}%')
"

# Add memory monitoring to config/live.yaml
monitoring:
  alerts:
    system:
      memory_usage_threshold: 85  # Alert at 85%
```

### 5. Data Issues

#### Issue: Missing Market Data
**Symptoms**:
```
WARN: No price updates received for 300 seconds
ERROR: Data gap detected: last update 10 minutes ago
```

**Solutions**:
```bash
# Check OANDA stream status
docker-compose exec swt-live-trader python -c "
from swt_live.data_feed import OANDADataFeed
import asyncio

async def check_stream():
    feed = OANDADataFeed()
    status = await feed.get_stream_status()
    print(f'Stream status: {status}')

asyncio.run(check_stream())
"

# Restart data feed
docker-compose restart swt-live-trader
```

#### Issue: Invalid Price Data
**Symptoms**:
```
ERROR: Price validation failed: change too large
Invalid price: 195.123 -> 205.456 (5.3% change)
```

**Solutions**:
```yaml
# Adjust validation thresholds in config/live.yaml
market_data:
  validation:
    price_change_threshold: 0.2  # Increase from 0.1
    gap_tolerance: 600  # Increase from 300
```

### 6. Redis Connection Issues

#### Issue: Redis Connection Failed
**Symptoms**:
```
ConnectionError: Error 111 connecting to redis:6379
redis.exceptions.ConnectionError: Connection refused
```

**Solutions**:
```bash
# Check Redis container status
docker-compose ps redis

# Restart Redis
docker-compose restart redis

# Test Redis connectivity
docker-compose exec redis redis-cli ping

# Check Redis configuration
docker-compose exec redis redis-cli info
```

## 🔍 Diagnostic Tools

### 1. Log Analysis Script
```bash
#!/bin/bash
# analyze_logs.sh - Quick log analysis

echo "=== Error Summary ==="
docker-compose logs swt-live-trader | grep -i error | tail -10

echo "=== Warning Summary ==="
docker-compose logs swt-live-trader | grep -i warn | tail -10

echo "=== Trading Activity ==="
docker-compose logs swt-live-trader | grep -i "trade executed" | tail -5

echo "=== Performance Metrics ==="
docker-compose logs swt-live-trader | grep -i "latency\|memory\|cpu" | tail -5
```

### 2. Health Check Script
```bash
#!/bin/bash
# health_check.sh - Comprehensive health check

echo "=== Container Status ==="
docker-compose ps

echo "=== Resource Usage ==="
docker stats --no-stream

echo "=== API Connectivity ==="
curl -sf http://localhost:8080/health || echo "Health check failed"

echo "=== OANDA Connectivity ==="
curl -sf "https://api-fxpractice.oanda.com/health" || echo "OANDA unreachable"

echo "=== Redis Status ==="
docker-compose exec -T redis redis-cli ping || echo "Redis connection failed"
```

### 3. Performance Monitor
```bash
#!/bin/bash
# monitor_performance.sh - Real-time performance monitoring

while true; do
    clear
    echo "=== $(date) ==="
    
    # Container stats
    docker stats --no-stream | head -2
    
    # API response time
    curl -w "Health check: %{time_total}s\n" -s -o /dev/null http://localhost:8080/health
    
    # Recent activity
    docker-compose logs --tail=3 swt-live-trader | grep -E "(trade|decision|error)"
    
    sleep 30
done
```

## 📞 Getting Help

### 1. Collect Debug Information
```bash
# Generate debug report
./scripts/generate_debug_report.sh

# The report includes:
# - System configuration
# - Container status
# - Recent logs
# - Performance metrics
# - Network connectivity tests
```

### 2. Emergency Procedures

**Emergency Stop**:
```bash
# Stop all trading immediately
docker-compose down

# Or pause trading only
docker-compose exec swt-live-trader python -c "
import redis
r = redis.Redis(host='redis', port=6379, db=0)
r.set('swt:emergency_stop', '1')
print('Emergency stop activated')
"
```

**Recovery Procedure**:
```bash
# 1. Stop system
docker-compose down

# 2. Check and fix issues
./scripts/validate_environment.sh

# 3. Restart system
docker-compose up -d

# 4. Verify operation
./scripts/health_check.sh
```

### 3. Support Escalation

Before escalating issues:

1. **Collect logs**: Last 1000 lines from all containers
2. **System status**: Output of `docker-compose ps` and `docker stats`
3. **Configuration**: Sanitized configuration files (remove secrets)
4. **Error reproduction**: Steps to reproduce the issue
5. **Impact assessment**: Trading impact and urgency level

---

**Next**: [Performance Tuning](performance.md) | [Log Analysis](log-analysis.md)