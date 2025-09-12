# Production Deployment Guide

## 🚀 Complete Production Deployment

This guide provides step-by-step instructions for deploying the SWT Live Trading System in a production environment.

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] Linux server (Ubuntu 20.04+ or CentOS 8+)
- [ ] 4+ CPU cores, 8GB+ RAM, 50GB+ SSD storage
- [ ] Docker 20.10+ and Docker Compose 2.0+
- [ ] Stable internet connection with backup
- [ ] OANDA broker account with API access

### Security Requirements
- [ ] Non-root user account configured
- [ ] SSH key-based authentication
- [ ] Firewall configured (UFW or iptables)
- [ ] SSL certificates obtained (Let's Encrypt recommended)
- [ ] Backup strategy implemented

### Prerequisites
- [ ] Episode 13475 checkpoint file available
- [ ] OANDA API credentials (Account ID and Token)
- [ ] System monitoring tools configured
- [ ] Incident response procedures documented

## 🔧 Step 1: Environment Setup

### 1.1 Create Deployment User
```bash
# Create dedicated deployment user
sudo useradd -m -s /bin/bash swt-trader
sudo usermod -aG docker swt-trader

# Setup SSH access
sudo mkdir -p /home/swt-trader/.ssh
sudo cp ~/.ssh/authorized_keys /home/swt-trader/.ssh/
sudo chown -R swt-trader:swt-trader /home/swt-trader/.ssh
sudo chmod 700 /home/swt-trader/.ssh
sudo chmod 600 /home/swt-trader/.ssh/authorized_keys
```

### 1.2 Install System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y curl wget git unzip htop tree jq

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker swt-trader

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
     -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 1.3 Configure Firewall
```bash
# Enable firewall
sudo ufw --force enable

# Allow SSH
sudo ufw allow ssh

# Allow application ports
sudo ufw allow 80/tcp     # Nginx HTTP
sudo ufw allow 443/tcp    # Nginx HTTPS
sudo ufw allow 8080/tcp   # Health check (optional, for monitoring)

# Check status
sudo ufw status verbose
```

## 📦 Step 2: Application Deployment

### 2.1 Deploy Application
```bash
# Switch to deployment user
sudo su - swt-trader

# Clone repository
git clone https://github.com/your-org/new_muzero.git
cd new_muzero/new_swt

# Create required directories
mkdir -p {config,checkpoints,data,logs,results,ssl}
```

### 2.2 Configuration Setup
```bash
# Copy configuration templates
cp config/live.yaml.example config/live.yaml
cp config/production.env.example config/production.env

# Edit configuration files
nano config/live.yaml
nano config/production.env
```

**Key Configuration Parameters** (`config/live.yaml`):
```yaml
# Trading Configuration
instrument: "GBP_JPY"
position_size: 1000  # Units per trade
min_confidence: 0.6  # Minimum confidence threshold

# Risk Management
max_daily_loss: 500.0
max_position_size: 5000
max_concurrent_positions: 1

# OANDA Configuration
oanda:
  environment: "live"  # or "practice" for testing
  stream_timeout: 30
  request_timeout: 10

# System Configuration
agent_system: "stochastic_muzero"
log_level: "INFO"
checkpoint_path: "/app/checkpoints/episode_13475.pth"
```

**Environment Variables** (`config/production.env`):
```bash
# OANDA Credentials (NEVER commit these to version control)
SWT_OANDA_ACCOUNT_ID=your_account_id_here
SWT_OANDA_API_TOKEN=your_api_token_here
SWT_OANDA_ENVIRONMENT=live

# System Configuration
SWT_ENVIRONMENT=production
SWT_LOG_LEVEL=INFO
SWT_CHECKPOINT_PATH=/app/checkpoints/episode_13475.pth

# Resource Limits
SWT_MAX_MEMORY=2G
SWT_MAX_CPU=2
SWT_MAX_PROCESSES=100
```

### 2.3 Checkpoint Setup
```bash
# Copy Episode 13475 checkpoint to deployment location
cp /path/to/episode_13475.pth checkpoints/

# Verify checkpoint integrity
python -c "
import torch
checkpoint = torch.load('checkpoints/episode_13475.pth', map_location='cpu')
print(f'Checkpoint loaded successfully')
print(f'Episode: {checkpoint.get(\"episode\", \"Unknown\")}')
print(f'Keys: {list(checkpoint.keys())}')
"
```

## 🐳 Step 3: Docker Deployment

### 3.1 Build Images
```bash
# Build production images
docker-compose build --no-cache

# Verify images
docker images | grep swt
```

### 3.2 Deploy Core Services
```bash
# Start core services (live trading + redis)
docker-compose up -d

# Check service status
docker-compose ps
docker-compose logs -f swt-live-trader
```

### 3.3 Deploy Monitoring Stack (Optional)
```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Start with full stack (monitoring + proxy)
docker-compose --profile monitoring --profile proxy up -d
```

## 🔍 Step 4: Verification & Testing

### 4.1 Health Checks
```bash
# Check container health
docker-compose ps

# Test health endpoints
curl -f http://localhost:8080/health
curl -f http://localhost:8080/metrics

# Check logs for errors
docker-compose logs --tail=50 swt-live-trader
```

### 4.2 Trading System Validation
```bash
# Run system validation
docker-compose exec swt-live-trader python -m pytest test_live_trading_system.py -v

# Run checkpoint validation
docker-compose exec swt-live-trader python scripts/validate_checkpoint.py \
    --checkpoint checkpoints/episode_13475.pth \
    --config config/live.yaml

# Run performance benchmark
docker-compose exec swt-live-trader python scripts/benchmark_system.py \
    --config config/live.yaml \
    --duration 60
```

### 4.3 Market Connectivity Test
```bash
# Test OANDA API connectivity
docker-compose exec swt-live-trader python -c "
import asyncio
from swt_live.data_feed import OANDADataFeed

async def test_connection():
    feed = OANDADataFeed()
    try:
        await feed.test_connection()
        print('✅ OANDA API connection successful')
    except Exception as e:
        print(f'❌ OANDA API connection failed: {e}')

asyncio.run(test_connection())
"
```

## 📊 Step 5: Monitoring Setup

### 5.1 Prometheus Configuration
Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'swt-live-trader'
    static_configs:
      - targets: ['swt-live-trader:8080']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 5.2 Grafana Dashboard Setup
```bash
# Copy dashboard configurations
cp -r monitoring/grafana/dashboards/* ./monitoring/grafana/dashboards/

# Import dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/swt-trading-dashboard.json
```

### 5.3 Alert Configuration
Create `monitoring/alert_rules.yml`:
```yaml
groups:
  - name: swt_trading_alerts
    rules:
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"

      - alert: TradingSystemDown
        expr: up{job="swt-live-trader"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SWT trading system is down"

      - alert: ExcessiveDrawdown
        expr: swt_daily_pnl < -1000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss exceeds threshold"
```

## 🔒 Step 6: Security Hardening

### 6.1 SSL/TLS Configuration
```bash
# Generate SSL certificates (using Let's Encrypt)
sudo apt install certbot python3-certbot-nginx

# Obtain certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates to application
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/
sudo chown swt-trader:swt-trader ssl/*
```

### 6.2 Network Security
```bash
# Configure fail2ban for SSH protection
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure intrusion detection
sudo apt install aide
sudo aideinit
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
```

### 6.3 Container Security
```bash
# Set resource limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2'

# Enable security options
security_opt:
  - no-new-privileges:true
  - apparmor:docker-default

# Use read-only filesystem where possible
read_only: true
tmpfs:
  - /tmp
  - /var/run
```

## 📈 Step 7: Production Readiness

### 7.1 Backup Strategy
```bash
# Create backup script
cat > scripts/backup_production.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/swt-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
tar -czf "$BACKUP_DIR/config.tar.gz" config/

# Backup logs
tar -czf "$BACKUP_DIR/logs.tar.gz" logs/

# Backup trading results
tar -czf "$BACKUP_DIR/results.tar.gz" results/

# Backup docker state
docker-compose config > "$BACKUP_DIR/docker-compose-state.yml"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x scripts/backup_production.sh
```

### 7.2 Log Rotation Setup
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/swt-trading << 'EOF'
/home/swt-trader/new_muzero/new_swt/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### 7.3 Automated Health Monitoring
```bash
# Create monitoring script
cat > scripts/health_monitor.sh << 'EOF'
#!/bin/bash
# Check system health and send alerts

HEALTH_URL="http://localhost:8080/health"
WEBHOOK_URL="your-slack-webhook-url"

if ! curl -sf "$HEALTH_URL" > /dev/null; then
    curl -X POST "$WEBHOOK_URL" \
         -H 'Content-Type: application/json' \
         -d '{"text":"🚨 SWT Trading System Health Check Failed"}'
fi
EOF

chmod +x scripts/health_monitor.sh

# Add to crontab
echo "*/5 * * * * /home/swt-trader/new_muzero/new_swt/scripts/health_monitor.sh" | crontab -
```

## 🔄 Step 8: Deployment Automation

### 8.1 Use Automated Deployment Script
```bash
# Run automated deployment
./scripts/deploy_production.sh

# The script will:
# 1. Validate environment
# 2. Build and deploy containers
# 3. Run health checks
# 4. Setup monitoring
# 5. Configure backups
```

### 8.2 Rollback Procedures
```bash
# Rollback to previous version
./scripts/deploy_production.sh --rollback

# Manual rollback
docker-compose down
docker-compose pull  # Get previous images
docker-compose up -d
```

## 📞 Step 9: Operations & Maintenance

### 9.1 Daily Operations
```bash
# Check system status
docker-compose ps
curl -s http://localhost:8080/health | jq .

# Review logs
docker-compose logs --tail=100 swt-live-trader

# Check trading performance
curl -s http://localhost:8080/metrics | grep swt_trades_total
```

### 9.2 Performance Monitoring
```bash
# System resources
docker stats

# Trading metrics
curl -s http://localhost:8080/metrics | grep -E "(trades|pnl|latency)"

# Error rates
docker-compose logs swt-live-trader | grep -i error | tail -10
```

### 9.3 Incident Response
1. **System Down**: Check container status, restart if needed
2. **High Memory**: Check for memory leaks, restart container
3. **Trading Errors**: Review logs, pause trading if necessary
4. **Network Issues**: Verify OANDA connectivity, check firewall

## ⚠️ Important Production Notes

### Critical Reminders
1. **Never commit credentials** to version control
2. **Test all changes** in staging environment first
3. **Monitor trading performance** continuously
4. **Maintain regular backups** of all critical data
5. **Keep system updated** with security patches

### Emergency Procedures
- **Emergency Stop**: `docker-compose down`
- **Force Restart**: `docker-compose restart swt-live-trader`
- **Clean Slate Deploy**: `docker-compose down -v && docker-compose up -d`

### Support Contacts
- **Technical Issues**: Check logs and troubleshooting guide
- **Trading Issues**: Review position reconciliation logs
- **Infrastructure**: Check system monitoring dashboards

---

**Next**: [Docker Guide](docker-guide.md) | [Security Guide](security.md)