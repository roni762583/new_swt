# SWT Live Trading System Documentation

## 📚 Documentation Overview

This documentation provides comprehensive guides for deploying, configuring, and maintaining the SWT (Stochastic Wavelet Transform) Live Trading System with Episode 13475 compatibility.

## 📖 Documentation Structure

### 🏗️ [Architecture](architecture/)
- **[System Architecture](architecture/system-architecture.md)** - Complete system design and component overview
- **[Trading Engine](architecture/trading-engine.md)** - Core trading logic and decision flow
- **[Data Pipeline](architecture/data-pipeline.md)** - Market data processing and feature engineering
- **[Infrastructure](architecture/infrastructure.md)** - Containerization and service orchestration

### 🚀 [Deployment](deployment/)
- **[Production Deployment](deployment/production-deployment.md)** - Complete production setup guide
- **[Docker Guide](deployment/docker-guide.md)** - Container deployment and management
- **[Environment Setup](deployment/environment-setup.md)** - Development and staging environments
- **[Security](deployment/security.md)** - Security hardening and best practices

### ⚙️ [Configuration](configuration/)
- **[System Configuration](configuration/system-config.md)** - Core system parameters
- **[Trading Parameters](configuration/trading-config.md)** - Trading strategy configuration
- **[OANDA Integration](configuration/oanda-config.md)** - Broker API configuration
- **[Monitoring Setup](configuration/monitoring-config.md)** - Metrics and alerting configuration

### 🔧 [Troubleshooting](troubleshooting/)
- **[Common Issues](troubleshooting/common-issues.md)** - Frequently encountered problems and solutions
- **[Performance](troubleshooting/performance.md)** - Performance optimization and tuning
- **[Debugging](troubleshooting/debugging.md)** - Debug tools and techniques
- **[Log Analysis](troubleshooting/log-analysis.md)** - Log interpretation and analysis

### 📡 [API Reference](api/)
- **[Live Trading API](api/live-trading-api.md)** - Live trading system API
- **[Monitoring API](api/monitoring-api.md)** - System monitoring endpoints
- **[Configuration API](api/configuration-api.md)** - Runtime configuration management

### 📊 [Validation](validation/)
- **[Validation Framework](../swt_validation/README.md)** - Comprehensive validation system
- **[Composite Scoring](validation/composite-scoring.md)** - Multi-factor performance assessment
- **[Monte Carlo CAR25](validation/monte-carlo.md)** - Conservative return estimation
- **[Walk-Forward Analysis](validation/walk-forward.md)** - Overfitting detection
- **[Automated Validation](validation/automated.md)** - Training integration

## 🎯 Quick Start

### Production Deployment
```bash
# 1. Clone and setup
cd /path/to/new_swt
cp config/live.yaml.example config/live.yaml

# 2. Configure credentials
export SWT_OANDA_ACCOUNT_ID="your_account_id"
export SWT_OANDA_API_TOKEN="your_api_token"

# 3. Validate checkpoint (recommended)
python validate_episode_13475.py \
  --checkpoint checkpoints/episode_13475.pth \
  --data data/GBPJPY_M1_202201-202508.csv

# 4. Deploy (if validation passes)
./scripts/deploy_production.sh
```

### Development Environment
```bash
# 1. Build and run locally
docker-compose up -d

# 2. Run validation tests
python swt_validation/composite_scorer.py  # Test scoring system
python swt_validation/monte_carlo_car25.py --checkpoint checkpoints/test.pth --data data/test.csv --runs 100

# 3. Run integration tests
docker-compose exec swt-live-trader python -m pytest test_live_trading_system.py

# 4. View logs
docker-compose logs -f swt-live-trader
```

## ✅ Validation Requirements

### Pre-Deployment Validation
Before deploying any checkpoint to production, ensure it passes:

1. **Composite Score**: ≥70/100
2. **CAR25**: ≥15% annual return
3. **Max Drawdown**: ≤25%
4. **Win Rate**: ≥40%
5. **Robustness Score**: ≥50%

Run the comprehensive validation:
```bash
python validate_episode_13475.py --checkpoint your_checkpoint.pth --data your_data.csv
```

### Continuous Validation
During training, enable automated validation:
```bash
python training_main.py --enable-validation --validation-data data/test.csv
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2 cores (Intel/AMD x64)
- **Memory**: 2GB RAM
- **Storage**: 10GB available space
- **Network**: Stable internet connection (1Mbps+)
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ with WSL2

### Recommended for Production
- **CPU**: 4+ cores (Intel/AMD x64)
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ SSD
- **Network**: Redundant internet connections
- **OS**: Ubuntu 22.04 LTS or CentOS 8+

## 🛡️ Security Considerations

- **API Keys**: Never commit credentials to version control
- **Network**: Use secure connections (HTTPS/WSS) for all external APIs
- **Containers**: Run with non-root users and resource limits
- **Monitoring**: Enable comprehensive logging and alerting
- **Backups**: Regular backup of configuration and trading data

## 📞 Support and Resources

### Getting Help
1. **Documentation**: Check relevant section in this documentation
2. **Troubleshooting**: Review [common issues](troubleshooting/common-issues.md)
3. **Logs**: Analyze system logs using [log analysis guide](troubleshooting/log-analysis.md)
4. **Performance**: Use [benchmarking tools](../scripts/benchmark_system.py)

### Resources
- **Episode 13475 Checkpoint**: Required for live trading operations
- **Market Data**: OANDA API provides real-time market data
- **System Monitoring**: Prometheus + Grafana stack for observability
- **Testing**: Comprehensive test suite for validation

## 📋 Maintenance Schedule

### Daily
- Monitor system health and trading performance
- Review trading logs for anomalies
- Verify market data connectivity

### Weekly
- Review system resource usage
- Analyze trading performance metrics
- Update monitoring dashboards

### Monthly
- System backup and recovery testing
- Security audit and updates
- Performance optimization review

---

**⚠️ Important**: This is a production-grade financial trading system. Always test thoroughly in a practice environment before deploying to live trading.