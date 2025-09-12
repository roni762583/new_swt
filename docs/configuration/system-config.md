# System Configuration Guide

## ⚙️ Configuration Overview

The SWT Live Trading System uses a hierarchical configuration approach combining YAML files, environment variables, and runtime parameters for maximum flexibility and security.

## 📁 Configuration Structure

```
config/
├── live.yaml              # Live trading configuration
├── training.yaml          # Training system configuration  
├── development.yaml       # Development environment
├── testing.yaml           # Testing configuration
├── production.env         # Production environment variables
└── secrets.env            # Encrypted secrets (not in git)
```

## 🔧 Core System Configuration

### Primary Configuration File (`config/live.yaml`)

```yaml
# SWT Live Trading System Configuration
# Episode 13475 Compatible

# ================================================
# TRADING SYSTEM CONFIGURATION
# ================================================

# Agent Configuration
agent:
  system: "stochastic_muzero"
  checkpoint_path: "/app/checkpoints/episode_13475.pth"
  
  # Episode 13475 Specific Parameters
  mcts:
    num_simulations: 15        # MCTS simulation count
    c_puct: 1.25              # PUCT exploration constant
    dirichlet_alpha: 0.25     # Root noise alpha
    exploration_fraction: 0.25 # Exploration noise fraction
  
  # Feature Processing
  features:
    window_size: 100          # Historical data window
    feature_dims: 9           # 9D feature vector
    normalization: "0-1"      # Normalization method
    
  # WST (Wavelet Scattering Transform) Parameters
  wst:
    J: 2                      # Wavelet scale parameter
    Q: 6                      # Wavelet quality factor
    backend: "manual"         # Use manual backend (no kymatio)

# ================================================
# TRADING CONFIGURATION
# ================================================

trading:
  # Instrument Settings
  instrument: "GBP_JPY"
  position_size: 1000         # Base position size in units
  
  # Decision Making
  min_confidence: 0.6         # Minimum confidence for trade execution
  decision_frequency: 60      # Seconds between trading decisions
  
  # Risk Management
  risk:
    max_daily_loss: 500.0     # Maximum daily loss in account currency
    max_position_size: 5000   # Maximum position size in units
    max_concurrent_positions: 1 # Maximum simultaneous positions
    max_daily_trades: 20      # Maximum trades per day
    stop_loss_pips: 50        # Stop loss in pips
    take_profit_pips: 100     # Take profit in pips
    
  # Position Management
  position:
    entry_timeout: 30         # Order entry timeout in seconds
    exit_timeout: 30          # Order exit timeout in seconds
    slippage_tolerance: 2     # Maximum acceptable slippage in pips
    
# ================================================
# MARKET DATA CONFIGURATION
# ================================================

market_data:
  # Data Source
  provider: "oanda"
  
  # Streaming Configuration
  stream:
    instruments: ["GBP_JPY"]
    heartbeat_interval: 5     # Heartbeat interval in seconds
    reconnect_delay: 5        # Initial reconnect delay
    max_reconnect_delay: 300  # Maximum reconnect delay
    reconnect_attempts: 10    # Maximum reconnect attempts
    
  # Data Validation
  validation:
    price_change_threshold: 0.1  # Maximum price change % to accept
    gap_tolerance: 300           # Maximum data gap in seconds
    duplicate_tolerance: 0       # Tolerance for duplicate data points

# ================================================
# OANDA BROKER CONFIGURATION
# ================================================

oanda:
  # Environment (set via environment variable)
  environment: "${SWT_OANDA_ENVIRONMENT}"  # "practice" or "live"
  
  # API Configuration
  api:
    timeout: 10               # API request timeout in seconds
    retry_attempts: 3         # Number of retry attempts
    retry_delay: 1            # Delay between retries
    rate_limit: 100           # Requests per second limit
    
  # Streaming Configuration
  stream:
    timeout: 30               # Stream timeout in seconds
    buffer_size: 1000         # Price buffer size
    
# ================================================
# SYSTEM CONFIGURATION
# ================================================

system:
  # Logging
  logging:
    level: "${SWT_LOG_LEVEL}"  # DEBUG, INFO, WARNING, ERROR
    format: "json"             # "json" or "text"
    file: "/app/logs/swt_live.log"
    max_size: "100MB"
    backup_count: 5
    
  # Performance
  performance:
    inference_timeout: 1.0    # Maximum inference time in seconds
    batch_processing: false   # Disable batch processing for live trading
    memory_limit: "2GB"       # Memory usage limit
    
  # Health Monitoring
  health:
    check_interval: 60        # Health check interval in seconds
    metrics_port: 8080        # Metrics and health endpoint port
    
# ================================================
# REDIS CONFIGURATION
# ================================================

redis:
  url: "${REDIS_URL}"
  database: 0
  connection_pool:
    max_connections: 20
    retry_on_timeout: true
    socket_timeout: 5
    socket_connect_timeout: 5
    
# ================================================
# MONITORING CONFIGURATION
# ================================================

monitoring:
  # Metrics Collection
  metrics:
    enabled: true
    collection_interval: 10   # Seconds between metric collections
    retention_days: 30        # Days to retain detailed metrics
    
  # Alerting
  alerts:
    enabled: true
    
    # Trading Alerts
    trading:
      consecutive_losses: 5   # Alert after N consecutive losses
      daily_loss_threshold: 300  # Alert threshold for daily loss
      position_age_hours: 24  # Alert if position open > N hours
      
    # System Alerts  
    system:
      memory_usage_threshold: 80     # Memory usage % threshold
      cpu_usage_threshold: 80       # CPU usage % threshold
      disk_usage_threshold: 85      # Disk usage % threshold
      response_time_threshold: 5.0  # Response time threshold in seconds
      
  # Performance Tracking
  performance:
    track_latency: true       # Track system latency
    track_throughput: true    # Track data throughput
    track_accuracy: true      # Track prediction accuracy
    
# ================================================
# FEATURE ENGINEERING
# ================================================

features:
  # Price Features
  price:
    lookback_periods: [5, 10, 20, 50]  # Moving average periods
    volatility_window: 20               # Volatility calculation window
    
  # Technical Indicators
  indicators:
    rsi:
      period: 14
      overbought: 70
      oversold: 30
      
    macd:
      fast_period: 12
      slow_period: 26
      signal_period: 9
      
    bollinger:
      period: 20
      std_dev: 2
      
  # Position Features (9-dimensional vector)
  position_features:
    - "position_type"        # 0=flat, 1=long, 2=short
    - "position_pnl"         # Normalized P&L
    - "position_duration"    # Bars since entry
    - "ma_short"            # Short-term moving average
    - "ma_long"             # Long-term moving average
    - "rsi"                 # Relative Strength Index
    - "volatility"          # Price volatility
    - "trend_strength"      # Trend momentum
    - "market_state"        # Market regime indicator
    
# ================================================
# SECURITY CONFIGURATION
# ================================================

security:
  # API Security
  api:
    require_auth: true
    rate_limiting: true
    max_requests_per_minute: 60
    
  # Data Protection
  data:
    encrypt_sensitive: true
    audit_logging: true
    
  # Container Security
  container:
    run_as_non_root: true
    read_only_filesystem: false  # Need write access for logs
    drop_capabilities: ["ALL"]
    add_capabilities: ["NET_BIND_SERVICE"]

# ================================================
# DEVELOPMENT OVERRIDES
# ================================================

# The following section is only used in development
# In production, these should be set via environment variables

development:
  # Mock Trading (for testing)
  mock_trading: false
  
  # Debug Features
  debug:
    save_inference_data: false
    detailed_logging: false
    profile_performance: false
    
  # Testing
  testing:
    fast_mode: false
    mock_oanda: false
    deterministic_seed: null
```

## 🌍 Environment Variables

### Core Environment Variables (`config/production.env`)

```bash
# ================================================
# OANDA BROKER CREDENTIALS
# ================================================
# CRITICAL: Never commit these to version control
SWT_OANDA_ACCOUNT_ID=your_account_id_here
SWT_OANDA_API_TOKEN=your_api_token_here
SWT_OANDA_ENVIRONMENT=practice  # or "live" for production

# ================================================
# SYSTEM ENVIRONMENT
# ================================================
SWT_ENVIRONMENT=production
SWT_LOG_LEVEL=INFO
SWT_CONFIG_FILE=/app/config/live.yaml

# ================================================
# CHECKPOINT AND MODEL
# ================================================
SWT_CHECKPOINT_PATH=/app/checkpoints/episode_13475.pth
SWT_AGENT_SYSTEM=stochastic_muzero

# ================================================
# TRADING PARAMETERS
# ================================================
SWT_INSTRUMENT=GBP_JPY
SWT_POSITION_SIZE=1000
SWT_MIN_CONFIDENCE=0.6
SWT_MAX_DAILY_LOSS=500.0

# ================================================
# REDIS CONFIGURATION
# ================================================
REDIS_URL=redis://redis:6379/0

# ================================================
# RESOURCE LIMITS
# ================================================
SWT_MAX_MEMORY=2G
SWT_MAX_CPU=2
SWT_MAX_PROCESSES=100
SWT_MAX_FILES=1024

# ================================================
# MONITORING
# ================================================
SWT_METRICS_PORT=8080
SWT_HEALTH_CHECK_INTERVAL=60

# ================================================
# DEVELOPMENT FLAGS (Production: all false)
# ================================================
SWT_DEBUG_MODE=false
SWT_MOCK_TRADING=false
SWT_SAVE_DEBUG_DATA=false
```

### Environment-Specific Variables

**Development**:
```bash
SWT_ENVIRONMENT=development
SWT_LOG_LEVEL=DEBUG
SWT_OANDA_ENVIRONMENT=practice
SWT_MOCK_TRADING=true
SWT_DEBUG_MODE=true
```

**Testing**:
```bash
SWT_ENVIRONMENT=testing
SWT_LOG_LEVEL=DEBUG
SWT_MOCK_TRADING=true
SWT_FAST_MODE=true
SWT_DETERMINISTIC_SEED=42
```

**Production**:
```bash
SWT_ENVIRONMENT=production
SWT_LOG_LEVEL=INFO
SWT_OANDA_ENVIRONMENT=live
SWT_MOCK_TRADING=false
SWT_DEBUG_MODE=false
```

## 🔧 Configuration Management

### 1. Configuration Loading Priority

The system loads configuration in the following order (later overrides earlier):

1. **Default values** (hardcoded in application)
2. **YAML configuration file** (`config/live.yaml`)
3. **Environment variables** (`.env` files and system env)
4. **Command-line arguments** (highest priority)

### 2. Environment Variable Interpolation

YAML files can reference environment variables:

```yaml
# Direct reference
oanda:
  environment: "${SWT_OANDA_ENVIRONMENT}"

# With default value
logging:
  level: "${SWT_LOG_LEVEL:-INFO}"

# Nested references
database:
  url: "${DATABASE_URL:-redis://localhost:6379/0}"
```

### 3. Configuration Validation

The system validates configuration on startup:

```python
# Configuration validation example
from swt_live.config import ConfigValidator

validator = ConfigValidator()
errors = validator.validate_config(config)
if errors:
    raise ConfigurationError(f"Invalid configuration: {errors}")
```

## 📊 Configuration Templates

### Development Configuration (`config/development.yaml`)

```yaml
# Extends live.yaml with development overrides
extends: "live.yaml"

# Development-specific overrides
trading:
  mock_trading: true
  min_confidence: 0.3  # Lower threshold for testing

oanda:
  environment: "practice"

system:
  logging:
    level: "DEBUG"
    
development:
  debug:
    save_inference_data: true
    detailed_logging: true
    profile_performance: true
```

### Testing Configuration (`config/testing.yaml`)

```yaml
# Testing configuration with mocked components
extends: "live.yaml"

trading:
  mock_trading: true
  fast_mode: true
  
system:
  logging:
    level: "DEBUG"
    
testing:
  deterministic_seed: 42
  mock_oanda: true
  fast_inference: true
```

## 🔒 Secrets Management

### 1. Environment-based Secrets

Store sensitive configuration in environment variables:

```bash
# Load from encrypted secrets file
set -a  # Export all variables
source config/secrets.env.encrypted
set +a
```

### 2. Docker Secrets

For Docker Swarm deployments:

```yaml
# docker-compose.yml
services:
  swt-live-trader:
    secrets:
      - oanda_token
      - oanda_account
    environment:
      - SWT_OANDA_API_TOKEN_FILE=/run/secrets/oanda_token

secrets:
  oanda_token:
    external: true
  oanda_account:
    external: true
```

### 3. External Secret Managers

Integration with external secret management:

```yaml
# config/live.yaml
secrets:
  provider: "vault"  # or "aws_secrets", "azure_keyvault"
  vault:
    url: "https://vault.company.com"
    auth_method: "kubernetes"
    path: "secret/swt-trading"
```

## 🔄 Configuration Updates

### 1. Runtime Configuration Changes

Some parameters can be updated at runtime:

```bash
# Update via API
curl -X POST http://localhost:8080/config \
  -H "Content-Type: application/json" \
  -d '{"trading.min_confidence": 0.7}'

# Update via environment variable
docker-compose exec swt-live-trader \
  env SWT_MIN_CONFIDENCE=0.7 python -c "import os; os.kill(1, signal.SIGHUP)"
```

### 2. Configuration Backup

Backup configuration before changes:

```bash
# Backup current configuration
cp config/live.yaml config/live.yaml.backup.$(date +%Y%m%d-%H%M%S)

# Restore from backup
cp config/live.yaml.backup.20250101-120000 config/live.yaml
```

## 📋 Configuration Checklist

### Pre-Production Checklist

- [ ] All environment variables set correctly
- [ ] OANDA credentials configured and tested
- [ ] Episode 13475 checkpoint available and validated
- [ ] Risk management parameters appropriate for account size
- [ ] Logging level set to INFO (not DEBUG)
- [ ] Mock trading disabled
- [ ] Resource limits configured
- [ ] Monitoring and alerting enabled
- [ ] Backup procedures configured

### Production Validation

```bash
# Validate configuration
python -c "
from swt_live.config import load_config
config = load_config('config/live.yaml')
print('✅ Configuration loaded successfully')
print(f'Environment: {config.system.environment}')
print(f'Trading instrument: {config.trading.instrument}')
print(f'OANDA environment: {config.oanda.environment}')
"

# Test OANDA connectivity
python -c "
import asyncio
from swt_live.brokers.oanda_client import OANDAClient
async def test():
    client = OANDAClient()
    account = await client.get_account_info()
    print(f'✅ Account connected: {account.id}')
asyncio.run(test())
"
```

---

**Next**: [Trading Configuration](trading-config.md) | [OANDA Integration](oanda-config.md)