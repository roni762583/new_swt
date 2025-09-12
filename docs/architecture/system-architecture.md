# SWT System Architecture

## 🏗️ Architecture Overview

The SWT Live Trading System is a production-grade, event-driven trading platform built on a microservices architecture with Episode 13475 MuZero agent integration.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWT Live Trading System                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Data Feed   │  │Event Trader │  │Trade Execute│  │Monitor  │ │
│  │             │  │             │  │             │  │         │ │
│  │ • OANDA API │─▶│ • Decision  │─▶│ • Order Mgmt│  │• Metrics│ │
│  │ • Streaming │  │ • MCTS      │  │ • Risk Mgmt │  │• Alerts │ │
│  │ • Buffering │  │ • Episode   │  │ • Execution │  │• Health │ │
│  │             │  │   13475     │  │             │  │         │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│         │                 │                 │            │     │
│         ▼                 ▼                 ▼            ▼     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Position Reconciler                           │ │
│  │        • Real-time sync • Discrepancy detection            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
          ┌─────────────────────────────────────────┐
          │            External Services            │
          │  ┌─────────┐ ┌─────────┐ ┌─────────────┐│
          │  │ OANDA   │ │ Redis   │ │ Prometheus  ││
          │  │ Broker  │ │ Cache   │ │ Monitoring  ││
          │  └─────────┘ └─────────┘ └─────────────┘│
          └─────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### 1. Market Data Ingestion
```
OANDA Streaming API → Data Feed → Event Buffer → Feature Processing
                                        ↓
                              Real-time Validation
                                        ↓
                            Historical Context Window
```

### 2. Trading Decision Flow
```
Market Data → Feature Engineering → WST Transform → MCTS Inference
     ↓              ↓                    ↓              ↓
Context Window → Normalization → Episode 13475 → Action/Confidence
                                        ↓
                              Trading Decision (BUY/SELL/HOLD)
```

### 3. Order Execution Flow
```
Trading Decision → Risk Validation → Position Sizing → Order Creation
       ↓                ↓                ↓               ↓
   Confidence → Daily Limits → Portfolio Mgmt → OANDA API
                                        ↓
                              Execution Confirmation
                                        ↓
                              Position Reconciliation
```

## 🏛️ Core Components

### 1. Data Feed Service (`swt_live/data_feed.py`)
**Responsibility**: Real-time market data streaming and preprocessing

**Key Features**:
- OANDA WebSocket streaming
- Connection resilience with exponential backoff
- Data validation and gap detection
- Event-driven callbacks

**Architecture**:
```python
class OANDADataFeed:
    async def start_streaming() -> None
    async def handle_price_update() -> None
    def register_callback() -> None
    def health_check() -> HealthStatus
```

### 2. Event Trader Service (`swt_live/event_trader.py`)
**Responsibility**: Main trading orchestration and decision engine

**Key Features**:
- Event-driven trading system
- Episode 13475 integration
- Frequency control (1 decision/minute)
- Complete trading lifecycle management

**Architecture**:
```python
class SWTEventTrader:
    async def handle_market_update() -> None
    async def make_trading_decision() -> TradingAction
    async def execute_trade() -> ExecutionResult
    def manage_session() -> SessionStatus
```

### 3. Trade Executor Service (`swt_live/trade_executor.py`)
**Responsibility**: Order execution and risk management

**Key Features**:
- Robust order execution with retry logic
- Multi-layer risk controls
- Slippage tracking and validation
- Execution reporting

**Architecture**:
```python
class SWTTradeExecutor:
    async def execute_order() -> ExecutionResult
    def validate_risk() -> RiskValidation
    def calculate_position_size() -> float
    def track_slippage() -> SlippageMetrics
```

### 4. Position Reconciler Service (`swt_live/position_reconciler.py`)
**Responsibility**: Real-time position synchronization

**Key Features**:
- Continuous position monitoring
- Discrepancy detection and resolution
- Broker state synchronization
- Reconciliation statistics

**Architecture**:
```python
class SWTPositionReconciler:
    async def reconcile_positions() -> ReconciliationResult
    def detect_discrepancies() -> List[Discrepancy]
    async def resolve_discrepancy() -> Resolution
    def generate_statistics() -> ReconciliationStats
```

### 5. Monitoring Service (`swt_live/monitoring.py`)
**Responsibility**: System health and performance monitoring

**Key Features**:
- Real-time metrics collection
- Configurable alerting
- Performance analysis
- Health status reporting

**Architecture**:
```python
class SWTLiveMonitor:
    def record_trade() -> None
    def check_thresholds() -> List[Alert]
    def generate_performance_report() -> PerformanceReport
    def get_system_health() -> SystemHealth
```

## 🧠 Episode 13475 Integration

### Agent Architecture
```
Episode 13475 Checkpoint
         ↓
   Model Loading
         ↓
   Configuration
   • MCTS simulations: 15
   • C_PUCT: 1.25
   • WST: J=2, Q=6
         ↓
   Feature Processing
   • 9D position features
   • WST backend: manual
   • Normalization: 0-1 scaling
         ↓
   MCTS Inference
         ↓
   Action Selection
```

### Feature Engineering Pipeline
```python
# 9-Dimensional Feature Vector
features = [
    position_type,      # 0=flat, 1=long, 2=short
    position_pnl,       # Normalized P&L
    position_duration,  # Bars since entry
    ma_short,          # Short-term moving average
    ma_long,           # Long-term moving average
    rsi,               # Relative Strength Index
    volatility,        # Price volatility
    trend_strength,    # Trend momentum
    market_state       # Market regime indicator
]
```

## 🔧 Infrastructure Architecture

### Container Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ swt-live-trader │ │  swt-training   │ │    swt-redis    │   │
│  │                 │ │                 │ │                 │   │
│  │ • Live Trading  │ │ • Model Train   │ │ • Caching       │   │
│  │ • Health Check  │ │ • GPU Support   │ │ • Pub/Sub       │   │
│  │ • Resource Mgmt │ │ • Checkpointing │ │ • Persistence   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   prometheus    │ │     grafana     │ │      nginx      │   │
│  │                 │ │                 │ │                 │   │
│  │ • Metrics       │ │ • Dashboards    │ │ • Load Balance  │   │
│  │ • Time Series   │ │ • Alerting      │ │ • SSL Term      │   │
│  │ • Retention     │ │ • Visualization │ │ • Rate Limiting │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Network Architecture
```
Internet
    │
    ▼
┌─────────┐    ┌─────────────────────────────────────┐
│  nginx  │───▶│           Internal Network          │
│ (proxy) │    │        (172.20.0.0/16)             │
└─────────┘    │                                     │
               │  ┌─────────┐  ┌─────────┐           │
               │  │  Live   │  │Training │           │
               │  │Trader   │  │System   │           │
               │  │:8080    │  │:8081    │           │
               │  └─────────┘  └─────────┘           │
               │       │            │                │
               │       ▼            ▼                │
               │  ┌─────────┐  ┌─────────┐           │
               │  │ Redis   │  │Monitor  │           │
               │  │ :6379   │  │ Stack   │           │
               │  └─────────┘  └─────────┘           │
               └─────────────────────────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │  OANDA Broker   │
                     │  External API   │
                     └─────────────────┘
```

## 📊 Performance Characteristics

### Latency Requirements
- **Market Data**: <100ms processing latency
- **Trading Decisions**: <500ms inference time
- **Order Execution**: <1s total execution time
- **Position Reconciliation**: <30s sync interval

### Throughput Specifications
- **Market Updates**: 1-10 Hz (depends on volatility)
- **Trading Frequency**: Max 1 decision/minute
- **Order Processing**: 1-5 orders/hour typical
- **Monitoring Updates**: 1 Hz metrics collection

### Resource Utilization
- **CPU**: 50-80% utilization during active trading
- **Memory**: 1-2GB baseline, 4GB peak
- **Network**: 1-10 Mbps data streaming
- **Storage**: 100MB/day logs, 1GB/month historical data

## 🔐 Security Architecture

### Authentication & Authorization
- **API Keys**: Environment-based credential management
- **Token Rotation**: Automatic token refresh for long-running sessions
- **Access Control**: Container-level isolation and least privilege

### Network Security
- **Encryption**: TLS 1.3 for all external communications
- **Firewall**: Container network isolation
- **VPN**: Optional VPN integration for enhanced security

### Data Protection
- **Encryption at Rest**: Sensitive configuration encrypted
- **Encryption in Transit**: All API communications over HTTPS/WSS
- **Audit Logging**: Comprehensive audit trail for all trading actions

## 🔄 Failure Modes & Recovery

### Automatic Recovery
1. **Connection Loss**: Exponential backoff reconnection
2. **API Errors**: Retry logic with circuit breakers
3. **Data Gaps**: Historical data backfill
4. **Position Drift**: Automatic reconciliation

### Manual Intervention Required
1. **Broker API Changes**: Configuration updates needed
2. **Model Corruption**: Checkpoint restoration required
3. **Systematic Errors**: Emergency stop and investigation
4. **Infrastructure Failure**: Disaster recovery procedures

---

**Next**: [Trading Engine Architecture](trading-engine.md)