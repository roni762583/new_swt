# Live Trading API Reference

## 📡 API Overview

The SWT Live Trading System exposes a RESTful API for monitoring, configuration, and control. All endpoints are available on port 8080 and return JSON responses.

## 🔗 Base URL

```
Production: https://your-domain.com/api/v1
Development: http://localhost:8080/api/v1
Health Check: http://localhost:8080/health
```

## 🔐 Authentication

API endpoints require Bearer token authentication for write operations. Read-only monitoring endpoints are accessible without authentication in development mode.

```bash
# Set authentication header
export API_TOKEN="your_api_token_here"
curl -H "Authorization: Bearer $API_TOKEN" http://localhost:8080/api/v1/status
```

## 📊 System Monitoring Endpoints

### GET /health
**Description**: System health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "memory_percent": 45.2,
  "cpu_percent": 15.8,
  "disk_percent": 32.1,
  "environment": "production",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "components": {
    "trading_engine": "healthy",
    "data_feed": "healthy",
    "position_reconciler": "healthy",
    "oanda_connection": "healthy",
    "redis_connection": "healthy"
  }
}
```

**Status Codes**:
- `200`: System healthy
- `503`: System unhealthy or degraded

**Example**:
```bash
curl -f http://localhost:8080/health
```

### GET /api/v1/status
**Description**: Detailed system status and trading metrics

**Response**:
```json
{
  "system": {
    "status": "running",
    "trading_active": true,
    "last_decision": "2025-01-15T10:29:30Z",
    "uptime": "2 hours 15 minutes",
    "version": "1.0.0"
  },
  "trading": {
    "current_session": {
      "start_time": "2025-01-15T08:00:00Z",
      "trades_today": 5,
      "daily_pnl": 125.50,
      "unrealized_pnl": 45.20,
      "win_rate": 0.80,
      "avg_trade_duration": "45 minutes"
    },
    "current_positions": [
      {
        "instrument": "GBP_JPY",
        "units": 1000,
        "side": "long",
        "entry_price": 195.245,
        "current_price": 195.301,
        "unrealized_pnl": 56.00,
        "entry_time": "2025-01-15T09:15:00Z"
      }
    ]
  },
  "performance": {
    "inference_latency": {
      "avg_ms": 245,
      "max_ms": 890,
      "p95_ms": 450
    },
    "data_feed": {
      "last_update": "2025-01-15T10:29:58Z",
      "updates_per_minute": 45,
      "connection_uptime": "99.8%"
    }
  }
}
```

### GET /api/v1/metrics
**Description**: Prometheus-compatible metrics endpoint

**Response**: Plain text Prometheus metrics format

**Example metrics**:
```
# HELP swt_trades_total Total number of trades executed
# TYPE swt_trades_total counter
swt_trades_total{instrument="GBP_JPY",side="long"} 23
swt_trades_total{instrument="GBP_JPY",side="short"} 18

# HELP swt_pnl_daily Daily profit/loss in account currency
# TYPE swt_pnl_daily gauge
swt_pnl_daily 125.50

# HELP swt_inference_duration_seconds Time spent on trading inference
# TYPE swt_inference_duration_seconds histogram
swt_inference_duration_seconds_bucket{le="0.1"} 45
swt_inference_duration_seconds_bucket{le="0.5"} 342
swt_inference_duration_seconds_bucket{le="1.0"} 456
swt_inference_duration_seconds_sum 234.5
swt_inference_duration_seconds_count 500

# HELP swt_position_count Current number of open positions
# TYPE swt_position_count gauge
swt_position_count{instrument="GBP_JPY"} 1

# HELP swt_connection_status Connection status (1=connected, 0=disconnected)
# TYPE swt_connection_status gauge
swt_connection_status{service="oanda"} 1
swt_connection_status{service="redis"} 1
```

## 📈 Trading Information Endpoints

### GET /api/v1/positions
**Description**: Current trading positions

**Response**:
```json
{
  "positions": [
    {
      "instrument": "GBP_JPY",
      "units": 1000,
      "side": "long",
      "entry_price": 195.245,
      "current_price": 195.301,
      "unrealized_pnl": 56.00,
      "realized_pnl": 0.00,
      "entry_time": "2025-01-15T09:15:00Z",
      "duration": "1 hour 14 minutes",
      "stop_loss": 195.095,
      "take_profit": 195.345
    }
  ],
  "summary": {
    "total_positions": 1,
    "total_unrealized_pnl": 56.00,
    "total_exposure": 195301.00,
    "margin_used": 3906.02
  }
}
```

### GET /api/v1/trades
**Description**: Recent trading history

**Query Parameters**:
- `limit` (optional): Number of trades to return (default: 50, max: 500)
- `since` (optional): ISO timestamp to filter trades after
- `instrument` (optional): Filter by trading instrument

**Response**:
```json
{
  "trades": [
    {
      "id": "trade_12345",
      "instrument": "GBP_JPY",
      "units": 1000,
      "side": "long",
      "entry_price": 195.123,
      "exit_price": 195.178,
      "entry_time": "2025-01-15T08:30:00Z",
      "exit_time": "2025-01-15T09:15:00Z",
      "duration": "45 minutes",
      "pnl": 55.00,
      "commission": -1.50,
      "net_pnl": 53.50,
      "confidence": 0.75,
      "exit_reason": "take_profit"
    }
  ],
  "pagination": {
    "total": 156,
    "returned": 50,
    "has_more": true
  },
  "summary": {
    "total_trades": 156,
    "winning_trades": 98,
    "losing_trades": 58,
    "win_rate": 0.628,
    "avg_win": 67.30,
    "avg_loss": -35.20,
    "total_pnl": 2567.80
  }
}
```

**Example**:
```bash
# Get last 10 trades
curl "http://localhost:8080/api/v1/trades?limit=10"

# Get trades since yesterday
curl "http://localhost:8080/api/v1/trades?since=2025-01-14T00:00:00Z"
```

### GET /api/v1/performance
**Description**: Trading performance analytics

**Query Parameters**:
- `period` (optional): Time period (1d, 7d, 30d, all) default: 1d

**Response**:
```json
{
  "period": "1d",
  "summary": {
    "total_trades": 8,
    "winning_trades": 6,
    "losing_trades": 2,
    "win_rate": 0.75,
    "profit_factor": 2.45,
    "sharpe_ratio": 1.82,
    "max_drawdown": -87.50,
    "total_pnl": 267.80,
    "avg_trade_duration": "52 minutes"
  },
  "daily_stats": [
    {
      "date": "2025-01-15",
      "trades": 8,
      "pnl": 267.80,
      "win_rate": 0.75,
      "max_drawdown": -45.20
    }
  ],
  "hourly_performance": {
    "best_hour": {"hour": 14, "avg_pnl": 35.60},
    "worst_hour": {"hour": 7, "avg_pnl": -12.30}
  }
}
```

## ⚙️ Configuration Endpoints

### GET /api/v1/config
**Description**: Current system configuration (sanitized)

**Response**:
```json
{
  "trading": {
    "instrument": "GBP_JPY",
    "position_size": 1000,
    "min_confidence": 0.6,
    "decision_frequency": 60
  },
  "risk": {
    "max_daily_loss": 500.0,
    "max_position_size": 5000,
    "max_concurrent_positions": 1
  },
  "agent": {
    "system": "stochastic_muzero",
    "mcts": {
      "num_simulations": 15,
      "c_puct": 1.25
    }
  },
  "system": {
    "environment": "production",
    "log_level": "INFO"
  }
}
```

### POST /api/v1/config
**Description**: Update runtime configuration

**Authentication**: Required

**Request Body**:
```json
{
  "trading.min_confidence": 0.65,
  "risk.max_daily_loss": 600.0
}
```

**Response**:
```json
{
  "status": "success",
  "updated": [
    "trading.min_confidence",
    "risk.max_daily_loss"
  ],
  "restart_required": false
}
```

**Example**:
```bash
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trading.min_confidence": 0.65}' \
  http://localhost:8080/api/v1/config
```

## 🎛️ Control Endpoints

### POST /api/v1/trading/pause
**Description**: Pause trading operations

**Authentication**: Required

**Request Body**:
```json
{
  "duration_seconds": 3600,  // Optional: auto-resume after duration
  "reason": "Manual intervention required"
}
```

**Response**:
```json
{
  "status": "success",
  "trading_paused": true,
  "paused_at": "2025-01-15T10:30:00Z",
  "auto_resume_at": "2025-01-15T11:30:00Z"
}
```

### POST /api/v1/trading/resume
**Description**: Resume trading operations

**Authentication**: Required

**Response**:
```json
{
  "status": "success",
  "trading_active": true,
  "resumed_at": "2025-01-15T10:35:00Z"
}
```

### POST /api/v1/positions/close
**Description**: Close specific or all positions

**Authentication**: Required

**Request Body**:
```json
{
  "instrument": "GBP_JPY",  // Optional: specific instrument
  "reason": "Manual close"   // Optional: reason for closure
}
```

**Response**:
```json
{
  "status": "success",
  "closed_positions": [
    {
      "instrument": "GBP_JPY",
      "units": 1000,
      "close_price": 195.234,
      "pnl": 45.60
    }
  ]
}
```

### POST /api/v1/system/restart
**Description**: Restart trading system components

**Authentication**: Required

**Request Body**:
```json
{
  "component": "data_feed",  // Options: data_feed, agent, all
  "force": false
}
```

**Response**:
```json
{
  "status": "success",
  "restarted_components": ["data_feed"],
  "restart_time": "2025-01-15T10:40:00Z"
}
```

## 📊 Market Data Endpoints

### GET /api/v1/market/current
**Description**: Current market data

**Response**:
```json
{
  "instrument": "GBP_JPY",
  "bid": 195.234,
  "ask": 195.237,
  "spread": 0.003,
  "timestamp": "2025-01-15T10:30:00Z",
  "change_24h": 0.15,
  "change_percent_24h": 0.077,
  "volatility": 0.012
}
```

### GET /api/v1/market/history
**Description**: Historical market data

**Query Parameters**:
- `instrument`: Trading instrument (required)
- `granularity`: Time granularity (M1, M5, H1, D) default: M1
- `count`: Number of candles (max: 500) default: 100

**Response**:
```json
{
  "instrument": "GBP_JPY",
  "granularity": "M1",
  "candles": [
    {
      "time": "2025-01-15T10:29:00Z",
      "bid": {"o": 195.230, "h": 195.245, "l": 195.225, "c": 195.234},
      "ask": {"o": 195.233, "h": 195.248, "l": 195.228, "c": 195.237},
      "volume": 1250
    }
  ]
}
```

## 🚨 Alert and Notification Endpoints

### GET /api/v1/alerts
**Description**: Current system alerts

**Response**:
```json
{
  "alerts": [
    {
      "id": "alert_001",
      "level": "warning",
      "type": "trading",
      "message": "Daily loss approaching limit: -450 / -500",
      "timestamp": "2025-01-15T10:25:00Z",
      "acknowledged": false
    }
  ],
  "counts": {
    "critical": 0,
    "warning": 1,
    "info": 0
  }
}
```

### POST /api/v1/alerts/{alert_id}/acknowledge
**Description**: Acknowledge an alert

**Authentication**: Required

**Response**:
```json
{
  "status": "success",
  "alert_id": "alert_001",
  "acknowledged_at": "2025-01-15T10:30:00Z"
}
```

## 🧪 Testing and Debug Endpoints

### POST /api/v1/debug/test-inference
**Description**: Test trading inference with current market data

**Authentication**: Required

**Response**:
```json
{
  "status": "success",
  "inference_result": {
    "action": "hold",
    "confidence": 0.45,
    "inference_time_ms": 234,
    "features": [0.2, 0.8, 0.1, 0.5, 0.6, 0.3, 0.4, 0.7, 0.9],
    "explanation": "Low confidence due to mixed technical signals"
  }
}
```

### GET /api/v1/debug/logs
**Description**: Recent log entries

**Query Parameters**:
- `level`: Log level filter (DEBUG, INFO, WARNING, ERROR)
- `lines`: Number of lines (default: 100, max: 1000)

**Response**:
```json
{
  "logs": [
    {
      "timestamp": "2025-01-15T10:30:00Z",
      "level": "INFO",
      "module": "trading_engine",
      "message": "Trading decision made: HOLD (confidence: 0.45)"
    }
  ]
}
```

## 📝 Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Detailed error description",
    "details": {
      "field": "min_confidence",
      "value": "invalid_value",
      "expected": "float between 0.0 and 1.0"
    }
  },
  "request_id": "req_12345",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Common Error Codes**:
- `400 BAD_REQUEST`: Invalid request parameters
- `401 UNAUTHORIZED`: Missing or invalid authentication
- `403 FORBIDDEN`: Insufficient permissions
- `404 NOT_FOUND`: Resource not found
- `409 CONFLICT`: Trading system in conflicting state
- `429 RATE_LIMITED`: Too many requests
- `500 INTERNAL_ERROR`: System error
- `503 SERVICE_UNAVAILABLE`: System temporarily unavailable

## 📖 Usage Examples

### Monitor Trading Session
```bash
#!/bin/bash
# Monitor current trading session

echo "=== System Status ==="
curl -s http://localhost:8080/api/v1/status | jq '.system, .trading.current_session'

echo -e "\n=== Current Positions ==="
curl -s http://localhost:8080/api/v1/positions | jq '.positions[]'

echo -e "\n=== Recent Performance ==="
curl -s http://localhost:8080/api/v1/performance?period=1d | jq '.summary'
```

### Emergency Trading Halt
```bash
#!/bin/bash
# Emergency stop all trading

echo "Pausing trading..."
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Emergency halt"}' \
  http://localhost:8080/api/v1/trading/pause

echo "Closing all positions..."
curl -X POST \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Emergency close"}' \
  http://localhost:8080/api/v1/positions/close
```

---

**Next**: [Monitoring API](monitoring-api.md) | [Configuration API](configuration-api.md)