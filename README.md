# 🚀 PPO Trading System - GBPJPY Forex Implementation
**Version 3.2 | Last Updated: September 29, 2025 | Database Optimized**

## 🔴 MAJOR DATABASE CLEANUP & ENHANCEMENT (September 29, 2025)

### Database Consolidation & Optimization
All databases now consolidated in `/micro/nano/picco-ppo/`:

#### 📊 Database Structure
1. **master.duckdb** (168 MB) - M1 OHLCV data with swing detection
   - 1,333,912 rows of GBPJPY M1 data (2022-2025)
   - 11 columns including 4 new swing detection columns
   - Swing Statistics:
     - M1 swing highs: 274,699 (20.59%)
     - M1 swing lows: 274,264 (20.56%)
     - H1 swing highs: 22,394 (1.68%)
     - H1 swing lows: 22,394 (1.68%)

2. **precomputed_features.duckdb** (15 MB) - M5 ML features
   - 99,950 rows with 22 engineered features
   - Ready for ML training pipelines

### Recent Improvements
- ✅ **Deleted 323 redundant columns** (c_* and wst_*) reducing DB size by 90%
- ✅ **Restored clean OHLCV structure** with proper 'close' column
- ✅ **Implemented HalfTrend indicator** (tested and documented)
- ✅ **Added swing point detection** for M1 and H1 timeframes
- ✅ **New swing query functions** for dynamic SL/TP calculation:
  - `get_last_two_swings()` - Returns last 2 swing highs/lows for each timeframe
  - `get_swing_levels_for_trading()` - Returns formatted levels for trading

### Swing Detection Features
New columns added to master table:
- `swing_high_m1` / `swing_low_m1`: 3-bar pattern swing detection
- `swing_high_h1` / `swing_low_h1`: Hourly swing points from M1 data

## 🎯 ACTIVE SYSTEM: PPO (Proximal Policy Optimization)
**Location**: `/micro/nano/picco-ppo/` | **Status**: Ready for Training

### Current Configuration (v3.2)
- **Architecture**: PPO with 128×128 MLP
- **Features**: 17 carefully selected (7 market + 6 position + 4 time)
- **Reward**: AMDDP1 (pips - 0.01×drawdown)
- **Trading Costs**: 4 pip fixed spread
- **Database**: Clean, optimized with swing detection

### Training Approach
1. **Phase 1**: Learn from winners only (first 1000 profitable trades)
2. **Phase 2**: Weighted learning (winners: 1.0, losers: 0.2→1.0)
3. **Phase 3**: Normal learning from all trades

## 📊 System Overview

A **production-grade PPO implementation** for forex trading (GBPJPY M5/H1) using clean, optimized data with swing-based stop loss and take profit levels.

### Key Features
- **Clean Database**: Optimized from 2.8GB to 168MB with essential data
- **Swing Detection**: Automatic identification of support/resistance levels
- **Feature Engineering**: 22 pre-computed M5 features + swing levels
- **Modular Scripts**: Consolidated in picco-ppo directory
- **Docker Ready**: Container-based training environment

## 🛠️ Project Structure

```
new_swt/
├── micro/
│   └── nano/
│       └── picco-ppo/           # Main PPO implementation
│           ├── master.duckdb            # M1 OHLCV + swing data (168MB)
│           ├── precomputed_features.duckdb  # M5 ML features (15MB)
│           ├── db-state.txt             # Database documentation
│           ├── add_swing_points.py      # Swing detection script
│           ├── trading_env_4action.py   # Trading environment
│           ├── train_ppo_weighted.py    # PPO training script
│           └── ppo_agent.py            # PPO agent implementation
└── README.md
```

## 🚀 Quick Start

### 1. Verify Database Integrity
```bash
cd micro/nano/picco-ppo
python3 -c "import duckdb; conn = duckdb.connect('master.duckdb'); print(conn.execute('SELECT COUNT(*) FROM master').fetchone())"
# Should output: (1333912,)
```

### 2. Check Swing Detection
```bash
python3 -c "from add_swing_points import get_swing_levels_for_trading; import duckdb; conn = duckdb.connect('master.duckdb'); print(get_swing_levels_for_trading(conn))"
```

### 3. Run PPO Training
```bash
docker compose up -d --build
docker logs -f ppo-training
```

## 📈 Database Details

### Master Table Schema (11 columns)
1. `timestamp` - Bar timestamp (PRIMARY KEY)
2. `bar_index` - Sequential bar number
3. `open`, `high`, `low`, `close` - OHLCV data
4. `volume` - Trading volume
5. `swing_high_m1`, `swing_low_m1` - M1 swing points
6. `swing_high_h1`, `swing_low_h1` - H1 swing points

### M5 Features Table (22 columns)
Technical indicators including SMA, RSI, ATR, Bollinger Bands, efficiency ratio, and H1 trend/momentum features.

## 🔄 Recent Updates (September 29, 2025)

1. **Database Cleanup**: Removed 323 redundant columns, saving 2.6GB
2. **HalfTrend Implementation**: Created and tested, then removed (kept swing detection)
3. **Swing Detection**: Added automatic support/resistance identification
4. **Script Consolidation**: All scripts now in picco-ppo directory
5. **Documentation**: Complete db-state.txt with formulas and statistics

## 📝 Notes

- All timestamps are in broker timezone (likely GMT+2/3)
- Swing detection uses 3-bar pattern for M1, hourly aggregation for H1
- Database optimized for fast queries with proper indexing
- Ready for production trading with clean, verified data

## 🏁 Next Steps

1. Run full PPO training to 1M timesteps
2. Backtest with swing-based SL/TP levels
3. Deploy to live trading with proper risk management
4. Monitor and adjust based on performance metrics

---

**Contact**: roni762583@protonmail.com
**Last Database Update**: September 29, 2025
**Status**: Production Ready