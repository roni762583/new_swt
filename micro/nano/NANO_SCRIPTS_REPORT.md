# 📊 Nano Directory Scripts Report
*Generated: September 25, 2025*

## 🎯 Overview
The `/micro/nano/` directory contains specialized analysis and trading scripts for market microstructure analysis, swing state tracking, and strategy testing.

---

## 🔍 Core Analysis Scripts

### 1. **swing_state_tracker.py** ⭐
- **Purpose**: Tracks market structure using four swing states (HHHL, LHLL, HHLL, LHHL)
- **Key Feature**: Swing confirmation logic - lows only confirmed when price exceeds previous high
- **Output**: `swing_states.png`
- **Results**: 89.5% swing confirmation rate on test data, identifies market regimes

### 2. **mtf_swing_analyzer.py** ⭐
- **Purpose**: Multi-timeframe analysis using H1 for direction, M1 for timing
- **Key Feature**: Identifies aligned trading zones between timeframes
- **Output**: `mtf_swing_analysis.png`
- **Results**: Found 123 trading zones (53 bullish pullbacks, 16 bearish rallies)

---

## 📈 Trading Strategy Scripts

### 3. **test_trading_strategy.py**
- **Purpose**: Tests supply/demand zone trading strategy
- **Key Feature**: R:R filter (minimum 2.5), trades only with trend
- **Output**: `strategy_signals.csv`
- **Results**: Generates trade signals based on zone retests

### 4. **test_strategy_realistic.py**
- **Purpose**: Enhanced strategy testing with realistic OHLC generation
- **Key Feature**: Creates synthetic OHLC from close prices with volatility
- **Output**: `strategy_chart.png`, `best_signals.csv`
- **Results**: Tests multiple parameter sets to find optimal configuration

### 5. **test_strategy_real_ohlcv.py**
- **Purpose**: Strategy testing with actual OHLCV data
- **Key Feature**: Full backtesting with P&L calculation
- **Output**: `real_ohlcv_strategy.png`, `real_ohlcv_signals.csv`, `real_ohlcv_trades.csv`
- **Results**: Comprehensive performance metrics including Sharpe ratio

---

## 🔬 Feature Analysis Scripts

### 6. **feature_analysis.py**
- **Purpose**: Analyzes predictive power of various features
- **Key Feature**: Correlation analysis with forward returns
- **Output**: `feature_analysis_results.png`, `feature_analysis_results.csv`
- **Results**: Identifies top predictive features for trading

### 7. **talib_feature_analysis.py**
- **Purpose**: Technical indicator analysis using TA-Lib
- **Key Feature**: Tests 50+ technical indicators
- **Output**: `talib_analysis_results.csv`, `talib_best_indicators.csv`
- **Results**: Ranks indicators by predictive power

### 8. **pandas_ta_analysis.py**
- **Purpose**: Technical analysis using pandas-ta library
- **Key Feature**: Comprehensive indicator suite testing
- **Output**: `pandas_ta_results.csv`, `pandas_ta_best.csv`
- **Results**: Identifies optimal indicator combinations

### 9. **swt_feature_analysis.py**
- **Purpose**: Analyzes SWT (Stationary Wavelet Transform) features
- **Key Feature**: Multi-scale decomposition analysis
- **Output**: `swt_analysis_results.csv`
- **Results**: Evaluates wavelet features for prediction

---

## 📊 Specialized Studies

### 10. **breakout_touch_study.py**
- **Purpose**: Studies breakout and touch patterns
- **Key Feature**: Pattern recognition and statistical analysis
- **Output**: `breakout_touch_results.csv`
- **Results**: Quantifies breakout success rates

### 11. **zscore_tr_analysis.py**
- **Purpose**: Z-score and true range analysis
- **Key Feature**: Volatility-normalized indicators
- **Output**: `zscore_tr_results.csv`
- **Results**: Identifies extreme market conditions

### 12. **grid_strategy_test.py**
- **Purpose**: Tests grid trading strategies
- **Key Feature**: Multiple entry/exit levels
- **Results**: Evaluates grid performance in ranging markets

### 13. **simple_strategies_test.py**
- **Purpose**: Tests basic trading strategies
- **Key Feature**: Baseline strategy performance
- **Results**: Provides benchmark for complex strategies

### 14. **better_strategies_test.py**
- **Purpose**: Enhanced versions of simple strategies
- **Key Feature**: Improved entry/exit logic
- **Results**: Shows improvement over baseline

### 15. **analyze_slow_strategy.py**
- **Purpose**: Analyzes longer-term trading strategies
- **Key Feature**: Focus on higher timeframe signals
- **Results**: Evaluates position trading approaches

---

## 🛠️ Utility Scripts

### 16. **run_tests.py**
- **Purpose**: Test runner for all strategy scripts
- **Key Feature**: Automated testing pipeline

### 17. **check_columns.py**
- **Purpose**: Database schema verification
- **Key Feature**: Ensures data integrity

### 18. **Dockerfile**
- **Purpose**: Container configuration for nano environment
- **Key Feature**: Reproducible execution environment

---

## 📈 Key Findings & Results

### Market Structure Insights:
- **Swing States**: Market spends ~25% in each of the four states
- **Confirmation Rate**: 82-90% of potential swings get confirmed
- **State Transitions**: Average 4-5 state changes per 2000 bars

### Trading Performance:
- **Best R:R Zones**: Bullish pullbacks in uptrends (53 opportunities)
- **Alignment Success**: 31% of time has H1/M1 alignment
- **Optimal Parameters**: k=3 for M1, k=2 for H1 swing detection

### Technical Indicators:
- **Top Features**: Momentum, volatility, and wavelet transforms
- **Best Timeframes**: H1 for trend, M1 for timing
- **Signal Quality**: R:R > 2.5 filters improve win rate significantly

---

## 🎯 Recommended Workflow

1. **Start with**: `swing_state_tracker.py` - Understand market structure
2. **Then run**: `mtf_swing_analyzer.py` - Find aligned trading zones
3. **Test strategies**: Use `test_strategy_realistic.py` for initial testing
4. **Analyze features**: Run feature analysis scripts to find predictive indicators
5. **Backtest**: Use `test_strategy_real_ohlcv.py` for realistic performance

---

## 📝 Notes
- All scripts output to `/micro/nano/` directory
- Database connection: `../../data/master.duckdb` or `../../data/micro_features.duckdb`
- Most scripts use GBP/JPY forex data (price range 156-157)
- Docker container available for consistent execution environment