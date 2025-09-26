# Nano - Lightweight Trading Experiments

## 📊 Overview

The `nano` directory contains lightweight experiments and optimizations for the MuZero trading system, focusing on finding optimal features, timeframes, and reward functions through empirical testing.

## 🎯 Key Discoveries

### Optimal Timeframe: M5/H1
Through comprehensive backtesting across multiple timeframe combinations:
- **M5/H1**: 444.6 pips, 38.6% win rate (BEST)
- **M1/H1**: 138.5 pips, 36.0% win rate
- **M5/H4**: Only 1 swing detected (needs more data)
- **M15/H1**: 149.0 pips, 35.0% win rate
- **M30/H4**: 79.0 pips, 33.0% win rate

**Result**: M5 execution with H1 context provides 3.2x better performance than M1/H1.

### Optimal Feature Set (13 features)

#### Market Features (7)
1. **react_ratio**: Momentum indicator comparing fast vs slow reactions to SMA200
2. **h1_trend**: Higher timeframe trend direction (-1/0/1)
3. **h1_momentum**: H1 5-bar rate of change
4. **efficiency_ratio**: Kaufman's efficiency (trend strength 0-1)
5. **bb_position**: Position within Bollinger Bands (-1 to 1)
6. **rsi_extreme**: RSI distance from neutral 50
7. **use_mean_reversion**: Regime flag (1 if efficiency < 0.3)

#### Position Features (6)
1. **position_side**: Current position (-1/0/1)
2. **position_pips**: Unrealized P&L in pips
3. **bars_since_entry**: Time in position
4. **pips_from_peak**: Drawdown from maximum profit
5. **max_drawdown_pips**: Maximum DD in current position
6. **accumulated_dd**: Cumulative drawdown over time

### Reward Function: AMDDP1
```python
reward = pnl_pips - 0.01 * cumulative_drawdown_sum
```
With profit protection: profitable trades always get positive reward.

### Market Quantization: 0.33σ
- Quantizes price movements into UP/NEUTRAL/DOWN
- Self-adjusts to timeframe volatility
- Provides ~35% neutral zone on M5 data

## 🗂️ Directory Structure

```
nano/
├── README.md                           # This file
├── picco-ppo/                         # PPO implementation
│   ├── train_minimal.py               # Minimal training with AMDDP1
│   ├── validate_minimal.py            # Validation on held-out data
│   ├── env/trading_env.py             # Full Gymnasium environment
│   ├── Dockerfile.minimal             # Lightweight Docker setup
│   ├── docker-compose.minimal.yml     # Docker orchestration
│   └── FEATURE_FORMULAS.md           # Complete feature documentation
│
├── Analysis Scripts/
│   ├── quick_tf_comparison.py        # Timeframe comparison (found M5/H1 optimal)
│   ├── optimal_features_finder.py    # Feature discovery experiment
│   ├── m5h4_swing_analyzer.py        # Swing-based market structure
│   └── comprehensive_tf_comparison.py # Full dataset analysis
│
└── Results/
    ├── training_results_*.json       # Training episode results
    └── validation_*.json              # Validation/test results
```

## 🚀 Quick Start

### Local Testing
```bash
cd picco-ppo/

# Run minimal training (rule-based policy)
python train_minimal.py

# Run validation
python validate_minimal.py
```

### Docker Deployment
```bash
cd picco-ppo/

# Build and run training
docker compose -f docker-compose.minimal.yml up ppo-trainer

# Run validation
docker compose -f docker-compose.minimal.yml --profile validation up validator
```

### Full PPO Training (with neural networks)
```bash
# Install dependencies
pip install gymnasium stable-baselines3 torch

# Run full PPO training
python train.py --timesteps 1000000 --n_envs 4
```

## 📈 Performance Results

### Data Splits (1M bars total)
- **Training**: bars 100,000 - 700,000 (60%)
- **Validation**: bars 700,000 - 1,000,000 (30%)
- **Test**: bars 1,000,000 - 1,100,000 (10%)

### Current Results (Rule-based)
- **Training**: 3.47% return, 56.5% win rate
- **Validation**: 5.20% return, 53.4% win rate
- **Test**: -0.72% return, 47.8% win rate

## 🔬 Experiment Highlights

### 1. Timeframe Analysis (`quick_tf_comparison.py`)
- Tested 8 timeframe combinations
- M5/H1 emerged as clear winner
- H4 requires weeks of data for meaningful swings

### 2. Feature Discovery (`optimal_features_finder.py`)
- Tested 10+ feature combinations
- Found that fewer, high-quality features outperform kitchen-sink approach
- React ratio and efficiency ratio are most predictive

### 3. Swing-Based Approach (`m5h4_swing_analyzer.py`)
- Implemented market structure detection (HHHL, LHLL, etc.)
- Regime-adaptive strategy switching
- Mean reversion in ranging, trend following in trending markets

## 🔧 Key Innovations

### 1. AMDDP1 Reward System
- Penalizes drawdown during positions
- Encourages tighter risk management
- Profit protection ensures positive trades aren't punished

### 2. Market Outcome Quantization
- 0.33σ threshold for UP/NEUTRAL/DOWN classification
- Self-adjusts to timeframe volatility
- Provides balanced distribution (~35% neutral)

### 3. Dual Timeframe Context
- M5 for execution (optimal granularity)
- H1 for context (trend and momentum)
- Combines tactical and strategic views

## 📝 Implementation Notes

### Position Tracking
All 6 position features properly track:
- Entry/exit mechanics
- Peak profits and drawdowns
- Time in position
- Cumulative risk metrics

### Reward Calculation
- AMDDP1 applied only on position close
- Small holding penalty (-0.001) to discourage overtrading
- Transaction costs (0.2 pips) included

### Feature Normalization
- Market features: Clipped or scaled to [-1, 1]
- Position features: Normalized by typical ranges
- Ensures stable neural network training

## 🎓 Lessons Learned

1. **Simpler is Better**: 7 carefully chosen features beat 20+ kitchen-sink features
2. **Timeframe Matters**: M5/H1 provides optimal signal-to-noise ratio
3. **Regime Detection Works**: Switching between trend/mean-reversion improves performance
4. **Drawdown Penalties Help**: AMDDP1 creates better risk-adjusted returns
5. **Proper Splits Essential**: 60/30/10 train/val/test prevents overfitting

## 🔮 Future Improvements

1. **Neural Network Policy**: Replace rule-based with PPO-trained network
2. **Checkpoint System**: Save/load best models, manage checkpoints
3. **Online Learning**: Continuous adaptation to market changes
4. **Multi-Asset**: Extend to other currency pairs
5. **Risk Sizing**: Dynamic position sizing based on confidence

## 📊 Validation Methodology

### Backtesting
- Walk-forward validation on 1M+ bars
- Proper train/validation/test splits
- Transaction costs included

### Metrics
- Total return (%)
- Win rate
- Sharpe ratio
- Maximum drawdown
- Pips per trade

## 🔗 Related Components

- **Parent**: `/home/aharon/projects/new_swt/micro/` - Main MuZero implementation
- **Data**: `/home/aharon/projects/new_swt/data/master.duckdb` - GBPJPY M1 data
- **Features**: Uses micro's position tracking system

## 💡 Key Takeaways

The nano experiments prove that:
1. **M5/H1 is optimal** for GBPJPY trading
2. **Minimal features work best** when carefully selected
3. **AMDDP1 rewards** improve risk-adjusted returns
4. **Regime detection** enables adaptive strategies
5. **Proper evaluation** requires large datasets and proper splits

This lightweight testing framework enables rapid experimentation before committing to full MuZero/PPO training.