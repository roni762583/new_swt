# PPO M5/H1 Trading System

Proximal Policy Optimization implementation for the optimal M5/H1 trading strategy discovered through comprehensive timeframe analysis.

**Latest Updates (Sept 28, 2025)**:
- ✅ Van Tharp Expectancy_R rating system
- ✅ Rolling expectancy tracking (100/500/1000 trade windows)
- ✅ AMDDP1 reward function (pips-based with drawdown penalty)
- ✅ 4 pip spread cost on position opening
- ✅ Cyclic time features (sin/cos hour of day/week)
- ✅ Real PPO with neural networks (stable-baselines3)
- ✅ Checkpoint management (keep 2 + best)
- ✅ Docker BuildKit optimization
- ✅ 1M+ bars support (60/30/10 splits)
- ✅ Live training showing 17,400+ pips profit

**NEW - Optimized Data Loading & Winner-Focused Learning (Sept 28)**:
- 🔥 **Precomputed Features Database**: All technical indicators computed once and stored in DuckDB
  - No recalculation on every episode
  - No lost bars from indicator initialization
  - Efficient slicing - only load what's needed per episode
- 🎯 **Winner-Focused Learning Strategy**:
  - Removed curriculum learning - full 4 pip spread from start
  - Phase 1: Learn only from profitable trades (ignore losses) until 1000 wins
  - Phase 2: Normal learning with both profits and losses
- ⚡ **Performance Optimizations**:
  - Multi-environment support restored with precomputed features
  - Optimized H1-to-M5 mapping using pandas merge_asof
  - Minimal dependencies option saves 2.6GB
- 📊 **Enhanced Monitoring**:
  - Rolling expectancy tracking (100/500/1000 trade windows)
  - Profitable trade counter
  - Learning phase indicator
- 🔧 **Centralized Configuration System**:
  - All settings now in `config.py` (no more hardcoded values!)
  - Instrument configs: pip values, spread, trade size
  - Training params: timesteps, environments, checkpoint frequency (10k)
  - PPO hyperparameters: learning rates, batch sizes, network architecture
  - Fixed GBPJPY pip calculations (0.01 instead of 0.0001)

## 📊 Key Results from Analysis

- **Best Timeframe**: M5 execution with H1 context
- **Historical Performance**: 444.6 pips, 38.6% win rate
- **Optimal Features**: 7 market + 6 position + 4 time = 17 total
- **Reward Function**: AMDDP1 = pnl_pips - 0.01 * cumulative_drawdown
- **Trading Costs**: 4 pip spread on position opening
- **Performance Rating**: Van Tharp Expectancy_R system

## 🏗️ Architecture

### Environment (`env/trading_env.py`)
- **State Space (17 features)**:
  - 7 Market Features:
    - `react_ratio`: (close - SMA200)/(SMA20 - SMA200), clipped [-5, 5]
    - `h1_trend`: Higher timeframe direction (-1, 0, 1)
    - `h1_momentum`: H1 5-bar rate of change
    - `efficiency_ratio`: Kaufman's efficiency (0-1)
    - `bb_position`: Position in Bollinger Bands (-1 to 1)
    - `rsi_extreme`: (RSI - 50)/50, range [-1, 1]
    - `use_mean_reversion`: 1 if efficiency < 0.3, else 0
  - 6 Position Features (from micro setup):
    - `position_side`: -1 (short), 0 (flat), 1 (long)
    - `position_pips`: Current unrealized P&L in pips
    - `bars_since_entry`: Time in position
    - `pips_from_peak`: Distance from maximum profit
    - `max_drawdown_pips`: Max DD in current position
    - `accumulated_dd`: Cumulative drawdown over time
  - 4 Time Features (cyclic encoding):
    - `sin_hour_day`: sin(2π * hour / 24)
    - `cos_hour_day`: cos(2π * hour / 24)
    - `sin_hour_week`: sin(2π * hour / 120) - 120hr trading week
    - `cos_hour_week`: cos(2π * hour / 120) - Sun 5pm to Fri 5pm EST

- **Action Space**:
  - 0: Hold
  - 1: Buy
  - 2: Sell
  - 3: Close

### PPO Configuration
- **Network Architecture**: [256, 256] MLP with ReLU activation
- **Learning Rate**: 3e-4 (adaptive)
- **Batch Size**: 64 minibatch
- **N Steps**: 2048 (rollout buffer)
- **Gamma**: 0.99 (discount factor)
- **GAE Lambda**: 0.95 (advantage estimation)
- **Clip Range**: 0.2 (PPO clipping)
- **Entropy Coef**: 0.01 (exploration)
- **Training Data**: 600k bars (60% of dataset)
- **Validation Data**: 300k bars (30% of dataset)

## ⚙️ Configuration Management

All settings are centralized in `config.py`:

```python
# config.py structure
INSTRUMENTS = {
    "GBPJPY": {
        "pip_value": 0.01,      # Correct for JPY pairs
        "spread": 4.0,          # Full spread from start
        "trade_size": 1000.0
    }
}

TRAINING = {
    "total_timesteps": 1_000_000,
    "n_envs": 4,
    "save_freq": 10_000,     # Checkpoint every 10k steps
    "episode_length": 2000,  # M5 bars per episode
    "initial_balance": 10000.0
}

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    # ... all PPO hyperparameters
}
```

To modify settings, edit `config.py` - no need to change code!

## 🚀 Quick Start

### Docker Training (Recommended)

```bash
# Navigate to PPO directory
cd /home/aharon/projects/new_swt/micro/nano/picco-ppo/

# Build with BuildKit caching
DOCKER_BUILDKIT=1 docker compose -f docker-compose.buildkit.yml build

# Run complete session (train + validate)
DOCKER_BUILDKIT=1 docker compose -f docker-compose.buildkit.yml up

# Or run specific operations:
# Full PPO with neural networks (recommended)
docker compose -f docker-compose.buildkit.yml run ppo python train.py

# Minimal rule-based version (faster, no GPU needed)
docker compose -f docker-compose.buildkit.yml run ppo python train_minimal.py

# Validation only
docker compose -f docker-compose.buildkit.yml run ppo python validate_minimal.py

# Resume from checkpoint
docker compose -f docker-compose.buildkit.yml run ppo python train_minimal.py --load-checkpoint checkpoints/best_checkpoint.pkl
```

### Local Training

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (includes PyTorch, Stable-Baselines3)
pip install -r requirements-minimal.txt

# Run training with checkpoints
python train_minimal.py --save-freq 2

# Run validation
python validate_minimal.py
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --model models/best/best_model.zip --n_episodes 10

# Or with Docker
docker compose --profile eval up evaluator
```

## 📁 Project Structure

```
picco-ppo/
├── env/
│   ├── trading_env.py             # Original Gymnasium environment
│   └── trading_env_optimized.py   # Optimized with precomputed features
├── precompute_features_to_db.py  # Precompute all features to DuckDB
├── run.py                         # Unified runner for all operations
├── train.py                       # PPO training with SB3
├── train_minimal.py               # AMDDP1 training implementation
├── validate_minimal.py            # Validation with expectancy_R
├── checkpoint_manager.py          # Checkpoint save/load system
├── entrypoint.sh                 # Docker entrypoint (auto-precompute)
├── models/                        # Saved models
├── checkpoints/                   # Training checkpoints
├── results/                       # Evaluation results
├── Dockerfile.buildkit           # BuildKit-optimized Docker
├── docker-compose.buildkit.yml   # Docker orchestration
├── requirements-minimal.txt       # Minimal dependencies
└── FEATURE_FORMULAS.md           # Complete feature documentation
```

## 🎯 Training Commands

### Full PPO Training (Neural Networks)
```bash
# Basic training with defaults
python train.py

# Custom configuration
python train.py \
    --timesteps 1000000 \
    --n_envs 4 \
    --eval_freq 10000 \
    --save_freq 50000
```

### Minimal Training (Rule-based)
```bash
# Quick testing without neural networks
python train_minimal.py
```

### Resume Training
```bash
python train.py --load_model models/checkpoint_1000000_steps.zip
```

## 📈 Performance Monitoring

### TensorBoard Metrics
- Episode rewards
- Policy loss
- Value loss
- Entropy
- Learning rate
- Custom trading metrics

### Evaluation Metrics
- **Expectancy_R**: Van Tharp R-multiple system
- **Total return (%)**: Overall profitability
- **Win rate**: Percentage of winning trades
- **Sharpe ratio**: Risk-adjusted returns
- **Max drawdown**: Worst peak-to-trough
- **Trade frequency**: Trades per episode

## 🔧 Key Features

### Checkpoint Management
```python
from checkpoint_manager import CheckpointManager

# Automatically manages checkpoints
manager = CheckpointManager("checkpoints")
manager.save_checkpoint(state, episode, expectancy_R, metrics)
```
- Keeps 2 recent + best performer
- Tracks expectancy_R for each checkpoint
- Resume from any checkpoint

### AMDDP1 Reward Function
```python
# Asymmetric Mean Drawdown Duration Penalty
# Penalizes drawdowns to encourage risk management
reward = pnl_pips - 0.01 * cumulative_drawdown_sum

# Profit protection: Don't punish profitable trades
if pnl_pips > 0 and reward < 0:
    reward = 0.001  # Small positive reward

# Applied with 4 pip spread cost on position opening
```

### Rolling Expectancy Tracking
```python
from rolling_expectancy import RollingExpectancyTracker

# Track performance over multiple windows
tracker = RollingExpectancyTracker(window_sizes=[100, 500, 1000])

# Add trades and monitor evolution
for trade_pips in trades:
    tracker.add_trade(trade_pips)
    expectancies = tracker.calculate_expectancies()

    # Shows:
    # - 100-trade window: Quick response to changes
    # - 500-trade window: Medium-term stability
    # - 1000-trade window: Long-term validation
```

### Van Tharp Expectancy Rating
```python
# Calculate R-multiple expectancy
avg_loss = abs(np.mean(losses))  # R value
expectancy_R = expectancy_pips / avg_loss

# System quality rating
if expectancy_R > 0.5:
    quality = "EXCELLENT"
elif expectancy_R > 0.25:
    quality = "GOOD"
```

### Winner-Focused Learning (NEW)
```python
def _calculate_reward(self, prev_equity: float) -> float:
    """Phase-based reward system"""
    pnl_change = self.equity - prev_equity

    if self.profitable_trades_count < 1000:
        # PHASE 1: Learn from winners only
        if trade_just_closed:
            if self.last_trade_result > 0:
                reward = self.last_trade_result * scaling  # Learn from profit
            else:
                reward = 0.0  # Ignore losses completely
    else:
        # PHASE 2: Normal AMDDP1 reward
        reward = pnl_change - 0.005 * self.accumulated_dd

    return reward * self.reward_scaling
```

## 🐳 Docker Build Options

### Option 1: Full Image (with monitoring)
```bash
# Build with all features
docker build -t picco-ppo:latest -f Dockerfile .
# Size: ~12.6GB
# Includes: TensorBoard, Matplotlib, Ray
```

### Option 2: Minimal Image (core only)
```bash
# Build lightweight version
docker build -t picco-ppo:minimal -f Dockerfile.minimal .
# Size: ~10GB (saves 2.6GB)
# Excludes: TensorBoard, Matplotlib, Ray
```

### Dependency Breakdown
- **Core (Required)**: torch, numpy, stable-baselines3, gymnasium, pandas, duckdb, numba
- **TensorBoard (Optional, +200MB)**: tensorboard, grpcio, protobuf, werkzeug
- **Visualization (Optional, +150MB)**: matplotlib, pillow, contourpy, fonttools
- **Ray (Optional, +100MB)**: ray, msgpack, jsonschema, requests

## 🚀 Running the System

### Quick Start
```bash
# One-time: Precompute features (if not done already)
python3 precompute_features_to_db.py

# Build with cache (fast) - BuildKit automatically caches layers
DOCKER_BUILDKIT=1 docker compose build ppo-training

# Start training with multi-environment support (4 parallel envs)
docker compose up -d ppo-training

# View logs
docker logs -f ppo-training

# Check metrics
docker exec ppo-training cat results/latest.json
```

### Current Configuration
- **Environments**: 4 (parallel environments with precomputed features)
- **Timesteps**: 1,000,000
- **Data Loading**: Optimized with precomputed features database
- **Spread**: 4 pips fixed (no curriculum)
- **Learning Phases**:
  - Phase 1: Learn from winners only (until 1000 profitable trades)
  - Phase 2: Normal learning (all trades)

## 📈 Performance & Resource Usage

### Training Speed
- **Initialization**: 3-5 minutes (loading 140k bars)
- **Training**: ~1000 steps/minute with single env
- **Full run**: ~16-20 hours for 1M timesteps

### Resource Requirements
- **CPU**: ~99% during training (normal for PPO)
- **Memory**: 8GB allocated, ~4GB typical usage
- **Disk**: ~100MB for checkpoints + logs
- **Docker Image**: 10-12.6GB depending on build

## 🔧 Troubleshooting

### Common Issues

1. **Training doesn't start**:
   ```bash
   # Check logs
   docker logs ppo-training

   # Verify environment loads
   docker exec ppo-training python -c "from env.trading_env import TradingEnv; print('OK')"
   ```

2. **Multiprocessing errors**:
   - Use `n_envs=1` in docker-compose.yml
   - Mount env/ directory as volume

3. **Slow builds**:
   - Use minimal requirements to save 2.6GB
   - BuildKit cache configured for subsequent builds
   - First build downloads PyTorch (888MB)

## 🔧 Customization

### Modify Hyperparameters

Edit `train.py`:
```python
ppo_config = {
    "learning_rate": 3e-4,  # Adjust learning rate
    "n_steps": 2048,        # Steps before update
    "batch_size": 64,       # Minibatch size
    "ent_coef": 0.01,      # Exploration coefficient
}
```

### Change Network Architecture

```python
"policy_kwargs": {
    "net_arch": [512, 256, 128],  # Deeper network
    "activation_fn": torch.nn.Tanh,  # Different activation
}
```

### Adjust Reward Function

Edit `env/trading_env.py`:
```python
def _calculate_reward(self, prev_equity: float) -> float:
    # Customize reward shaping here
    pass
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce `n_envs` parameter
- Decrease `batch_size`
- Use smaller network architecture

### Slow Training
- Enable GPU: Check CUDA availability
- Use SubprocVecEnv for true parallelization
- Reduce evaluation frequency

### Poor Performance
- Increase training timesteps
- Tune hyperparameters
- Check data quality and features

## 📊 Empirical Foundation

This implementation is based on extensive backtesting that showed:
- **M5/H1 Performance**: 444.6 pips with 38.6% win rate
- **3.2x Better**: Outperforms M1/H1 (138.5 pips)
- **Optimal Features**: 17 carefully selected features
- **Regime Adaptive**: Switches between trend/mean-reversion
- **Realistic Costs**: 4 pip spread matches live trading
- **Time Aware**: Captures market session patterns

## 📈 Training Results

**Latest Training Run (Sept 26, 2025):**
- **Steps Trained**: 119,700+
- **Total Profit**: 17,422 pips
- **Total Trades**: 5,913
- **Average**: ~2.95 pips/trade (after 4 pip spread)
- **Status**: Profitable system demonstrating AMDDP1 effectiveness

## 🔗 Related Projects

- Main MuZero implementation: `/home/aharon/projects/new_swt/micro/`
- Swing analysis tools: `/home/aharon/projects/new_swt/micro/nano/`
- Data pipeline: `/home/aharon/projects/new_swt/data/`
- Rolling expectancy approach: Similar to [peoplesfintech.github.io](https://github.com/roni762583/peoplesfintech.github.io)

## 📝 Citation

Based on the optimal feature analysis and timeframe comparison performed in the nano experiments, achieving 444.6 pips with 38.6% win rate on GBPJPY M5/H1 data.

## 📄 License

Proprietary - Internal use only
## 📐 Feature Implementation Details

### Position Features Normalization
```python
# All position features properly normalized for NN input
position_features = np.array([
    self.position_side,              # Already -1, 0, 1
    self.position_pips / 100.0,      # Scale to ~[-1, 1] range
    self.bars_since_entry / 100.0,   # Scale to ~[0, 1] range
    self.pips_from_peak / 100.0,     # Scale drawdown
    self.max_drawdown_pips / 100.0,  # Scale max DD
    self.accumulated_dd / 1000.0     # Scale cumulative DD
])
```

### Market Feature Calculations

#### React Ratio (Momentum)
```python
reactive = close - SMA(close, 200)
less_reactive = SMA(close, 20) - SMA(close, 200)
react_ratio = reactive / (less_reactive + 0.0001)
react_ratio = clip(react_ratio, -5, 5)
```

#### Efficiency Ratio (Trend Strength)
```python
direction = abs(close[t] - close[t-10])
volatility = sum(abs(close[i] - close[i-1]) for i in range(t-9, t+1))
efficiency_ratio = direction / (volatility + 0.0001)
# Range: [0, 1] where higher = stronger trend
```

#### Bollinger Band Position
```python
bb_middle = SMA(close, 20)
bb_std = STD(close, 20)
bb_position = (close - bb_middle) / (2 * bb_std + 0.0001)
bb_position = clip(bb_position, -1, 1)
# -1 = lower band, 0 = middle, 1 = upper band
```

#### RSI Extreme (Overbought/Oversold)
```python
rsi = calculate_rsi(close, 14)
rsi_extreme = (rsi - 50) / 50
# Range: [-1, 1] where -1 = oversold, 1 = overbought
```

## 📁 Project Structure
```
picco-ppo/
├── env/
│   └── trading_env.py         # Trading environment with 17 features
├── train.py                   # Main training script
├── validate_minimal.py        # Validation script
├── monitor_expectancy_live.py # Live monitoring
├── requirements.txt           # Full dependencies (12.6GB image)
├── requirements-minimal.txt   # Core only (10GB image)
├── Dockerfile                 # Full feature build
├── Dockerfile.minimal         # Lightweight build
├── docker-compose.yml         # Container orchestration
├── checkpoints/              # Model checkpoints (volume)
├── results/                  # Training results (volume)
└── README.md                 # This file
```

## 📊 Latest Performance Metrics

> 🎉 **MILESTONE ACHIEVED**: System has achieved **POSITIVE EXPECTANCY** (+0.022 pips/trade) while overcoming a 4-pip spread handicap!

### Training Results (Sept 28, 2025 - Live Update)
- **Training Status**: Running with centralized config (21:02 EST)
- **Current Phase**: Phase 1 - Learning from winners only
- **Checkpoint Settings**: Saving every 10,000 timesteps
- **Configuration**: All settings now in `config.py`
- **Previous Best**: +0.022 pips expectancy at 53,700 trades
- **Key Improvements**:
  - ✅ Fixed GBPJPY pip calculations (0.01 not 0.0001)
  - ✅ Centralized all settings in config.py
  - ✅ Checkpoint frequency reduced to 10k (was 50k)
  - ✅ All mounts properly configured (config, database, env, train)

### Key Improvements
- Removed curriculum learning for consistent 4 pip spread
- Winner-focused learning shows faster convergence
- Single environment eliminates Docker multiprocessing issues
- Optimized dependencies reduce image by 2.6GB

---
*Documentation consolidated from README.md and FEATURE_FORMULAS.md*
*Last updated: September 28, 2025*
