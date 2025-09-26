# PPO M5/H1 Trading System

Proximal Policy Optimization implementation for the optimal M5/H1 trading strategy discovered through comprehensive timeframe analysis.

**Latest Updates (Sept 2025)**:
- ✅ Van Tharp Expectancy_R rating system
- ✅ AMDDP1 reward function (pips-based with drawdown penalty)
- ✅ 4 pip spread cost on position opening
- ✅ Cyclic time features (sin/cos hour of day/week)
- ✅ Real PPO with neural networks (stable-baselines3)
- ✅ Checkpoint management (keep 2 + best)
- ✅ Docker BuildKit optimization
- ✅ 1M+ bars support (60/30/10 splits)

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
│   └── trading_env.py             # Gymnasium environment
├── run.py                         # Unified runner for all operations
├── train_minimal.py               # AMDDP1 training implementation
├── validate_minimal.py            # Validation with expectancy_R
├── checkpoint_manager.py          # Checkpoint save/load system
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

## 🔗 Related Projects

- Main MuZero implementation: `/home/aharon/projects/new_swt/micro/`
- Swing analysis tools: `/home/aharon/projects/new_swt/micro/nano/`
- Data pipeline: `/home/aharon/projects/new_swt/data/`

## 📝 Citation

Based on the optimal feature analysis and timeframe comparison performed in the nano experiments, achieving 444.6 pips with 38.6% win rate on GBPJPY M5/H1 data.

## 📄 License

Proprietary - Internal use only