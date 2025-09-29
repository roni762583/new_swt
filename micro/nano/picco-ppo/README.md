# PPO M5/H1 Trading System – 4-Action Setup

Proximal Policy Optimization with **explicit 4-action space**, **rolling σ-based gating**, and **128→128 MLP policy architecture**.
This version improves transparency and control by separating **Hold** and **Close** into distinct actions.

---

## 🎯 Action Space (4 explicit actions)

* **0 = Hold** → Stay flat if no position, or hold current position if already open.
* **1 = Buy** → Open a new long position (only if flat).
* **2 = Sell** → Open a new short position (only if flat).
* **3 = Close** → Exit current position (only if in a trade).

### 🔒 Action Masking

* If **flat**: mask *Close (3)*.
* If **in position**: mask *Buy (1)* and *Sell (2)*.
* PPO sees 4 outputs every step, but invalid actions are masked to prevent wasted probability.

This ensures:

* **Clarity** → "Hold" always means continue, "Close" always means exit.
* **Risk control** → Close is always available, even during gate blocks.
* **Learning efficiency** → Agent doesn't waste exploration on impossible actions.

---

## 🏗️ Neural Network Architecture

Policy/value function networks use a **2-layer MLP**:

```
Input → Dense(128, ReLU) → Dense(128, ReLU) → Policy / Value heads
```

* **Hidden size**: [128, 128] (lighter than previous 256 setup, reduces overfitting).
* **Activation**: ReLU.
* **Policy head**: 4 logits (softmax for action probabilities).
* **Value head**: scalar (state-value estimate).

---

## ⚙️ PPO Configuration

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [128, 128],
        "activation_fn": torch.nn.ReLU
    }
}
```

---

## 🔎 Gating System

* **Rolling σ (12-bar M5)** defines volatility threshold.
* **Hard gate** (early) → invalid entries masked, only Hold/Close remain.
* **Soft gate** (later) → entries allowed but penalized.
* **Threshold annealing**: k = 0.15 → 0.25 over training.

### Threshold Formula
```
threshold = max(k × σ, 2 × spread, 2 pips)
```

Where:
- **σ** = rolling standard deviation of 12-bar returns (M5)
- **k** = threshold multiplier (anneals from 0.15 to 0.25)
- **spread** = instrument spread (e.g., 4 pips for GBPJPY)

---

## 📈 Weighted Learning Strategy

Replaces winner-only learning with balanced weighting:

* **Winners**: weight = 1.0 (full learning)
* **Losers**: weight = 0.2 → 1.0 (annealed over 200k steps)

This prevents selection bias while still emphasizing profitable patterns early in training.

---

## 🎯 Training Objective

**AMDDP1 Reward Function**:
```
R = ΔPnL - 0.01 × AccumulatedDD + GatePenalty
```

Where:
- **ΔPnL** = Change in equity
- **AccumulatedDD** = Cumulative drawdown from peak
- **GatePenalty** = -0.1 when soft gating triggers

---

## 🚀 Quick Start

### 1. Precompute features (if not already done)
```bash
python precompute_features_to_db.py
```

### 2. Train with 4-action environment
```bash
python train_improved.py
```

### 3. Monitor training
```bash
# Real-time monitoring
python monitor_continuous.py

# Or bash script
./monitor_training_live.sh
```

### 4. Validate checkpoint
```bash
python validate_improved.py --model checkpoints/ppo_4action_500000_steps.zip
```

### 5. Live trading (with deployment risk controls)
```bash
python trade_live_improved.py --model checkpoints/best_model.zip
```

---

## 📊 Performance Metrics

### Latest Training Results (1M timesteps)
- **Win Rate**: ~28% (targeting 30%+)
- **Expectancy**: -2.1 pips/trade (improving)
- **Gate Rate**: 45% (filtering noise effectively)
- **Sharpe Ratio**: -0.8 (early stage)

### Key Indicators
- **Van Tharp SQN**: Score > 2.0 indicates good system
- **Recovery Ratio**: Profit factor / max drawdown
- **Calmar Ratio**: Annual return / max drawdown

---

## 🛡️ Risk Management

### Training Environment Constraints
- **Fixed position size**: 1000 units (no compounding)
- **Max positions**: 1 (no pyramiding)
- **Episode termination**: 20% drawdown

### Deployment Risk Controls (Live Trading Only)
These are **NOT** enforced during training to avoid biasing the learning:
- **Max daily loss**: 2%
- **Max drawdown**: 5%
- **Consecutive loss circuit breaker**: 5 trades

---

## ✅ Advantages of 4-Action Setup

* **Transparency**: Each action has single meaning (no dual-purpose "hold/close")
* **Control**: Close action always explicit and available
* **Better debugging**: Logs and reward attribution clearer
* **Safer risk management**: Stops and overrides don't get trapped by gating

---

## 📁 Project Structure

```
picco-ppo/
├── env/
│   └── trading_env_4action.py     # 4-action environment with masking
├── config_improved.py              # Configuration settings
├── train_improved.py               # Training script (now uses 4-action)
├── validate_improved.py            # Validation and metrics
├── trade_live_improved.py          # Live trading with risk controls
├── monitor_continuous.py           # Python monitoring
├── monitor_training_live.sh        # Bash monitoring
└── checkpoints/                    # Saved models
```

---

## 🔬 Technical Details

### Feature Space (24 dimensions)
- **Market features** (7): OHLC ratios, volume, EMAs, RSI
- **Position features** (9): Position state, PnL, time in position, drawdown metrics
- **Account features** (1): Balance change ratio
- **Gating features** (3): Rolling σ, threshold, gate flag
- **Action mask** (4): Valid action indicators

### Action Masking Implementation
The environment provides a 4-element mask in the observation:
- `[1, 1, 1, 0]` when flat (can Hold, Buy, Sell, but not Close)
- `[1, 0, 0, 1]` when in position (can Hold or Close, but not Buy/Sell)

---

This setup balances **learning efficiency** (with action masking) and **practical trading control**, while keeping the lighter **128→128 MLP** for stability and speed.