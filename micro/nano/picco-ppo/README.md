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
├── checkpoints/                    # Saved models
├── master.duckdb                   # M1 OHLCV + swing features (1.3M bars)
├── generate_all_features.py        # Master feature generation script
├── add_swing_points.py             # Swing detection (M1 and H1)
├── add_swing_range_position.py     # H1 range position calculation
├── add_arctan_zscore_features.py   # Z-score features with fixed std
├── analyze_extreme_events.py       # Extreme event profitability analysis
├── feature_zscore_config.json      # Fixed std configuration
└── db-state.txt                    # Complete database documentation
```

---

## 📊 Feature Engineering & Analysis

### Master Database (master.duckdb)

1.3M M1 bars (GBPJPY, 2022-2025) with comprehensive swing-based features:

#### Swing Detection
- **M1 swings**: 274,699 highs, 274,264 lows (20.6% of bars)
- **H1 swings**: 22,394 highs, 22,394 lows (1.7% of bars)
- Pattern: `h[i] > h[i-1] AND h[i] > h[i+1]`

#### Position Features
- **h1_swing_range_position**: Price location within H1 range [0, 1]
- **swing_point_range**: H1 range magnitude (consolidation detector)

#### Z-Score Features (Window=20, Fixed Std)
- **h1_swing_range_position_zsarctan_w20**:
  - Detects extreme price positions within H1 range
  - Fixed std=3.421357 (923K training rows)
  - Extremes (|z|>0.8): 3,228 events (0.25%)
- **swing_point_range_zsarctan**:
  - Detects when H1 range itself is extreme (tight/wide)
  - Fixed std=0.221478 (933K training rows)
  - Extremes (|z|>0.8): 3,237 events (0.24%)
- **combo_geometric** (Interaction Feature):
  - Formula: `sign(r×p) × sqrt(|range_z| × |position_z|)`
  - Geometric mean: penalizes imbalance (small×big < medium×medium)
  - **69% better** than arithmetic multiply (0.077 vs 0.046 avg correlation)
  - Strong signals (|z|>0.5): 7,828 events (0.60%)
  - Quadrant-specific predictive power:
    - Q2 (big range, centered): **+0.184** correlation (strongest!)
    - Q1 (both big): +0.104 correlation
    - Q3 (consolidation extreme): -0.062 (reversal signal)

### Extreme Event Analysis Results

**Key Findings** (from analyze_extreme_events.py):

| Metric | Window=20 | Window=500 |
|--------|-----------|------------|
| Events/day | 3.5 | 7.6 |
| Continuation rate | 62.6% | 55.8% |
| Mean 30-min return | +5.70 pips | +1.29 pips |
| Pullback rate | 70% | 76% |

**Entry Strategy**:
- Immediate entry (0% retracement): +4.08 pips mean
- Wait for 22% retracement: +1.40 pips mean (misses 43% of trades)
- **Conclusion**: Immediate entry optimal; S5/tick data NOT recommended

**Feature Value for PPO**:
- Weak standalone edge (5.8% over random)
- Break-even after 4 pip spread
- BUT: Captures real regime information
- PPO can learn optimal timing within 30-min window
- **Recommendation**: Include as supporting feature, combine with reactive/lessreactive

### Feature Generation

**Master script**: `generate_all_features.py`

Regenerates all features in correct dependency order:
1. Swing point detection (M1 and H1)
2. Last swing tracking (indices + prices)
3. H1 swing range position
4. Swing point range
5. Z-score features with Window=20
6. combo_geometric interaction feature

Runtime: ~7.5 minutes for 1.3M bars

**Interaction Feature Development**:
- Tested 7 combination methods (multiply, ratio, weighted, conditional, etc.)
- Geometric mean emerged as **best predictor** (69% improvement)
- Differentiates market regimes: wide range vs consolidation breakouts
- Test script: `test_feature_combinations.py`

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