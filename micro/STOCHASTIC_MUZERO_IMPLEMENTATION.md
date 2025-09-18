# Stochastic MuZero Implementation for Micro Trading

## Overview
This implementation adds market uncertainty modeling to MuZero through stochastic planning with discrete market outcomes. This directly addresses the Hold-only collapse problem by allowing the model to reason about uncertain outcomes rather than avoiding them.

## Key Components

### 1. Market Outcome Modeling
- **3 Discrete Outcomes**: UP, NEUTRAL, DOWN
- **Based on Rolling Standard Deviation**:
  - UP: price change > 0.5 * rolling_stdev
  - NEUTRAL: price change within ±0.5 * rolling_stdev
  - DOWN: price change < -0.5 * rolling_stdev
- **Why Stdev over ATR**: Simpler, more responsive at micro timeframes

### 2. Network Architecture Changes

#### OutcomeProbabilityNetwork (`micro_networks.py`)
```python
class OutcomeProbabilityNetwork:
    # Predicts P(UP|s,a), P(NEUTRAL|s,a), P(DOWN|s,a)
    # Input: hidden_state + action -> Output: softmax probabilities
```

#### Modified DynamicsNetwork
```python
# Old: s' = f(s, a, z)  # Generic stochastic z
# New: s' = f(s, a, outcome)  # Market outcome conditioning
```

### 3. Stochastic MCTS (`stochastic_mcts.py`)

#### Tree Structure
- **DecisionNode**: Agent chooses action (existing)
- **ChanceNode**: Market determines outcome (NEW)
- Alternating layers: Decision → Chance → Decision

#### Planning Process
1. From DecisionNode, select action using UCB
2. Transition to ChanceNode for that action
3. Sample market outcome based on predicted probabilities
4. Transition to next DecisionNode based on outcome
5. Backup values through both node types

### 4. Training Integration

#### Experience Structure
```python
class Experience:
    observation: np.ndarray
    action: int
    policy: np.ndarray
    value: float
    reward: float
    done: bool
    market_outcome: int  # NEW: Actual outcome (0=UP, 1=NEUTRAL, 2=DOWN)
    outcome_probs: np.ndarray  # NEW: Predicted probabilities
```

#### Loss Components
1. **Policy Loss**: Cross-entropy between MCTS policy and network output
2. **Value Loss**: MSE between MCTS value and network prediction
3. **Reward Loss**: MSE between actual and predicted rewards
4. **Outcome Loss** (NEW): Cross-entropy between predicted and actual outcomes

## Benefits

### 1. Solves Hold-Only Collapse
- Model understands that uncertain trades can have positive expected value
- No longer avoids all action due to unpredictability
- Can distinguish between "risky but good" vs "certain loss"

### 2. Better Risk Assessment
- Explicitly models different market scenarios
- Plans through multiple possible outcomes
- More robust decision-making under uncertainty

### 3. Improved Exploration
- Strong Dirichlet noise (α=1.0, fraction=0.5)
- High initial temperature (10.0) for action diversity
- Chance nodes naturally add exploration through outcome sampling

## Configuration

### MCTS Parameters
```python
num_simulations: 25  # Balanced depth vs width
depth_limit: 3  # 3 steps ahead (action → outcome → action → outcome → action)
dirichlet_alpha: 1.0  # Strong exploration
exploration_fraction: 0.5  # 50% exploration at root
```

### Market Outcome Parameters
```python
window_size: 20  # Rolling window for stdev calculation
threshold_multiplier: 0.5  # 0.5σ threshold for significant moves
```

## Testing

Run comprehensive tests:
```bash
python3 micro/tests/test_stochastic_components.py
```

Tests verify:
- Network output shapes and probability validity
- Information flow between components
- MCTS tree structure with chance nodes
- Market outcome calculations
- End-to-end integration

## Usage

### Fresh Training
```bash
# Clear old checkpoints and buffer
rm -rf micro/checkpoints/*.pth micro/buffer/*

# Start training with stochastic MuZero
docker compose up -d --build micro-training

# Monitor progress
docker logs -f micro_training | grep -E "Expectancy|Action distribution"
```

### Key Metrics to Watch
1. **Action Distribution**: Should show diversity, not 100% Hold
2. **Expectancy**: Should gradually improve from negative to positive
3. **Outcome Prediction Accuracy**: Model should learn market patterns

## Implementation Status
✅ OutcomeProbabilityNetwork implemented
✅ DynamicsNetwork modified for outcome conditioning
✅ StochasticMCTS with chance nodes
✅ Market outcome calculator with rolling stdev
✅ Comprehensive test suite (all tests passing)
✅ Training loop integration prepared

## Next Steps
1. Run fresh training with stochastic implementation
2. Monitor for Hold-only behavior elimination
3. Fine-tune hyperparameters based on results
4. Consider adaptive threshold multipliers