# 🎯 Micro Stochastic MuZero Trader

## Executive Summary

A streamlined proof-of-concept implementation of Stochastic MuZero for forex trading using only **14 essential features** instead of the full 337-feature WST pipeline. This micro variant serves as both a **rapid development testbed** and **performance baseline** for the larger system.

### Key Advantages
- **10x faster training** - Reduced input dimension (14 vs 337)
- **Cleaner signal** - Focus on proven technical indicators
- **Parallel development** - Can run alongside main WST version
- **Baseline metrics** - Direct comparison with full model
- **TCN integration** - Temporal Convolutional Network captures market dynamics

### Architecture Innovation
The TCN encoder is **integrated directly inside the Representation Network** as its first module, creating an end-to-end learnable temporal feature extractor without requiring a separate autoencoder.

---

## 📊 Feature Set (14 Features)

### 1. Technical Indicators (4)
- `position_in_range_60` - Price position in 60-bar range [0,1]
- `min_max_scaled_momentum_60` - Long-term momentum normalized
- `min_max_scaled_rolling_range` - Volatility indicator
- `min_max_scaled_momentum_5` - Short momentum in long context

### 2. Cyclical Time Features (4)
- `dow_cos_final` - Day of week cosine encoding
- `dow_sin_final` - Day of week sine encoding
- `hour_cos_final` - Hour of day cosine encoding
- `hour_sin_final` - Hour of day sine encoding

### 3. Position State (6)
Simplified and consistently scaled position features:
1. **position_side** - Single categorical: -1 (short), 0 (flat), 1 (long)
2. **position_pips** - Current P&L: tanh(pips/100)
3. **bars_since_entry** - Time in position: tanh(bars/100)
4. **pips_from_peak** - Distance from best: tanh(pips/100)
5. **max_drawdown_pips** - Worst drawdown: tanh(pips/100)
6. **accumulated_dd** - Total drawdown area: tanh(accumulated_dd/100)

---

## 🏗️ Network Architecture

### Input Pipeline
```
Input: (batch_size, 64, 14)
       ↓
TCN Encoder (inside Representation)
       ↓
Latent: 48D vector (with attention pooling)
       ↓
5 Stochastic MuZero Networks
```

### Network 1: Representation (with embedded TCN)
```python
class RepresentationNetwork(nn.Module):
    def __init__(self):
        # TCN Front-End (integrated for end-to-end learning)
        self.tcn_encoder = TCNBlock(
            in_channels=14,
            out_channels=48,  # Optimal compression
            kernel_size=3,
            dilations=[1, 2, 4, 8],  # Multi-scale: 1-min, 2-min, 4-min, 8-min
            dropout=0.1,
            causal=True
        )

        # Temporal attention pooling (learns which timesteps matter)
        self.time_attention = nn.Linear(48, 1)

        # Skip connection projection (48D TCN + 14D raw = 62D)
        self.projection = nn.Linear(62, 256)

        # Standard residual blocks with dropout
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.1) for _ in range(3)
        ])

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, x):
        # x shape: (batch, 64, 14)
        batch_size = x.size(0)

        # TCN encoding with multi-scale temporal patterns
        tcn_out = self.tcn_encoder(x)  # (batch, 48, 64)
        tcn_out = self.dropout(tcn_out)  # Dropout after TCN

        # Attention-weighted temporal pooling
        attention_logits = self.time_attention(tcn_out.transpose(1, 2))  # (batch, 64, 1)
        attention_weights = F.softmax(attention_logits, dim=1)
        pooled = (tcn_out * attention_weights.transpose(1, 2)).sum(dim=2)  # (batch, 48)

        # Skip connection: combine temporal features with current state
        current_features = x[:, -1, :]  # Last timestep raw features (batch, 14)
        combined = torch.cat([pooled, current_features], dim=1)  # (batch, 62)

        # Project to hidden dimension
        hidden = self.projection(combined)  # (batch, 256)
        hidden = self.layer_norm(hidden)
        hidden = F.relu(hidden)

        # Apply residual blocks
        for block in self.residual_blocks:
            hidden = block(hidden)

        return hidden  # (batch, 256)
```

### Network 2: Dynamics
```python
class DynamicsNetwork(nn.Module):
    def __init__(self):
        # Input: hidden(256) + action(4) + stochastic_z(16) = 276D
        self.input_projection = nn.Linear(276, 256)

        # 3 residual blocks for transition modeling
        self.dynamics_blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.1) for _ in range(3)
        ])

        # Separate heads for next state and reward
        self.next_state_head = nn.Linear(256, 256)
        self.reward_head = nn.Linear(256, 1)

        self.layer_norm = nn.LayerNorm(256)

    def forward(self, hidden, action, z):
        # Combine inputs
        x = torch.cat([hidden, action, z], dim=1)  # (batch, 276)
        x = self.input_projection(x)
        x = F.relu(x)

        # Apply dynamics blocks
        for block in self.dynamics_blocks:
            x = block(x)

        # Generate next state and reward
        next_hidden = self.next_state_head(x)
        next_hidden = self.layer_norm(next_hidden)

        reward = self.reward_head(x)

        return next_hidden, reward
```

### Network 3: Policy
```python
class PolicyNetwork(nn.Module):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

        # 2 residual blocks for policy
        self.policy_blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.1) for _ in range(2)
        ])

        # Action head: 4 actions [Hold, Buy, Sell, Close]
        self.action_head = nn.Linear(256, 4)

    def forward(self, hidden):
        x = hidden

        # Apply policy blocks
        for block in self.policy_blocks:
            x = block(x)

        # Generate action logits with temperature scaling
        logits = self.action_head(x) / self.temperature

        return logits  # (batch, 4)
```

### Network 4: Value
```python
class ValueNetwork(nn.Module):
    def __init__(self, support_size=300):
        # Categorical value distribution: [-300, +300] pips in 601 bins
        self.support_size = support_size
        self.num_atoms = 2 * support_size + 1  # 601

        # 2 residual blocks for value estimation
        self.value_blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.1) for _ in range(2)
        ])

        # Value distribution head
        self.value_head = nn.Linear(256, self.num_atoms)

    def forward(self, hidden):
        x = hidden

        # Apply value blocks
        for block in self.value_blocks:
            x = block(x)

        # Generate value distribution logits
        value_logits = self.value_head(x)  # (batch, 601)
        value_probs = F.softmax(value_logits, dim=1)

        return value_probs
```

### Network 5: Afterstate
```python
class AfterstateNetwork(nn.Module):
    def __init__(self):
        # Input: hidden(256) + action(4) = 260D
        self.input_projection = nn.Linear(260, 256)

        # 2 residual blocks for afterstate modeling
        self.afterstate_blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.1) for _ in range(2)
        ])

        self.layer_norm = nn.LayerNorm(256)

    def forward(self, hidden, action):
        # Combine hidden state with action
        x = torch.cat([hidden, action], dim=1)  # (batch, 260)
        x = self.input_projection(x)
        x = F.relu(x)

        # Apply afterstate blocks
        for block in self.afterstate_blocks:
            x = block(x)

        # Generate afterstate
        afterstate = self.layer_norm(x)

        return afterstate  # (batch, 256)
```

---

## 📋 Implementation Work Plan

### Phase 1: Data Pipeline (Week 1)
- [ ] Create micro data loader using master.duckdb
- [ ] Extract 14 features for training windows
- [ ] Implement sliding window with lag=64
- [ ] Create position state calculator
- [ ] Setup train/validation/test splits

### Phase 2: TCN-Integrated Networks (Week 1-2)
- [ ] Implement TCNBlock with causal convolutions
- [ ] Create RepresentationNetwork with embedded TCN
- [ ] Port DynamicsNetwork from main project
- [ ] Port PolicyNetwork and ValueNetwork
- [ ] Implement AfterstateNetwork
- [ ] Create MicroStochasticMuZero wrapper class

### Phase 3: Training Infrastructure (Week 2)
- [ ] Adapt MCTS for micro features
- [ ] Setup experience replay buffer
- [ ] Implement training loop
- [ ] Add tensorboard logging
- [ ] Create checkpoint system

### Phase 4: Validation & Comparison (Week 3)
- [ ] Run baseline training (1M steps)
- [ ] Compare with full WST model
- [ ] Analyze TCN learned features
- [ ] Performance profiling
- [ ] Hyperparameter tuning

---

## 🔧 Technical Specifications

### Architecture Improvements
Key enhancements over basic design:
- **Attention Pooling**: Learns which timesteps are most important
- **Skip Connections**: Preserves raw features alongside temporal patterns
- **Temperature Scaling**: Controls exploration in policy/value networks
- **Dropout Strategy**: After TCN and within residual blocks
- **Layer Normalization**: Stabilizes training across all networks

### TCN Configuration
```python
tcn_config = {
    'in_channels': 14,
    'out_channels': 48,  # Optimal compression ratio
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8],  # Multi-scale temporal patterns
    'causal': True,
    'dropout': 0.1,
    'activation': 'relu',
    'use_batch_norm': True,
    'receptive_field': 64  # Covers full input window
}
```

### Training Configuration
```python
training_config = {
    'batch_size': 64,
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'gradient_clip': 10.0,
    'buffer_size': 100000,
    'num_unroll_steps': 5,
    'td_steps': 10,
    'discount': 0.997,
    'num_simulations': 15,  # MCTS
    'latent_z_dim': 16,     # Stochastic
    'kl_weight': 0.1        # KL regularization
}
```

### Model Dimensions
```
Total Parameters: ~1.3M (vs 2.5M full model)
- TCN Encoder: ~75K (includes attention)
- Representation: ~320K (with skip connections)
- Dynamics: ~400K (3 blocks + heads)
- Policy: ~150K (2 blocks + action head)
- Value: ~250K (2 blocks + 601-dim output)
- Afterstate: ~105K (2 blocks)
```

### Key Network Features
- **TCN → 48D compression**: Balanced information preservation
- **256D hidden state**: Consistent across all networks
- **16D stochastic z**: Captures market uncertainty
- **601 value bins**: Fine-grained value estimation
- **Attention mechanism**: Dynamic temporal focus

---

## 🚀 Quick Start

### 1. Prepare Data
```bash
cd /home/aharon/projects/new_swt/micro
python prepare_micro_data.py  # Extract 17 features from master.duckdb
```

### 2. Train Model
```bash
python train_micro_muzero.py \
    --config config/micro_config.json \
    --data data/micro_features.h5 \
    --epochs 1000
```

### 3. Evaluate
```bash
python evaluate_micro.py \
    --checkpoint checkpoints/best_model.pth \
    --test_data data/test_features.h5
```

---

## 📈 Expected Outcomes

### Performance Targets
- Training time: 24 hours to 1M steps (vs 5 days full model)
- Memory usage: <4GB GPU (vs 16GB full model)
- Inference speed: 1000 episodes/sec (vs 200 full model)
- Baseline Sharpe: >0.8 on test data

### Comparison Metrics
1. **Speed**: 10x faster training iteration
2. **Memory**: 4x less GPU memory
3. **Convergence**: 2x faster to baseline performance
4. **Interpretability**: TCN filters directly visualizable

---

## 🔬 Research Questions

1. **Can 14 features match WST performance?**
   - Direct A/B testing on same data splits
   - Statistical significance testing

2. **What temporal patterns does TCN learn?**
   - Visualize learned kernels
   - Attention weight analysis

3. **Is stochastic modeling necessary with fewer features?**
   - Compare with deterministic MuZero variant
   - Analyze uncertainty estimates

4. **Optimal lag window size?**
   - Test lag ∈ {32, 64, 128}
   - Receptive field analysis

---

## 📁 Directory Structure

```
micro/
├── README.md                  # This file
├── config/
│   └── micro_config.json     # All configurations
├── data/
│   ├── prepare_micro_data.py # Feature extraction
│   └── micro_dataloader.py   # PyTorch dataset
├── models/
│   ├── tcn_block.py          # TCN implementation
│   ├── micro_networks.py     # 5 networks with TCN
│   └── micro_muzero.py       # Main model wrapper
├── training/
│   ├── train_micro_muzero.py # Training script
│   ├── mcts_micro.py         # MCTS for micro
│   └── replay_buffer.py      # Experience replay
├── evaluation/
│   ├── evaluate_micro.py     # Testing script
│   └── compare_models.py     # WST vs Micro
└── notebooks/
    ├── tcn_analysis.ipynb    # TCN visualization
    └── performance.ipynb     # Results analysis
```

---

## 🎯 Success Criteria

✅ **Phase 1 Success**: Clean data pipeline with 17 features
✅ **Phase 2 Success**: All 5 networks training without errors
✅ **Phase 3 Success**: Convergence to positive Sharpe ratio
✅ **Phase 4 Success**: Baseline established for WST comparison

---

## 💡 Key Insights

The micro variant leverages the insight that **most trading signals come from simple technical indicators and time patterns**. By using a TCN to capture temporal dependencies directly in the Representation network, we eliminate the need for complex wavelet transforms while maintaining the ability to learn sophisticated market dynamics.

This approach serves as both a **rapid prototyping platform** and **interpretable baseline**, enabling faster iteration on core algorithmic improvements before scaling to the full feature set.

---

## 🚀 Performance Optimizations (NEW)

### Fast Initial Buffer Collection (100x speedup)
- **`fast_buffer_init.py`**: Skip MCTS for initial experiences
- Random or guided policy for instant buffer filling
- From hours to seconds initialization

### Parallel MCTS Implementation (4x speedup)
- **`parallel_mcts.py`**: Multi-threaded tree simulations
- `BatchedMCTS`: Process multiple observations simultaneously
- `ParallelMCTS`: Run 4 parallel MCTS trees
- `AsyncExperienceCollector`: Background experience generation

### Optimized Training Script
- **`optimized_train.py`**: Integrated all optimizations
- Fast buffer initialization
- Parallel simulations
- Batched inference
- Async collection

### Enhanced Quality Score Calculation
Heavily weighted towards trading performance:
- **Pip P&L**: 2.0x (profit) / 0.3x (loss)
- **AMDDP1**: 1.5x weight (risk-adjusted returns)
- **Trade Completion**: +10.0/+2.0 bonus
- **SQN Component**: Up to +15.0 for excellent systems
- **Position Changes**: +3.0 (action diversity)
- **TD Error**: +5.0/+3.0/+1.0 (prediction accuracy)

### Checkpoint Preservation
- Validation container copies best checkpoints to safe location
- Prevents training from overwriting best models
- Maintains history of best performing checkpoints

**Current Status**: System running with all optimizations enabled!

## 📝 Feature Summary

**Total: 15 features** = 5 technical + 4 cyclical + 6 position
- Consistent `tanh(x/100)` scaling for all continuous features
- Single position_side feature instead of 3 binary flags
- Focus on essential trading signals without redundancy

## 🔧 Running the System

### Docker Compose (Recommended)
```bash
# Build and start all containers
docker compose up -d --build

# Monitor logs
docker logs -f micro_training
docker logs -f micro_validation

# Check status
docker ps | grep micro
```

### Optimized Training (Standalone)
```bash
python micro/training/optimized_train.py
```

This uses all performance optimizations for fastest training.