# Training Improvements Implemented

## Date: 2025-09-22

### Problem Identified
System stabilized at negative expectancy (-4.0 pips) due to:
1. Poor MCTS signal quality (only 5 simulations)
2. Misaligned reward shaping encouraging overtrading
3. High exploration noise making targets random
4. Learning rate too high for stable updates

### Changes Implemented

#### 1. MCTS Improvements ✅
- **num_simulations**: 5 → 25 (5x increase for better targets)
- **dirichlet_alpha**: 1.0 → 0.3 (less exploration noise)
- **exploration_fraction**: 0.5 → 0.25 (clearer targets)

#### 2. Training Hyperparameters ✅
- **learning_rate**: 0.002 → 0.0005 (4x reduction for stability)
- **batch_size**: 64 → 128 (2x increase for better value learning)
- **value_loss_weight**: Already 1.0 (equal weighting maintained)

#### 3. Reward Simplification ✅
**OLD SCHEME:**
- Entry (BUY/SELL): +1.0 bonus
- Hold when flat: -0.05 penalty
- Close: AMDDP1 reward

**NEW SCHEME:**
- Entry (BUY/SELL): 0.0 (no bonus - only reward at close)
- Hold when flat: -0.01 (small idle penalty)
- Hold in position: 0.0
- Close: AMDDP1 reward (unchanged)
- Invalid actions: 0.0 (no noise)

### Expected Improvements (2-4 hours)

1. **KL(policy||visits)**: Should drop by 30-50%
2. **Value correlation**: Should rise from ~0 to >0.3
3. **Expectancy**: Should move toward 0 from -4
4. **Win rate**: May drop initially but stabilize higher
5. **Trade ratio**: Should decrease (fewer but better trades)

### Files Modified

1. `/micro/training/train_micro_muzero.py`
   - Updated TrainingConfig parameters
   - Modified MCTS initialization

2. `/micro/training/episode_runner.py`
   - Removed entry bonuses
   - Simplified reward structure

3. `/micro/training/parallel_episode_collector.py`
   - Fixed override that forced 5 simulations

4. `/micro/monitoring/training_diagnostics.py`
   - Created new diagnostic monitoring tool

### Validation Configuration Fixed
- MC_RUNS: 1000 → 200 (for faster validation)
- Added environment variable support
- Fixed TensorBoard import issues

### Monitoring Commands

```bash
# Check training progress
docker logs micro_training --tail 100 | grep Episode

# Run diagnostics
python3 /home/aharon/projects/new_swt/micro/monitoring/training_diagnostics.py

# Monitor simple dashboard
python3 /home/aharon/projects/new_swt/monitor_simple.py

# Check validation
docker logs micro_validation --tail 50
```

### Success Criteria
- Expectancy improves to > -2.0 within 2 hours
- Expectancy reaches positive within 6 hours
- Validation PDFs generate successfully
- Trade quality improves (longer average trade length)

### Notes
- AMDDP1 reward formula: `pnl_pips - 0.01 * cumulative_drawdown`
- System now optimizes for profitable trades, not activity
- Simplified reward aligns training with true objective