# Archived Training Scripts

These scripts were moved to archive on 2025-09-21 during codebase cleanup.
They are preserved here for reference but are no longer actively used.

## Archived Files

### 1. `mcts_micro.py`
- **Reason**: Duplicate of `stochastic_mcts.py`
- **Status**: Superseded by stochastic_mcts.py which has the stochastic outcomes feature

### 2. `parallel_mcts.py`
- **Reason**: Unused parallel MCTS implementation
- **Status**: Not imported anywhere, parallel collection handled by `parallel_episode_collector.py`

### 3. `memory_cache.py`
- **Reason**: Old cache implementation
- **Status**: Superseded by `optimized_cache.py`

### 4. `simple_memory_cache.py`
- **Reason**: Another cache variant
- **Status**: Superseded by `optimized_cache.py`

### 5. `train_micro_muzero_v2.py`
- **Reason**: Old version of training script
- **Status**: Superseded by `train_micro_muzero.py`

### 6. `fast_buffer_init.py`
- **Reason**: Buffer initialization optimization
- **Status**: Features may be useful - contains FastBufferInitializer with:
  - `collect_random_experiences()` - fast random policy collection
  - `collect_guided_experiences()` - guided exploration
  - Could be reintegrated if buffer warmup becomes a bottleneck

## Note
These files are kept for reference and potential future use. The main training
flow now uses:
- `train_micro_muzero.py` - main training script
- `stochastic_mcts.py` - MCTS with market outcomes
- `optimized_cache.py` - memory-efficient data caching
- `parallel_episode_collector.py` - parallel episode collection