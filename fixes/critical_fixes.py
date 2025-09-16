#!/usr/bin/env python3
"""
Critical fixes for SWT Trading System
Addresses all major issues identified in code review
"""

# Issue 1: SPREAD COST FIX - Apply spread on position entry
def fix_spread_cost_in_environment():
    """
    Fix: Apply spread cost immediately when opening position
    Current: No spread deduction on entry
    Impact: Overly optimistic training
    """
    code_fix = """
    # In swt_forex_env.py - _execute_action() method

    elif action == SWTAction.BUY:
        if not (self.position.is_long or self.position.is_short):
            # Open long position
            self.position.is_long = True
            self.position.entry_price = current_price

            # FIX: Apply spread cost immediately
            self.position.unrealized_pnl_pips = -self.spread_pips  # Start with spread cost
            self.position.initial_spread_cost = self.spread_pips

    elif action == SWTAction.SELL:
        if not (self.position.is_long or self.position.is_short):
            # Open short position
            self.position.is_short = True
            self.position.entry_price = current_price

            # FIX: Apply spread cost immediately
            self.position.unrealized_pnl_pips = -self.spread_pips  # Start with spread cost
            self.position.initial_spread_cost = self.spread_pips
    """
    return code_fix


# Issue 2: CIRCULAR REPLAY BUFFER FIX
def fix_replay_buffer_memory_leak():
    """
    Fix: Implement circular buffer with max size
    Current: Unbounded growth causes OOM
    Impact: Training crashes after ~1000 episodes
    """
    code_fix = """
    # In training_main.py - SWTReplayBuffer class

    class SWTReplayBuffer:
        def __init__(self, max_size: int = 10000):
            self.buffer = []
            self.max_size = max_size
            self.position = 0

        def add(self, trajectory):
            if len(self.buffer) < self.max_size:
                self.buffer.append(trajectory)
            else:
                # Circular overwrite
                self.buffer[self.position] = trajectory
                self.position = (self.position + 1) % self.max_size

        def sample(self, batch_size: int):
            # Sample uniformly from available trajectories
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
            return [self.buffer[i] for i in indices]
    """
    return code_fix


# Issue 3: MCTS TEMPERATURE ANNEALING
def add_temperature_annealing():
    """
    Fix: Implement temperature schedule for exploration->exploitation
    Current: Fixed temperature = 1.0 forever
    Impact: Never converges to optimal policy
    """
    code_fix = """
    # In swt_mcts/swt_mcts_agent.py

    def get_temperature(self, episode: int, total_episodes: int) -> float:
        '''Annealing schedule: exploration -> exploitation'''

        # Phase 1: High exploration (episodes 0-30%)
        if episode < total_episodes * 0.3:
            return 1.5

        # Phase 2: Linear decay (episodes 30-70%)
        elif episode < total_episodes * 0.7:
            progress = (episode - total_episodes * 0.3) / (total_episodes * 0.4)
            return 1.5 - (1.5 - 0.5) * progress

        # Phase 3: Low temperature exploitation (episodes 70-100%)
        else:
            return 0.1  # Near-deterministic

    # Use in MCTS selection
    temperature = self.get_temperature(episode, total_episodes)
    action = self.select_action(root, temperature=temperature)
    """
    return code_fix


# Issue 4: WALK-FORWARD VALIDATION SPLITS
def implement_walk_forward_validation():
    """
    Fix: Proper temporal data splits
    Current: Random sampling causes data leakage
    Impact: Overfitting, poor live performance
    """
    code_fix = """
    # In data_utils.py or environment setup

    def split_data_temporally(df, train_ratio=0.6, val_ratio=0.2):
        '''Chronological split to prevent lookahead bias'''

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_data = df.iloc[:train_end]  # 2022-01 to 2023-06
        val_data = df.iloc[train_end:val_end]  # 2023-07 to 2024-06
        test_data = df.iloc[val_end:]  # 2024-07 to 2025-08

        print(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Val: {val_data.index[0]} to {val_data.index[-1]}")
        print(f"Test: {test_data.index[0]} to {test_data.index[-1]}")

        return train_data, val_data, test_data

    # In environment reset()
    def reset(self):
        if self.mode == 'train':
            # Only sample from training period
            valid_starts = self._get_valid_session_starts(self.train_data)
        elif self.mode == 'validation':
            # Only sample from validation period
            valid_starts = self._get_valid_session_starts(self.val_data)
    """
    return code_fix


# Issue 5: WST CACHING FOR PERFORMANCE
def add_wst_caching():
    """
    Fix: Cache WST features in memory
    Current: Disk I/O every step
    Impact: 10x slower training
    """
    code_fix = """
    # In swt_environments/swt_precomputed_wst_loader.py

    class PrecomputedWSTLoader:
        def __init__(self, h5_path: str, cache_size: int = 50000):
            self.h5_file = h5py.File(h5_path, 'r')
            self.wst_features = self.h5_file['wst_features']

            # FIX: LRU cache for frequently accessed features
            self.cache = {}
            self.cache_size = cache_size
            self.access_counts = {}

        def get_wst_features(self, index: int) -> np.ndarray:
            # Check cache first
            if index in self.cache:
                self.access_counts[index] += 1
                return self.cache[index]

            # Load from disk
            features = self.wst_features[index]

            # Add to cache
            if len(self.cache) >= self.cache_size:
                # Evict least recently used
                lru_index = min(self.access_counts, key=self.access_counts.get)
                del self.cache[lru_index]
                del self.access_counts[lru_index]

            self.cache[index] = features
            self.access_counts[index] = 1

            return features
    """
    return code_fix


# Issue 6: CIRCUIT BREAKERS FOR SAFETY
def add_circuit_breakers():
    """
    Fix: Emergency stop conditions
    Current: No safety limits
    Impact: Can blow account
    """
    code_fix = """
    # In live_trading/swt_live_trader.py

    class CircuitBreaker:
        def __init__(self):
            self.daily_loss_limit = 100  # pips
            self.consecutive_loss_limit = 5
            self.max_latency_ms = 1000
            self.max_drawdown_pips = 200

            # Tracking
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.is_active = True

        def check_conditions(self, trade_result, latency_ms):
            '''Check all circuit breaker conditions'''

            # Daily loss limit
            if self.daily_pnl <= -self.daily_loss_limit:
                logger.error(f"CIRCUIT BREAKER: Daily loss limit hit: {self.daily_pnl}")
                self.is_active = False
                return False

            # Consecutive losses
            if trade_result < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.consecutive_loss_limit:
                    logger.error(f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses")
                    self.is_active = False
                    return False
            else:
                self.consecutive_losses = 0

            # Latency check
            if latency_ms > self.max_latency_ms:
                logger.warning(f"CIRCUIT BREAKER: High latency {latency_ms}ms")
                return False  # Skip trade but don't disable

            return True

        def reset_daily(self):
            '''Reset daily counters at market close'''
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.is_active = True
    """
    return code_fix


# Issue 7: POSITION SIZE VALIDATION
def add_position_validation():
    """
    Fix: Validate every broker fill
    Current: Trusts broker blindly
    Impact: Position drift, wrong sizes
    """
    code_fix = """
    # In live_trading/oanda_client.py

    def validate_fill(self, requested_units, fill_response):
        '''Validate broker fill matches request'''

        filled_units = fill_response.get('units', 0)
        fill_price = fill_response.get('price', 0)

        # Check size match (allow 0.1% tolerance for rounding)
        size_diff = abs(filled_units - requested_units)
        if size_diff > abs(requested_units) * 0.001:
            raise ValueError(f"Fill size mismatch: requested {requested_units}, got {filled_units}")

        # Check price is reasonable (within 10 pips of last known)
        if self.last_price:
            price_diff_pips = abs(fill_price - self.last_price) / self.pip_value
            if price_diff_pips > 10:
                logger.warning(f"Large slippage: {price_diff_pips:.1f} pips")

        # Validate response structure
        required_fields = ['units', 'price', 'time', 'id']
        for field in required_fields:
            if field not in fill_response:
                raise ValueError(f"Missing field in fill response: {field}")

        return True
    """
    return code_fix


# Issue 8: BATCH MCTS INFERENCE
def add_batch_mcts():
    """
    Fix: Batch MCTS simulations
    Current: Sequential = 15x slower
    Impact: Slow training and inference
    """
    code_fix = """
    # In swt_mcts/swt_mcts_agent.py

    def run_batch_simulations(self, roots, num_simulations_per_root):
        '''Run MCTS simulations in parallel batches'''

        batch_size = min(len(roots), 32)  # GPU batch size

        # Collect all leaf nodes to evaluate
        leaves_to_evaluate = []
        for root in roots:
            for _ in range(num_simulations_per_root):
                leaf = self.select_leaf(root)
                leaves_to_evaluate.append(leaf)

        # Batch neural network evaluation
        if leaves_to_evaluate:
            states = torch.stack([leaf.state for leaf in leaves_to_evaluate])

            with torch.no_grad():
                # Single forward pass for all leaves
                policies, values = self.model(states)

            # Backpropagate results
            for i, leaf in enumerate(leaves_to_evaluate):
                self.expand_node(leaf, policies[i])
                self.backpropagate(leaf, values[i])
    """
    return code_fix


# Issue 9: ASYNC API CALLS
def add_async_api_calls():
    """
    Fix: Non-blocking API calls
    Current: Synchronous blocking
    Impact: Missed opportunities
    """
    code_fix = """
    # In live_trading/async_oanda_client.py

    import asyncio
    import aiohttp

    class AsyncOandaClient:
        def __init__(self):
            self.session = aiohttp.ClientSession()
            self.timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout

        async def get_price(self):
            '''Non-blocking price fetch'''
            try:
                async with self.session.get(
                    self.price_url,
                    headers=self.headers,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.parse_price(data)
                    else:
                        logger.error(f"Price fetch failed: {response.status}")
                        return None

            except asyncio.TimeoutError:
                logger.error("API timeout - using cached price")
                return self.last_price

        async def place_order_async(self, order_request):
            '''Non-blocking order placement'''
            async with self.session.post(
                self.order_url,
                json=order_request,
                headers=self.headers,
                timeout=self.timeout
            ) as response:
                return await response.json()
    """
    return code_fix


# Issue 10: PROPER LOGGING AND MONITORING
def add_comprehensive_logging():
    """
    Fix: Structured logging for debugging
    Current: Minimal logging
    Impact: Can't diagnose issues
    """
    code_fix = """
    # In all modules - standardized logging

    import structlog

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()

    # Use structured logging
    logger.info("trade_executed",
                action="BUY",
                price=current_price,
                spread=spread_pips,
                pnl=unrealized_pnl,
                duration_bars=position.duration_bars)
    """
    return code_fix


if __name__ == "__main__":
    print("Critical Fixes for SWT Trading System")
    print("=" * 60)

    fixes = [
        ("Spread Cost", fix_spread_cost_in_environment),
        ("Replay Buffer Memory Leak", fix_replay_buffer_memory_leak),
        ("MCTS Temperature Annealing", add_temperature_annealing),
        ("Walk-Forward Validation", implement_walk_forward_validation),
        ("WST Caching", add_wst_caching),
        ("Circuit Breakers", add_circuit_breakers),
        ("Position Validation", add_position_validation),
        ("Batch MCTS", add_batch_mcts),
        ("Async API", add_async_api_calls),
        ("Logging", add_comprehensive_logging),
    ]

    for name, fix_func in fixes:
        print(f"\n### {name} ###")
        print(fix_func.__doc__)
        print("\nImplementation:")
        print(fix_func())
        print("-" * 60)