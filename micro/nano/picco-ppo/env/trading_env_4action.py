"""Trading environment with 4-action space and action masking."""

import numpy as np
import pandas as pd
import duckdb
from gymnasium import spaces, Env
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TradingEnv4Action(Env):
    """Trading environment with explicit 4-action space and action masking.

    Action Space:
        0: Hold - Stay flat or maintain current position
        1: Buy - Open long position (only if flat)
        2: Sell - Open short position (only if flat)
        3: Close - Exit current position (only if in trade)
    """

    def __init__(
        self,
        db_path: str = "precomputed_features.duckdb",
        episode_length: int = 1000,
        initial_balance: float = 10000.0,
        instrument: str = "EUR_USD",
        reward_scaling: float = 100.0,
        seed: Optional[int] = None,
        # Gating parameters
        sigma_window: int = 12,
        k_threshold: float = 0.15,
        m_spread: float = 2.0,
        min_threshold_pips: float = 2.0,
        use_hard_gate: bool = True,
        gate_penalty: float = -0.1,
        # Weighted learning parameters
        winner_weight: float = 1.0,
        loser_weight: float = 0.2,
        weight_anneal_steps: int = 200_000,
    ):
        """Initialize trading environment with 4-action space."""
        super().__init__()

        # Trading parameters
        self.initial_balance = initial_balance
        self.instrument = instrument
        self.reward_scaling = reward_scaling
        self.episode_length = episode_length

        # Set instrument-specific parameters
        if instrument == "EUR_USD":
            self.pip_value = 0.0001
            self.spread = 0.00013  # 1.3 pips
        elif instrument == "GBP_JPY":
            self.pip_value = 0.01
            self.spread = 0.04  # 4 pips
        else:
            self.pip_value = 0.0001
            self.spread = 0.0002

        # Gating parameters
        self.sigma_window = sigma_window
        self.k_threshold = k_threshold
        self.m_spread = m_spread
        self.min_threshold_pips = min_threshold_pips
        self.use_hard_gate = use_hard_gate
        self.gate_penalty_value = gate_penalty

        # Weighted learning parameters
        self.winner_weight = winner_weight
        self.loser_weight = loser_weight
        self.weight_anneal_steps = weight_anneal_steps
        self.global_timesteps = 0

        # Fixed position sizing
        self.trade_size = 1000

        # Connect to database
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

        # Get available data count
        total_rows = self.conn.execute(
            "SELECT COUNT(*) FROM m5_features"
        ).fetchone()[0]

        if total_rows < episode_length:
            raise ValueError(f"Not enough data: {total_rows} < {episode_length}")

        self.max_start_index = total_rows - episode_length
        logger.info(f"Environment initialized with {total_rows} rows for {instrument}")

        # Initialize state variables
        self.position = 0
        self.entry_price = 0.0
        self.balance = initial_balance
        self.equity = initial_balance
        self.current_step = 0
        self.episode_data = None
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Performance tracking
        self.total_trades_count = 0
        self.profitable_trades_count = 0
        self.accumulated_dd = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown_pips = 0.0
        self.pips_from_peak = 0.0
        self._last_action = 0
        self.entry_step = 0

        # Gating tracking
        self.gates_triggered = 0
        self.false_rejects = 0
        self.successful_gates = 0
        self.rolling_std_values = []
        self.price_history = []
        self.current_rolling_std = 0.0
        self.current_threshold = self.min_threshold_pips
        self.gate_allowed = True

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: hold, 1: buy, 2: sell, 3: close

        # 24 features total (20 original + 4 action mask)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

        # Set seed
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode."""
        if seed is not None:
            self.seed(seed)

        # Reset trading state
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.entry_step = 0

        # Reset episode tracking
        self.current_step = 0
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Reset performance tracking
        self.accumulated_dd = 0.0
        self.peak_equity = self.initial_balance
        self.max_drawdown_pips = 0.0
        self.pips_from_peak = 0.0
        self._last_action = 0

        # Reset gating tracking
        self.gates_triggered = 0
        self.false_rejects = 0
        self.successful_gates = 0
        self.rolling_std_values = []
        self.price_history = []
        self.current_rolling_std = 0.0
        self.current_threshold = self.min_threshold_pips

        # Select random episode
        start_index = self.np_random.randint(0, self.max_start_index)
        end_index = start_index + self.episode_length

        # Load episode data
        query = f"""
        SELECT *
        FROM m5_features
        ORDER BY timestamp
        LIMIT {self.episode_length}
        OFFSET {start_index}
        """
        self.episode_data = self.conn.execute(query).df()

        # Return initial observation
        return self._get_observation(), {}

    def _calculate_rolling_std(self) -> Tuple[float, float]:
        """Calculate rolling standard deviation and threshold."""
        if len(self.price_history) < self.sigma_window:
            return 0.0, self.min_threshold_pips

        # Get last sigma_window prices
        prices = np.array(self.price_history[-self.sigma_window:])

        # Calculate returns in pips
        returns_pips = np.diff(prices) * 100

        # Calculate rolling std
        rolling_std = np.std(returns_pips) if len(returns_pips) > 0 else 0.0

        # Calculate annealed k_threshold
        anneal_progress = min(self.global_timesteps / 500_000, 1.0)
        current_k = self.k_threshold + (0.25 - self.k_threshold) * anneal_progress

        # Calculate threshold
        threshold = max(
            current_k * rolling_std,
            self.m_spread * self.spread * 100,  # Convert spread to pips
            self.min_threshold_pips
        )

        return rolling_std, threshold

    def _check_gate(self, action: int, recent_move_pips: float) -> Tuple[bool, float]:
        """Check if trade should be gated based on rolling std."""
        # Only gate new entry actions
        if action not in [1, 2]:  # Not buy or sell
            return True, 0.0

        # Check if movement exceeds threshold
        if abs(recent_move_pips) < self.current_threshold:
            self.gates_triggered += 1

            if self.use_hard_gate:
                # Hard gate: block trade
                return False, 0.0
            else:
                # Soft gate: allow but penalize
                return True, self.gate_penalty_value
        else:
            # Movement exceeds threshold - allow trade
            return True, 0.0

    def _get_action_mask(self) -> np.ndarray:
        """Get action mask for current state.

        Returns:
            Array of 4 values: 1 if action is valid, 0 if masked
        """
        mask = np.ones(4, dtype=np.float32)

        if self.position == 0:  # If flat
            mask[3] = 0  # Cannot close
        else:  # If in position
            mask[1] = 0  # Cannot buy
            mask[2] = 0  # Cannot sell

        return mask

    def _get_observation(self) -> np.ndarray:
        """Get current observation with action mask."""
        if self.current_step >= len(self.episode_data):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        row = self.episode_data.iloc[self.current_step]
        features = []

        # Market features (0-6)
        features.extend([
            row['open'] / row['close'] - 1,         # 0: Open ratio
            row['high'] / row['close'] - 1,         # 1: High ratio
            row['low'] / row['close'] - 1,          # 2: Low ratio
            row['volume'] / 1000,                   # 3: Volume (scaled)
            row['reactive'] / row['close'] - 1,     # 4: Fast EMA ratio
            row['lessreactive'] / row['close'] - 1,  # 5: Slow EMA ratio
            (row['rsi'] - 50) / 50,                 # 6: RSI (centered)
        ])

        # Account feature (7)
        features.append((self.balance - self.initial_balance) / self.initial_balance)

        # Position features (8-16)
        current_price = row['close']
        if self.position != 0:
            pnl = (current_price - self.entry_price) * self.position
            pnl_pips = pnl * 100
            bars_since_entry = (self.current_step - self.entry_step) / 100
        else:
            pnl_pips = 0.0
            bars_since_entry = 0.0

        # Update drawdown tracking
        current_equity_pips = pnl_pips if self.position != 0 else 0
        if current_equity_pips > self.pips_from_peak:
            self.pips_from_peak = current_equity_pips
        drawdown_pips = self.pips_from_peak - current_equity_pips
        self.max_drawdown_pips = max(self.max_drawdown_pips, drawdown_pips)

        features.extend([
            float(self.position),                    # 8: Current position
            pnl_pips / 100,                          # 9: PnL in pips (scaled)
            bars_since_entry,                        # 10: Time in position
            self.pips_from_peak / 100,              # 11: Pips from peak
            self.max_drawdown_pips / 100,           # 12: Max drawdown
            self.accumulated_dd / 1000,             # 13: Accumulated DD
            float(self._last_action),               # 14: Last action
            float(self.position == 0),              # 15: Is flat
            float(len(self.episode_trades) % 10) / 10,  # 16: Trade frequency
        ])

        # Gating features (17-19)
        features.extend([
            self.current_rolling_std / 10,          # 17: Rolling std (scaled)
            self.current_threshold / 10,            # 18: Current threshold (scaled)
            float(self.gate_allowed),                # 19: Gate allowed flag
        ])

        # Action mask (20-23)
        action_mask = self._get_action_mask()
        features.extend(action_mask.tolist())

        return np.array(features, dtype=np.float32)

    def _calculate_reward(self, prev_equity: float, gate_penalty: float = 0.0) -> float:
        """Calculate reward using AMDDP1 with optional gate penalty."""
        pnl_change = self.equity - prev_equity

        # AMDDP1 base reward
        reward = pnl_change - 0.01 * self.accumulated_dd

        # Add gate penalty if applicable
        reward += gate_penalty

        # Scale reward
        reward = reward * self.reward_scaling

        return reward

    def get_sample_weight(self) -> float:
        """Get sample weight for current experience based on trade outcome."""
        if self.last_trade_result == 0:
            return 1.0  # Default weight for non-trade steps

        # Calculate annealed weight
        anneal_progress = min(self.global_timesteps / self.weight_anneal_steps, 1.0)

        if self.last_trade_result > 0:
            # Winner: always full weight
            return self.winner_weight
        else:
            # Loser: gradually increase weight from loser_weight to 1.0
            current_loser_weight = self.loser_weight + (1.0 - self.loser_weight) * anneal_progress
            return current_loser_weight

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment with 4-action logic."""
        if self.current_step >= len(self.episode_data) - 1:
            return self._get_observation(), 0.0, False, True, self._get_info()

        row = self.episode_data.iloc[self.current_step]
        current_price = row['close']
        prev_equity = self.equity

        # Update price history for rolling std
        self.price_history.append(current_price)
        if len(self.price_history) > self.sigma_window * 2:
            self.price_history = self.price_history[-self.sigma_window * 2:]

        # Calculate rolling std and threshold
        self.current_rolling_std, self.current_threshold = self._calculate_rolling_std()
        self.rolling_std_values.append(self.current_rolling_std)

        # Calculate recent price movement
        if len(self.price_history) >= 2:
            recent_move_pips = (self.price_history[-1] - self.price_history[-2]) * 100
        else:
            recent_move_pips = 0.0

        # Get action mask
        action_mask = self._get_action_mask()

        # Apply action masking - if action is invalid, default to hold
        if action_mask[action] == 0:
            action = 0  # Default to hold if invalid action

        # Check gate for entry actions
        gate_penalty = 0.0
        if action in [1, 2]:  # Buy or Sell
            self.gate_allowed, gate_penalty = self._check_gate(action, recent_move_pips)

            # Apply hard gating if necessary
            if self.use_hard_gate and not self.gate_allowed:
                action = 0  # Force hold

        # Store last trade result
        self.last_trade_result = 0.0

        # Execute action
        if action == 0:  # Hold
            # Just hold - no position change
            pass

        elif action == 1 and self.position == 0:  # Buy (only if flat)
            # Open long position
            self.position = 1
            self.entry_price = current_price + self.spread * self.pip_value
            self.entry_step = self.current_step
            self.equity = self.balance

        elif action == 2 and self.position == 0:  # Sell (only if flat)
            # Open short position
            self.position = -1
            self.entry_price = current_price - self.spread * self.pip_value
            self.entry_step = self.current_step
            self.equity = self.balance

        elif action == 3 and self.position != 0:  # Close position
            # Close position
            if self.position > 0:  # Close long
                exit_price = current_price - self.spread * self.pip_value
            else:  # Close short
                exit_price = current_price + self.spread * self.pip_value

            pnl = (exit_price - self.entry_price) * self.position * self.trade_size
            self.balance += pnl
            self.equity = self.balance

            # Track trade result in pips
            trade_pips = (exit_price - self.entry_price) * self.position * 100
            self.last_trade_result = trade_pips
            self.episode_trades.append(trade_pips)

            # Update trade counts
            self.total_trades_count += 1
            if pnl > 0:
                self.profitable_trades_count += 1

            # Log trade result
            if self.total_trades_count <= 10 or self.total_trades_count % 100 == 0:
                status = "✅" if pnl > 0 else "❌"
                weight = self.get_sample_weight()
                logger.info(f"{status} Trade #{self.total_trades_count}: {trade_pips:.1f} pips | "
                          f"Win Rate: {self.profitable_trades_count/max(self.total_trades_count,1)*100:.1f}% | "
                          f"Weight: {weight:.2f} | σ: {self.current_rolling_std:.2f}")

            # Reset position
            self.position = 0
            self.entry_price = 0.0

        # Update equity if still in position
        if self.position != 0:
            if self.position > 0:
                mark_price = current_price - self.spread * self.pip_value
            else:
                mark_price = current_price + self.spread * self.pip_value
            unrealized_pnl = (mark_price - self.entry_price) * self.position * self.trade_size
            self.equity = self.balance + unrealized_pnl

        # Update drawdown tracking
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.accumulated_dd = 0.0
        else:
            dd_amount = self.peak_equity - self.equity
            self.accumulated_dd += dd_amount

        # Calculate reward
        reward = self._calculate_reward(prev_equity, gate_penalty)

        # Update step
        self.current_step += 1
        self._last_action = action
        self.global_timesteps += 1

        # Check if episode is done
        terminated = self.equity <= self.initial_balance * 0.8  # 20% loss limit
        truncated = self.current_step >= self.episode_length

        # Get next observation
        obs = self._get_observation()

        return obs, reward, terminated, truncated, self._get_info()

    def _get_info(self) -> Dict:
        """Get info dictionary with trading metrics."""
        info = {
            "equity": self.equity,
            "balance": self.balance,
            "position": self.position,
            "total_trades": self.total_trades_count,
            "sample_weight": self.get_sample_weight(),
            "rolling_std": self.current_rolling_std,
            "threshold": self.current_threshold,
            "gates_triggered": self.gates_triggered,
        }

        if len(self.episode_trades) > 0:
            trades_array = np.array(self.episode_trades)
            winners = trades_array > 0
            info.update({
                "win_rate": np.mean(winners),
                "expectancy_pips": np.mean(trades_array),
                "avg_win": np.mean(trades_array[winners]) if np.any(winners) else 0,
                "avg_loss": np.mean(trades_array[~winners]) if np.any(~winners) else 0,
                "num_trades": len(self.episode_trades),
                "gate_rate": self.gates_triggered / max(self.current_step, 1),
            })

        return info

    def close(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()