"""
Improved Trading Environment with rolling std-based gating and weighted learning.
Replaces ATR with rolling standard deviation for better noise scaling.
Implements weighted learning instead of winner-only phase.
"""

import numpy as np
import pandas as pd
import duckdb
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_instrument_config

logger = logging.getLogger(__name__)


class ImprovedTradingEnv(gym.Env):
    """Improved trading environment with rolling std gating and weighted learning."""

    def __init__(
        self,
        db_path: str = "precomputed_features.duckdb",
        episode_length: int = 1000,  # M5 bars per episode
        initial_balance: float = 10000.0,
        instrument: str = "GBPJPY",
        reward_scaling: float = 1.0,
        seed: Optional[int] = None,
        # New gating parameters
        sigma_window: int = 12,  # Rolling window for std (12 for M5)
        k_threshold: float = 0.25,  # Start lenient, can increase to 0.5
        m_spread: float = 2.0,  # Spread multiplier
        min_threshold_pips: float = 2.0,  # Absolute floor
        use_hard_gate: bool = True,  # Hard gate vs soft penalty
        gate_penalty: float = -0.01,  # Penalty for gated actions
        # Weighted learning parameters
        winner_weight: float = 1.0,  # Weight for profitable trades
        loser_weight: float = 0.2,  # Weight for losing trades (not 0!)
        weight_anneal_steps: int = 200000,  # Steps to anneal weights to 1.0
    ):
        super().__init__()

        # Get instrument config
        config = get_instrument_config(instrument)
        self.pip_value = config["pip_value"]
        self.pip_multiplier = config["pip_multiplier"]
        self.spread = config["spread"]
        # Use fixed position size from environment constraints
        from config_improved import ENV_CONSTRAINTS
        self.trade_size = ENV_CONSTRAINTS["fixed_position_size"]

        self.db_path = db_path
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling
        self.instrument = instrument

        # Gating parameters
        self.sigma_window = sigma_window
        self.k_threshold = k_threshold
        self.m_spread = m_spread
        self.min_threshold_pips = min_threshold_pips
        self.use_hard_gate = use_hard_gate
        self.gate_penalty = gate_penalty

        # Weighted learning parameters
        self.winner_weight = winner_weight
        self.loser_weight = loser_weight
        self.weight_anneal_steps = weight_anneal_steps
        self.global_timesteps = 0  # Track total timesteps for annealing

        # Connect to precomputed features database
        self.conn = duckdb.connect(db_path, read_only=True)

        # Get total number of available bars
        self.total_bars = self.conn.execute("SELECT COUNT(*) FROM m5_features").fetchone()[0]
        logger.info(f"Connected to database with {self.total_bars} M5 bars")

        # Calculate valid range for episode starts
        self.min_start = 300  # Need warm-up for indicators
        self.max_start = max(self.min_start + 1, self.total_bars - episode_length - 100)

        if self.max_start <= self.min_start:
            raise ValueError(f"Not enough data: {self.total_bars} bars, need at least {self.min_start + episode_length + 100}")

        # Trading state
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.balance = initial_balance
        self.equity = initial_balance
        self.entry_step = 0

        # Episode tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.episode_data = None
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Performance tracking
        self.profitable_trades_count = 0
        self.total_trades_count = 0
        self.accumulated_dd = 0.0
        self.peak_equity = initial_balance
        self.max_drawdown_pips = 0.0
        self.pips_from_peak = 0.0
        self._last_action = 0

        # Gating tracking
        self.gates_triggered = 0
        self.false_rejects = 0
        self.successful_gates = 0
        self.rolling_std_values = []

        # Rolling std calculation
        self.price_history = []
        self.current_rolling_std = 0.0
        self.current_threshold = self.min_threshold_pips
        self.gate_allowed = True

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold/close, 1: buy, 2: sell

        # 20 features total (17 original + 3 new gating features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
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
        self.price_history = []
        self.rolling_std_values = []

        # Sample a random starting point for the episode
        self.episode_start_idx = np.random.randint(self.min_start, self.max_start)

        # Load episode data from database
        query = f"""
        SELECT * FROM m5_features
        WHERE bar_index >= {self.episode_start_idx}
        AND bar_index < {self.episode_start_idx + self.episode_length + 100}
        ORDER BY bar_index
        """
        self.episode_data = pd.DataFrame(self.conn.execute(query).fetchall(),
                                        columns=[desc[0] for desc in self.conn.description])

        logger.debug(f"Loaded episode data: {len(self.episode_data)} bars starting at index {self.episode_start_idx}")

        return self._get_observation(), self._get_info()

    def _calculate_rolling_std(self) -> Tuple[float, float]:
        """Calculate rolling standard deviation and threshold."""
        if len(self.price_history) < self.sigma_window:
            # Not enough history, use minimum threshold
            return 0.0, self.min_threshold_pips

        # Calculate price returns in pips
        prices = np.array(self.price_history[-self.sigma_window:])
        returns_pips = np.diff(prices) * 100  # Convert to pips

        # Calculate rolling std
        if len(returns_pips) > 0:
            rolling_std = np.std(returns_pips)
        else:
            rolling_std = 0.0

        # Calculate threshold using formula
        threshold = max(
            self.k_threshold * rolling_std,
            self.m_spread * self.spread,
            self.min_threshold_pips
        )

        return rolling_std, threshold

    def _check_gate(self, action: int, recent_move_pips: float) -> Tuple[bool, float]:
        """Check if action should be gated based on recent price movement."""
        if action == 0 or self.position != 0:  # Only gate new entries
            return True, 0.0

        # Check if recent move exceeds threshold
        gate_allowed = abs(recent_move_pips) >= self.current_threshold

        if not gate_allowed:
            self.gates_triggered += 1
            if self.use_hard_gate:
                # Hard gate: block the action
                return False, self.gate_penalty
            else:
                # Soft gate: allow but with penalty
                return True, self.gate_penalty

        return True, 0.0

    def _get_observation(self) -> np.ndarray:
        """Get current observation including new gating features."""
        if self.current_step >= len(self.episode_data):
            return np.zeros(20, dtype=np.float32)

        row = self.episode_data.iloc[self.current_step]

        # Original market features
        features = [
            row['react_ratio'],
            row['bb_position'],
            row['efficiency_ratio'],
            row['rsi_extreme'],
            row['atr_ratio'],  # Will keep for compatibility but also add rolling std
            row['h1_trend'],
            row['h1_momentum'],
            row['use_mean_reversion'],
        ]

        # Position-dependent features
        current_price = row['close']

        if self.position != 0:
            pnl_pips = (current_price - self.entry_price) * self.position * 100
            bars_since_entry = min((self.current_step - self.entry_step) / 100, 5.0)
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

        # NEW: Add gating-related features
        features.extend([
            self.current_rolling_std / 10,          # 17: Rolling std (scaled)
            self.current_threshold / 10,            # 18: Current threshold (scaled)
            float(self.gate_allowed),                # 19: Gate allowed flag
        ])

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
        """Execute one step in the environment with gating logic."""
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

        # Check gate
        self.gate_allowed, gate_penalty = self._check_gate(action, recent_move_pips)

        # Apply hard gate if necessary
        if self.use_hard_gate and not self.gate_allowed:
            action = 0  # Force hold

        # Store last trade result
        self.last_trade_result = 0.0

        # Execute action
        if action == 0:  # Hold or close position
            if self.position != 0:
                # Close position
                exit_price = current_price - (self.spread * self.pip_value) * np.sign(self.position)
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

        elif action == 1:  # Buy
            if self.position <= 0:  # Not already long
                if self.position < 0:  # Close short first
                    exit_price = current_price + self.spread * self.pip_value
                    pnl = (self.entry_price - exit_price) * self.trade_size
                    self.balance += pnl
                    self.equity = self.balance

                    # Track trade result
                    trade_pips = (self.entry_price - exit_price) * 100
                    self.last_trade_result = trade_pips
                    self.episode_trades.append(trade_pips)

                    # Update trade counts
                    self.total_trades_count += 1
                    if pnl > 0:
                        self.profitable_trades_count += 1

                # Open long
                self.position = 1
                self.entry_price = current_price + self.spread * self.pip_value
                self.entry_step = self.current_step

        elif action == 2:  # Sell
            if self.position >= 0:  # Not already short
                if self.position > 0:  # Close long first
                    exit_price = current_price - self.spread * self.pip_value
                    pnl = (exit_price - self.entry_price) * self.trade_size
                    self.balance += pnl
                    self.equity = self.balance

                    # Track trade result
                    trade_pips = (exit_price - self.entry_price) * 100
                    self.last_trade_result = trade_pips
                    self.episode_trades.append(trade_pips)

                    # Update trade counts
                    self.total_trades_count += 1
                    if pnl > 0:
                        self.profitable_trades_count += 1

                # Open short
                self.position = -1
                self.entry_price = current_price - self.spread * self.pip_value
                self.entry_step = self.current_step

        # Update equity for open positions
        if self.position != 0:
            if self.position > 0:
                self.equity = self.balance + (current_price - self.entry_price) * self.trade_size
            else:
                self.equity = self.balance + (self.entry_price - current_price) * self.trade_size

        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        self.accumulated_dd += current_dd

        # Calculate reward
        reward = self._calculate_reward(prev_equity, gate_penalty)

        # Update timesteps
        self.current_step += 1
        self.global_timesteps += 1
        self._last_action = action

        # Check if episode is done
        done = self.current_step >= self.episode_length
        truncated = False

        # Add sample weight to info
        info = self._get_info()
        info['sample_weight'] = self.get_sample_weight()

        return self._get_observation(), reward, done, truncated, info

    def _get_info(self) -> Dict:
        """Get environment info including gating metrics."""
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'trades': len(self.episode_trades),
            'profitable_trades': self.profitable_trades_count,
            'total_trades': self.total_trades_count,
            'win_rate': self.profitable_trades_count / max(self.total_trades_count, 1),
            'accumulated_dd': self.accumulated_dd,
            'current_rolling_std': self.current_rolling_std,
            'current_threshold': self.current_threshold,
            'gates_triggered': self.gates_triggered,
            'gate_rate': self.gates_triggered / max(self.current_step, 1),
        }

        if self.episode_trades:
            trades_array = np.array(self.episode_trades)
            info['avg_trade'] = np.mean(trades_array)
            info['expectancy'] = np.mean(trades_array) if len(trades_array) > 0 else 0

        return info

    def close(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()