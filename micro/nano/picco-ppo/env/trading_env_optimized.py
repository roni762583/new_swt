"""
Optimized Trading Environment using precomputed features from DuckDB.
Efficiently loads only the data slice needed for each episode.
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
import json
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_instrument_config

logger = logging.getLogger(__name__)


class OptimizedTradingEnv(gym.Env):
    """Trading environment that queries precomputed features from DuckDB."""

    def __init__(
        self,
        db_path: str = "master.duckdb",
        episode_length: int = 360,  # M1 bars per episode (360 M1 bars = 6 hours)
        initial_balance: float = 10000.0,
        instrument: str = "GBPJPY",
        reward_scaling: float = 1.0,
        seed: Optional[int] = None,
        enable_phase1_learning: bool = False  # NEW: Phase1 disabled by default
    ):
        super().__init__()

        # Get instrument config
        config = get_instrument_config(instrument)
        self.pip_value = config["pip_value"]
        self.pip_multiplier = config["pip_multiplier"]
        self.spread = config["spread"]
        self.trade_size = config["trade_size"]

        self.db_path = db_path
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling
        self.instrument = instrument

        # Connect to master database (M1 data with all features)
        self.conn = duckdb.connect(db_path, read_only=True)

        # Get total number of available bars
        self.total_bars = self.conn.execute("SELECT COUNT(*) FROM master").fetchone()[0]
        logger.info(f"Connected to database with {self.total_bars:,} M1 bars")

        # Calculate valid range for episode starts (need room for episode + warm-up)
        self.min_start = 300  # Need warm-up for indicators
        # Ensure we have enough room for full episode
        self.max_start = max(self.min_start + 1, self.total_bars - episode_length - 100)  # Buffer at end

        if self.max_start <= self.min_start:
            raise ValueError(f"Not enough data: {self.total_bars} bars, need at least {self.min_start + episode_length + 100}")

        # Trading state
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.balance = initial_balance
        self.equity = initial_balance

        # Episode tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.episode_data = None
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Winner-focused learning (controlled by flag)
        self.enable_phase1 = enable_phase1_learning
        self.profitable_trades_count = 0
        self.total_trades_count = 0
        self.ignore_losses = enable_phase1_learning  # Only ignore losses if phase1 enabled

        # Performance tracking
        self.accumulated_dd = 0.0
        self.peak_pips = 0.0
        self._last_action = 0

        # Rolling expectancy tracking (class-level shared across all envs)
        if not hasattr(OptimizedTradingEnv, '_all_trades'):
            OptimizedTradingEnv._all_trades = deque(maxlen=10000)  # Keep last 10K trades
            OptimizedTradingEnv._expectancy_log_path = "results/rolling_expectancy.json"
            os.makedirs("results", exist_ok=True)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: hold, 1: buy, 2: sell, 3: close

        # 32 features total (26 ML features + 6 position features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )

        # Set seed
        if seed is not None:
            self.seed(seed)

    def __getstate__(self):
        """Prepare environment for pickling (close DB connection)."""
        state = self.__dict__.copy()
        # Remove the unpickleable database connection
        if 'conn' in state:
            state['conn'].close()
            del state['conn']
        return state

    def __setstate__(self, state):
        """Restore environment after unpickling (reopen DB connection)."""
        self.__dict__.update(state)
        # Reconnect to database
        self.conn = duckdb.connect(self.db_path, read_only=True)

    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]

    def _load_episode_data(self, start_idx: int) -> pd.DataFrame:
        """Load data slice for current episode from database."""
        # Query based on row position rather than bar_index
        query = f"""
        SELECT *
        FROM master
        ORDER BY bar_index
        LIMIT {self.episode_length}
        OFFSET {start_idx}
        """

        data = pd.read_sql(query, self.conn)

        # If we got less data than expected, try a different starting point
        if len(data) < self.episode_length:
            logger.debug(f"Got {len(data)} bars instead of {self.episode_length}, adjusting start position")
            # Try starting earlier
            new_start = max(0, self.total_bars - self.episode_length - 100)
            query = f"""
            SELECT *
            FROM master
            ORDER BY bar_index
            LIMIT {self.episode_length}
            OFFSET {new_start}
            """
            data = pd.read_sql(query, self.conn)

        return data.reset_index(drop=True)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.seed(seed)

        # Select random starting point for episode
        self.episode_start_idx = np.random.randint(self.min_start, self.max_start)

        # Load data for this episode
        self.episode_data = self._load_episode_data(self.episode_start_idx)

        if len(self.episode_data) < self.episode_length:
            # Use a safe starting point that guarantees enough data
            safe_start = min(self.min_start, self.total_bars - self.episode_length - 10)
            self.episode_start_idx = safe_start
            self.episode_data = self._load_episode_data(self.episode_start_idx)

            # If still not enough data, reduce episode length for this episode
            if len(self.episode_data) < self.episode_length:
                logger.debug(f"Adjusting episode length from {self.episode_length} to {len(self.episode_data)}")
                # Temporarily use the available data length
                self.current_episode_length = len(self.episode_data) - 1
            else:
                self.current_episode_length = self.episode_length
        else:
            self.current_episode_length = self.episode_length

        # Reset trading state
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        # Reset episode tracking
        self.current_step = 0
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Reset performance tracking
        self.accumulated_dd = 0.0
        self.peak_pips = 0.0
        self._last_action = 0

        # Check if we should still ignore losses
        if self.profitable_trades_count >= 1000:
            self.ignore_losses = False
            logger.info(f"Phase 2 activated: Now learning from all trades")

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector from M1 master.duckdb."""
        if self.current_step >= len(self.episode_data):
            # Return zeros if we've somehow gone past the end
            return np.zeros(32, dtype=np.float32)

        row = self.episode_data.iloc[self.current_step]

        # Extract features that ACTUALLY exist in master.duckdb
        features = [
            row['log_return_1m'],                        # 0: 1-min momentum
            row['log_return_5m'],                        # 1: 5-min momentum
            row['log_return_60m'],                       # 2: 60-min trend alignment
            row['efficiency_ratio_h1'],                  # 3: H1 efficiency (trending vs choppy)
            row['momentum_strength_10_zsarctan_w20'],    # 4: 10-bar momentum z-score
            row['atr_14'],                               # 5: Short-term volatility
            row['atr_14_zsarctan_w20'],                  # 6: ATR z-score (volatility extremes)
            row['vol_ratio_deviation'],                  # 7: Volatility regime
            row['realized_vol_60_zsarctan_w20'],         # 8: H1 volatility z-score
            row['h1_swing_range_position'],              # 9: Position within H1 swing range
            row['swing_point_range'],                    # 10: H1 range magnitude
            row['high_swing_slope_h1'],                  # 11: H1 swing high slope
            row['low_swing_slope_h1'],                   # 12: H1 swing low slope
            row['h1_trend_slope_zsarctan'],              # 13: H1 trend strength z-score
            row['h1_swing_range_position_zsarctan_w20'], # 14: Range position z-score
            row['swing_point_range_zsarctan_w20'],       # 15: Range magnitude z-score
            row['high_swing_slope_h1_zsarctan'],         # 16: High slope z-score
            row['low_swing_slope_h1_zsarctan'],          # 17: Low slope z-score
            row['high_swing_slope_m1_zsarctan_w20'],     # 18: M1 high slope z-score
            row['low_swing_slope_m1_zsarctan_w20'],      # 19: M1 low slope z-score
            row['combo_geometric'],                      # 20: Interaction feature
            row['bb_position'],                          # 21: Bollinger Bands position
            row['hour_sin'],                             # 22: Hour cyclical (sin)
            row['hour_cos'],                             # 23: Hour cyclical (cos)
            row['dow_sin'],                              # 24: Day of week (sin)
            row['dow_cos'],                              # 25: Day of week (cos)
        ]

        # Handle NaN/NULL values - replace with 0.0
        features = [0.0 if pd.isna(f) or np.isinf(f) else float(f) for f in features]

        # Add position-dependent features (these change with position)
        current_price = row['close']
        atr = row['atr_14'] if (not pd.isna(row['atr_14']) and row['atr_14'] > 0) else 0.0001

        if self.position != 0:
            pnl_pips = (current_price - self.entry_price) * self.position * 100  # GBPJPY: 1 pip = 0.01
            pnl_norm = pnl_pips / (atr * 10000) if atr > 0 else 0
            time_in_pos = min((self.current_step - self.entry_step) / 100, 5.0)
        else:
            pnl_pips = 0.0
            pnl_norm = 0.0
            time_in_pos = 0.0

        position_features = [
            float(self.position),       # 26: Current position
            pnl_pips / 100,             # 27: PnL in pips (scaled)
            pnl_norm,                   # 28: PnL normalized by ATR
            time_in_pos,                # 29: Time in position
            self.accumulated_dd / 100,  # 30: Accumulated DD
            self.peak_pips / 100,       # 31: Pips from peak
        ]

        # Handle NaN/Inf in position features as well
        position_features = [0.0 if pd.isna(f) or np.isinf(f) else float(f) for f in position_features]
        features.extend(position_features)

        return np.array(features, dtype=np.float32)

    def _update_expectancy_log(self):
        """Update rolling expectancy log file."""
        try:
            all_trades = list(OptimizedTradingEnv._all_trades)
            if len(all_trades) == 0:
                return

            # Calculate rolling windows
            data = {"episode": self.episode_start_idx // self.episode_length}

            for window in [100, 500, 1000]:
                if len(all_trades) >= window:
                    recent = all_trades[-window:]
                    avg_pips = np.mean(recent)
                    win_rate = sum(1 for t in recent if t > 0) / len(recent) * 100
                    # R-value: use ATR as risk unit (assume 12 pips avg)
                    R_value = 12.0
                    expectancy_R = avg_pips / R_value if R_value > 0 else 0

                    data[f'expectancy_R_{window}'] = expectancy_R
                    data[f'sample_size_{window}'] = len(recent)
                    data[f'win_rate_{window}'] = win_rate
                    data[f'avg_pips_{window}'] = avg_pips
                    data[f'R_value_{window}'] = R_value

            # Lifetime stats
            data['total_trades'] = len(all_trades)
            data['cumulative_pips'] = sum(all_trades)
            data['lifetime_expectancy_R'] = np.mean(all_trades) / 12.0
            data['lifetime_trades'] = len(all_trades)
            data['lifetime_win_rate'] = sum(1 for t in all_trades if t > 0) / len(all_trades) * 100

            # Write to file
            with open(OptimizedTradingEnv._expectancy_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to update expectancy log: {e}")

    def _calculate_reward(self, prev_equity: float) -> float:
        """Calculate reward based on winner-focused learning strategy."""
        pnl_change = (self.equity - prev_equity) / self.initial_balance * 100

        # Check if we're in phase 1 (learning from winners only)
        if self.ignore_losses and self.profitable_trades_count < 1000:
            # PHASE 1: Learn from winners only
            if self.last_trade_result != 0:  # Trade just closed
                if self.last_trade_result > 0:
                    # Profitable trade - give reward
                    reward = self.last_trade_result * self.reward_scaling
                else:
                    # Losing trade - ignore it
                    reward = 0.0
            else:
                # No trade closed - small penalty for time
                reward = -0.001
        else:
            # PHASE 2: Normal AMDDP1 reward (1% DD penalty)
            reward = pnl_change - 0.01 * self.accumulated_dd

            # Bonus for closing profitable trades
            if self.last_trade_result > 0:
                reward += 0.5

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.episode_data) - 1:
            return self._get_observation(), 0.0, False, True, self._get_info()

        row = self.episode_data.iloc[self.current_step]
        current_price = row['close']
        prev_equity = self.equity

        # Store last trade result (will be set if trade closes)
        self.last_trade_result = 0.0

        # Execute action: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
        if action == 0:  # HOLD
            pass  # Do nothing, maintain current position

        elif action == 1:  # BUY
            if self.position == 0:  # Only open if flat
                self.position = 1
                self.entry_price = current_price + self.spread * self.pip_value
                self.entry_step = self.current_step

        elif action == 2:  # SELL
            if self.position == 0:  # Only open if flat
                self.position = -1
                self.entry_price = current_price - self.spread * self.pip_value
                self.entry_step = self.current_step

        elif action == 3:  # CLOSE
            if self.position != 0:  # Only close if we have a position
                # Calculate exit price with spread
                exit_price = current_price - (self.spread * self.pip_value) * np.sign(self.position)
                pnl = (exit_price - self.entry_price) * self.position * self.trade_size
                self.balance += pnl
                self.equity = self.balance

                # Track trade result
                self.last_trade_result = pnl / self.initial_balance * 100
                self.episode_trades.append(self.last_trade_result)

                # Update trade counts
                self.total_trades_count += 1
                if pnl > 0:
                    self.profitable_trades_count += 1

                # Log trade result
                trade_pips = pnl / 100  # Convert to pips
                if self.total_trades_count <= 10 or self.total_trades_count % 100 == 0:
                    status = "✅" if pnl > 0 else "❌"
                    logger.info(f"{status} Trade #{self.total_trades_count}: {trade_pips:.1f} pips | "
                              f"Profitable: {self.profitable_trades_count}/{self.total_trades_count} | "
                              f"Phase: {'1-Winners' if self.ignore_losses else '2-All'}")

                # Update rolling expectancy (class-level tracking)
                OptimizedTradingEnv._all_trades.append(trade_pips)
                if len(OptimizedTradingEnv._all_trades) % 10 == 0:  # Update log every 10 trades
                    self._update_expectancy_log()

                # Reset position
                self.position = 0
                self.entry_price = 0.0
                self.entry_step = 0

        # Update equity with open position
        if self.position != 0:
            mark_price = current_price - (self.spread * 0.01) * np.sign(self.position)  # GBPJPY pip
            open_pnl = (mark_price - self.entry_price) * self.position * self.trade_size
            self.equity = self.balance + open_pnl

        # Update performance metrics
        pips_change = (self.equity - prev_equity) / self.trade_size * 10000
        self.accumulated_dd = max(0, self.accumulated_dd - pips_change)
        self.peak_pips = max(0, self.peak_pips - pips_change)

        # Store last action
        self._last_action = action

        # Calculate reward
        reward = self._calculate_reward(prev_equity)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = bool(self.equity <= self.initial_balance * 0.5)  # 50% loss
        truncated = bool(self.current_step >= self.current_episode_length - 1)

        # Get next observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        win_rate = 0.0
        if len(self.episode_trades) > 0:
            win_rate = sum(1 for t in self.episode_trades if t > 0) / len(self.episode_trades)

        expectancy = np.mean(self.episode_trades) if self.episode_trades else 0.0

        return {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position,
            'num_trades': len(self.episode_trades),
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profitable_trades_total': self.profitable_trades_count,
            'total_trades': self.total_trades_count,
            'learning_phase': 1 if self.ignore_losses else 2,
        }

    def close(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()