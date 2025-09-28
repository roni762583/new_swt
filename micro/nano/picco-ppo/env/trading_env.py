#!/usr/bin/env python3
"""
Optimal M5/H1 Trading Environment for PPO.

Features our discovered optimal 7 market features + 6 position features.
Based on empirical analysis showing M5/H1 achieves 444.6 pips with 38.6% win rate.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import duckdb
from typing import Dict, Tuple, Optional, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment with M5 execution and H1 context.

    State Space (13 features):
    - 7 market features (optimal set from analysis)
    - 6 position features

    Action Space:
    - 0: Hold
    - 1: Buy (Long)
    - 2: Sell (Short)
    - 3: Close position
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        db_path: str = "/app/data/master.duckdb",
        start_idx: int = 100000,
        end_idx: int = 200000,
        initial_balance: float = 10000.0,
        pip_multiplier: float = 100.0,  # JPY pairs
        max_position_size: float = 1.0,
        transaction_cost: float = 4.0,  # FULL 4 pip spread from start - NO curriculum
        max_episode_steps: int = 1000,
        reward_scaling: float = 0.01,
        render_mode: Optional[str] = None
    ):
        """Initialize trading environment."""
        super().__init__()

        self.db_path = db_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.initial_balance = initial_balance
        self.pip_multiplier = pip_multiplier
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost  # FULL 4 pip spread always
        self.profitable_trades_count = 0  # Count profitable trades for learning threshold
        self.total_trades_count = 0  # Total trades counter
        self.max_episode_steps = max_episode_steps
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)

        # State: 7 market + 6 position + 4 time features = 17 total
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )

        # Load and prepare data
        self._load_data()
        self._prepare_features()

        # Episode tracking
        self.current_step = 0
        self.episode_trades = []
        self.last_trade_result = 0.0
        self.ignore_losses = True  # Start by ignoring losses until 1000 profitable trades

    def _load_data(self):
        """Load M1 data and aggregate to M5 and H1."""
        conn = duckdb.connect(self.db_path, read_only=True)

        # Load M1 data
        query = f"""
        SELECT bar_index, timestamp, open, high, low, close, volume
        FROM master
        WHERE bar_index BETWEEN {self.start_idx} AND {self.end_idx}
        ORDER BY bar_index
        """

        self.m1_data = pd.read_sql(query, conn)
        conn.close()

        # Aggregate to M5
        self.m5_data = self._aggregate_bars(self.m1_data, 5)

        # Aggregate to H1
        self.h1_data = self._aggregate_bars(self.m1_data, 60)

        logger.info(f"Loaded {len(self.m5_data)} M5 bars, {len(self.h1_data)} H1 bars")

    def _aggregate_bars(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Aggregate M1 bars to higher timeframe."""
        data['group'] = data['bar_index'] // period

        agg = data.groupby('group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'bar_index': 'first',
            'timestamp': 'first'
        }).reset_index(drop=True)

        return agg

    def _prepare_features(self):
        """Calculate all features for M5 and H1 data."""

        # M5 features
        df = self.m5_data.copy()

        # Basic SMAs
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma200'] = df['close'].rolling(200).mean()

        # ATR for normalization
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()

        # 1. React Ratio: (close - sma200) / (sma20 - sma200)
        df['reactive'] = df['close'] - df['sma200']
        df['lessreactive'] = df['sma20'] - df['sma200']
        df['react_ratio'] = df['reactive'] / (df['lessreactive'] + 0.0001)
        df['react_ratio'] = df['react_ratio'].clip(-5, 5)

        # 2. Efficiency Ratio (Kaufman's)
        direction = (df['close'] - df['close'].shift(10)).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = direction / (volatility + 0.0001)

        # 3. Bollinger Band Position
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = (df['close'] - df['sma20']) / (bb_std * 2 + 0.0001)
        df['bb_position'] = df['bb_position'].clip(-1, 1)

        # 4. RSI Extreme
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['rsi_extreme'] = (df['rsi'] - 50) / 50

        # 5. ATR Ratio (volatility regime)
        df['atr_ratio'] = df['atr'] / (df['sma20'] + 0.0001)

        self.m5_features = df

        # H1 features
        h1 = self.h1_data.copy()
        h1['h1_sma20'] = h1['close'].rolling(20).mean()

        # H1 Trend (-1, 0, 1)
        h1['h1_trend'] = 0
        h1.loc[h1['close'] > h1['h1_sma20'] * 1.001, 'h1_trend'] = 1
        h1.loc[h1['close'] < h1['h1_sma20'] * 0.999, 'h1_trend'] = -1

        # H1 Momentum (5-bar ROC)
        h1['h1_momentum'] = h1['close'].pct_change(5).fillna(0)

        self.h1_features = h1

        # Map H1 features to M5 bars
        self._map_h1_to_m5()

    def _map_h1_to_m5(self):
        """Map H1 features to corresponding M5 bars."""
        self.m5_features['h1_trend'] = 0.0
        self.m5_features['h1_momentum'] = 0.0

        for i in range(len(self.m5_features)):
            m5_bar = self.m5_features.iloc[i]['bar_index']

            # Find corresponding H1 bar
            h1_idx = self.h1_features[
                self.h1_features['bar_index'] <= m5_bar
            ].index

            if len(h1_idx) > 0:
                h1_idx = h1_idx[-1]
                self.m5_features.loc[i, 'h1_trend'] = self.h1_features.loc[h1_idx, 'h1_trend']
                self.m5_features.loc[i, 'h1_momentum'] = self.h1_features.loc[h1_idx, 'h1_momentum']

        # Calculate use_mean_reversion flag
        self.m5_features['use_mean_reversion'] = (
            self.m5_features['efficiency_ratio'] < 0.3
        ).astype(float)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset position state (6 features from micro)
        self.position_side = 0.0  # -1 short, 0 flat, 1 long
        self.position_pips = 0.0  # Current unrealized P&L
        self.bars_since_entry = 0.0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.pips_from_peak = 0.0  # Distance from peak
        self.max_drawdown_pips = 0.0  # Max DD in position
        self.accumulated_dd = 0.0  # Cumulative DD
        self.peak_pips = 0.0  # Track peak for pips_from_peak
        self._last_action = 0  # Track last action for reward bonus

        # Reset account
        self.balance = self.initial_balance
        self.equity = self.initial_balance

        # Reset episode tracking
        self.current_step = np.random.randint(300, len(self.m5_features) - self.max_episode_steps - 100)
        self.start_step = self.current_step
        self.episode_trades = []
        self.last_trade_result = 0.0

        # Curriculum learning: gradually increase spread
        self.curriculum_episodes += 1
        if self.curriculum_episodes % 10 == 0:  # Every 10 episodes
            # Gradually increase spread from 0 to 4 pips over 100 episodes
            progress = min(1.0, self.curriculum_episodes / 100)
            self.transaction_cost = self.base_transaction_cost * progress
            if self.curriculum_episodes % 50 == 0:
                print(f"📈 Curriculum: Spread now {self.transaction_cost:.1f} pips (Episode {self.curriculum_episodes})")

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""

        # Store previous equity
        prev_equity = self.equity

        # Execute action
        self._execute_action(action)

        # Update position tracking
        self._update_position_tracking()

        # Calculate reward
        reward = self._calculate_reward(prev_equity)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step - self.start_step >= self.max_episode_steps

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int):
        """Execute trading action."""
        # Store action for reward calculation
        self._last_action = action

        current_price = self.m5_features.iloc[self.current_step]['close']

        if action == 0:  # Hold
            pass

        elif action == 1:  # Buy
            if self.position_side <= 0:  # Not long
                # Close short if exists
                if self.position_side < 0:
                    self._close_position(current_price)

                # Open long (pay spread cost)
                self.position_side = 1.0
                self.entry_price = current_price
                self.entry_bar = self.current_step
                self.bars_since_entry = 0.0
                self.position_pips = 0.0
                self.peak_pips = 0.0
                self.pips_from_peak = 0.0
                self.max_drawdown_pips = 0.0
                self.balance -= self.transaction_cost  # Pay FULL 4 pip spread
                self.accumulated_dd += self.transaction_cost  # Spread counts as DD

                self.episode_trades.append({
                    'action': 'BUY',
                    'price': current_price,
                    'bar': self.current_step
                })

                # Debug: log first few trades
                if len(self.episode_trades) <= 3:
                    print(f"     TRADE: BUY at {current_price:.5f}, step {self.current_step}")

        elif action == 2:  # Sell
            if self.position_side >= 0:  # Not short
                # Close long if exists
                if self.position_side > 0:
                    self._close_position(current_price)

                # Open short (pay spread cost)
                self.position_side = -1.0
                self.entry_price = current_price
                self.entry_bar = self.current_step
                self.bars_since_entry = 0.0
                self.position_pips = 0.0
                self.peak_pips = 0.0
                self.pips_from_peak = 0.0
                self.max_drawdown_pips = 0.0
                self.balance -= self.transaction_cost  # Pay FULL 4 pip spread
                self.accumulated_dd += self.transaction_cost  # Spread counts as DD

                self.episode_trades.append({
                    'action': 'SELL',
                    'price': current_price,
                    'bar': self.current_step
                })

        elif action == 3:  # Close
            if self.position_side != 0:
                self._close_position(current_price)

    def _close_position(self, current_price: float):
        """Close current position."""
        if self.position_side == 0:
            return

        # Calculate P&L
        if self.position_side > 0:  # Long
            pips = (current_price - self.entry_price) * self.pip_multiplier
        else:  # Short
            pips = (self.entry_price - current_price) * self.pip_multiplier

        # Update balance (no additional cost on close)
        self.balance += pips

        # Track profitable trades for learning threshold
        self.total_trades_count += 1
        if pips > 0:
            self.profitable_trades_count += 1

        # Update accumulated DD
        if pips < 0:
            self.accumulated_dd += abs(pips)

        # Store the trade result for info dict
        self.last_trade_result = pips

        # Reset position
        self.position_side = 0.0
        self.position_pips = 0.0
        self.bars_since_entry = 0.0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.peak_pips = 0.0
        self.pips_from_peak = 0.0
        self.max_drawdown_pips = 0.0

        self.episode_trades.append({
            'action': 'CLOSE',
            'price': current_price,
            'bar': self.current_step,
            'pips': pips
        })

    def _update_position_tracking(self):
        """Update position-related features."""
        current_price = self.m5_features.iloc[self.current_step]['close']

        # Calculate unrealized P&L
        if self.position_side != 0:
            if self.position_side > 0:
                self.position_pips = (current_price - self.entry_price) * self.pip_multiplier
            else:
                self.position_pips = (self.entry_price - current_price) * self.pip_multiplier

            # Update bars since entry
            self.bars_since_entry = self.current_step - self.entry_bar

            # Update peak and drawdown tracking
            self.peak_pips = max(self.peak_pips, self.position_pips)
            self.pips_from_peak = self.peak_pips - self.position_pips

            # Update max drawdown for this position
            if self.position_pips < 0:
                self.max_drawdown_pips = max(self.max_drawdown_pips, abs(self.position_pips))
        else:
            self.position_pips = 0.0

        # Update equity
        self.equity = self.balance + self.position_pips

    def _calculate_reward(self, prev_equity: float) -> float:
        """Calculate reward focusing ONLY on profitable trades initially.

        Phase 1 (< 1000 profitable trades):
        - Profitable trades get full positive reward
        - Losing trades get ZERO reward (ignored for learning)
        - No penalties for losses

        Phase 2 (>= 1000 profitable trades):
        - Normal reward calculation with both profits and losses
        """
        # Calculate PnL change in pips
        pnl_change = self.equity - prev_equity

        # Check if we just closed a trade
        trade_just_closed = (self.last_trade_result != 0)

        if self.ignore_losses and self.profitable_trades_count < 1000:
            # PHASE 1: Learn from winners only
            if trade_just_closed:
                if self.last_trade_result > 0:
                    # Profitable trade - give positive reward
                    reward = self.last_trade_result * self.reward_scaling
                else:
                    # Losing trade - ZERO reward (ignore completely)
                    reward = 0.0
            else:
                # No trade closed - neutral/small reward to encourage exploration
                reward = 0.001 * self.reward_scaling

        else:
            # PHASE 2: Normal reward calculation (after 1000 profitable trades)
            self.ignore_losses = False

            # Standard AMDDP1 reward
            amddp1_reward = pnl_change - 0.005 * self.accumulated_dd

            # Scale for stable training
            reward = amddp1_reward * self.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""

        # Get market features (7)
        current = self.m5_features.iloc[self.current_step]

        market_features = np.array([
            current['react_ratio'],
            current['h1_trend'],
            current['h1_momentum'],
            current['efficiency_ratio'],
            current['bb_position'],
            current['rsi_extreme'],
            current['atr_ratio']
        ], dtype=np.float32)

        # Position features (6 - correct from micro)
        position_features = np.array([
            self.position_side,  # -1, 0, 1
            self.position_pips / 100.0,  # Normalize
            self.bars_since_entry / 100.0,  # Normalize
            self.pips_from_peak / 100.0,  # Normalize
            self.max_drawdown_pips / 100.0,  # Normalize
            self.accumulated_dd / 1000.0  # Normalize cumulative DD
        ], dtype=np.float32)

        # Time features (4) - cyclic encoding for 120hr trading week
        timestamp = self.m5_features.iloc[self.current_step]['timestamp']
        hour_of_day = timestamp.hour + timestamp.minute / 60.0  # Fractional hours

        # Calculate hour of trading week (Sun 5pm = 0, Fri 5pm = 120)
        # Sunday = 6, Monday = 0 in pandas
        day_of_week = timestamp.weekday()
        if day_of_week == 6:  # Sunday
            hour_of_week = hour_of_day - 17  # Sunday 5pm = hour 0
            if hour_of_week < 0:
                hour_of_week = 0  # Before market open
        else:
            # Monday = 0 maps to hours 7-31 (5pm Sun + 24hrs)
            hour_of_week = 7 + day_of_week * 24 + hour_of_day - 17

        hour_of_week = hour_of_week % 120  # Wrap to 120hr week

        time_features = np.array([
            np.sin(2 * np.pi * hour_of_day / 24),  # sin(hour of day)
            np.cos(2 * np.pi * hour_of_day / 24),  # cos(hour of day)
            np.sin(2 * np.pi * hour_of_week / 120),  # sin(hour of week)
            np.cos(2 * np.pi * hour_of_week / 120)   # cos(hour of week)
        ], dtype=np.float32)

        # Combine all features
        obs = np.concatenate([market_features, position_features, time_features])

        return obs

    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate on bankruptcy
        if self.equity <= self.initial_balance * 0.5:
            return True

        # Terminate on huge profit
        if self.equity >= self.initial_balance * 2.0:
            return True

        return False

    def _get_info(self) -> Dict:
        """Get environment info."""
        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position': self.position_side,
            'trades': len(self.episode_trades),
            'current_bar': self.current_step,
            'max_drawdown': self.max_drawdown_pips,
            'profitable_trades': self.profitable_trades_count,
            'total_trades': self.total_trades_count,
            'learning_phase': 'winners_only' if self.ignore_losses else 'full_learning'
        }

        # Include trade result if a trade just closed
        if self.last_trade_result != 0:
            info['trade_result'] = self.last_trade_result
            self.last_trade_result = 0.0  # Reset after reporting

        return info

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            current = self.m5_features.iloc[self.current_step]
            print(f"\n=== Step {self.current_step - self.start_step} ===")
            print(f"Price: {current['close']:.3f}")
            print(f"Position: {self.position_side}")
            print(f"Equity: ${self.equity:.2f}")
            print(f"Trades: {len(self.episode_trades)}")
            print(f"Efficiency Ratio: {current['efficiency_ratio']:.3f}")
            print(f"React Ratio: {current['react_ratio']:.3f}")
            print(f"H1 Trend: {current['h1_trend']}")

    def close(self):
        """Clean up environment."""
        pass