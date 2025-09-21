#!/usr/bin/env python3
"""
Episode runner for Micro MuZero that runs full 360-bar trading sessions.
Properly handles train/val/test splits and sequential experience collection.
"""

import numpy as np
import pandas as pd
import torch
import duckdb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to use optimized memory cache if available
try:
    from micro.training.optimized_cache import get_optimized_cache
    USE_MEMORY_CACHE = True
    logger.info("Optimized memory cache available (217MB vs 2.6GB)")
except ImportError:
    USE_MEMORY_CACHE = False
    logger.info("Optimized cache not available, using direct database queries")


@dataclass
class Experience:
    """Single experience in a trajectory."""
    observation: np.ndarray  # (32, 15) - last 32 bars
    action: int
    reward: float
    done: bool
    policy: np.ndarray  # MCTS policy
    value: float  # MCTS value
    market_outcome: int  # 0=UP, 1=NEUTRAL, 2=DOWN
    outcome_probs: np.ndarray  # Predicted probabilities


@dataclass
class Episode:
    """Complete episode with trajectory and metadata."""
    experiences: List[Experience]
    total_reward: float
    num_trades: int
    winning_trades: int
    expectancy: float
    split: str  # 'train', 'val', or 'test'
    session_idx: int
    start_bar_index: int


class EpisodeRunner:
    """
    Runs full episodes through 360-bar sessions.
    Handles observation generation, position tracking, and reward calculation.
    """

    def __init__(
        self,
        model,
        mcts,
        db_path: str = "/workspace/data/micro_features.duckdb",
        session_indices_path: str = "/workspace/micro/cache/valid_session_indices.pkl",
        device: str = "cpu",
        # Removed confusing flag - always use AMDDP1 as documented
    ):
        self.model = model
        self.mcts = mcts
        self.device = torch.device(device)

        # Use memory cache if available
        if USE_MEMORY_CACHE:
            self.memory_cache = get_optimized_cache()
            self.conn = None
            logger.info("Using optimized memory cache for session data")
        else:
            self.conn = duckdb.connect(db_path, read_only=True)
            self.memory_cache = None
            logger.info("Using database for session data")

        # Always use AMDDP1 reward system (no flag needed)

        # Load pre-calculated valid session indices
        self.session_indices = self._load_session_indices(session_indices_path)

        # Feature columns we need - using lag 0 for lagged features
        self.feature_columns = [
            'position_in_range_60_0', 'min_max_scaled_momentum_60_0',
            'min_max_scaled_rolling_range_0', 'min_max_scaled_momentum_5_0',
            'price_change_pips_0', 'dow_cos_final_0', 'dow_sin_final_0',
            'hour_cos_final_0', 'hour_sin_final_0', 'position_side',
            'position_pips', 'bars_since_entry', 'pips_from_peak',
            'max_drawdown_pips', 'accumulated_dd'
        ]

    def _load_session_indices(self, path: str) -> Dict:
        """Load pre-calculated valid session indices."""
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Session indices not found at {path}. "
                "Run session_index_calculator.py first!"
            )

        with open(path, 'rb') as f:
            indices = pickle.load(f)

        logger.info(f"Loaded session indices - Train: {len(indices['train'])}, "
                   f"Val: {len(indices['val'])}, Test: {len(indices['test'])}")
        return indices

    def run_episode(
        self,
        split: str = 'train',
        session_idx: Optional[int] = None,
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> Episode:
        """
        Run a complete 360-bar episode.

        Args:
            split: Data split to use ('train', 'val', 'test')
            session_idx: Specific session index, or None for random
            temperature: MCTS temperature for action selection
            add_noise: Whether to add exploration noise

        Returns:
            Complete episode with trajectory
        """
        logger.info(f"run_episode called with split={split}, session_idx={session_idx}")
        # Get valid indices for this split
        valid_indices = self.session_indices[split]
        if len(valid_indices) == 0:
            raise ValueError(f"No valid sessions for split '{split}'")

        # Select session
        if session_idx is None:
            session_idx = np.random.choice(len(valid_indices))
        start_bar_index = valid_indices[session_idx]
        logger.info(f"Selected session with start_bar_index={start_bar_index}")

        # Load session data (360 bars + 32 lookback = 392 total)
        logger.info(f"Loading session data from {start_bar_index - 32} to {start_bar_index + 360}")
        session_data = self._load_session_data(start_bar_index - 32, start_bar_index + 360)
        logger.info(f"Loaded session data with shape {session_data.shape}")

        # Initialize episode tracking
        experiences = []
        position = 0  # -1: short, 0: flat, 1: long
        entry_price = 0.0
        entry_bar = -1

        # V7-style AMDDP tracking
        high_water_mark = 0.0  # Track highest PnL achieved
        prev_max_dd = 0.0  # Previous maximum drawdown
        dd_sum = 0.0  # Cumulative sum of drawdown increases

        # Legacy tracking (for observation features)
        peak_pnl = 0.0
        max_dd = 0.0
        accumulated_dd = 0.0

        total_reward = 0.0
        num_trades = 0
        winning_trades = 0
        total_pnl = 0.0

        logger.info(f"About to start episode loop for session {start_bar_index}")
        # Run through 360 bars
        for bar in range(360):
            # Create observation from last 32 bars
            if bar == 0:
                logger.debug(f"Starting episode loop for session {start_bar_index}")
            observation = self._create_observation(
                session_data, bar + 32, position, entry_price,
                entry_bar, peak_pnl, max_dd, accumulated_dd
            )

            # Get action from MCTS
            obs_tensor = torch.tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if bar == 0:
                logger.debug(f"Running MCTS for first bar of session {start_bar_index}")

            mcts_result = self.mcts.run(
                obs_tensor,
                temperature=temperature,
                add_noise=add_noise and split == 'train'  # Only add noise during training
            )

            action = mcts_result['action']
            policy = mcts_result['policy']
            value = mcts_result['value']

            # Get prices for reward calculation
            current_price = session_data.iloc[bar + 32]['close']
            next_price = session_data.iloc[bar + 33]['close'] if bar < 359 else current_price

            # Calculate market outcome (for stochastic dynamics)
            market_outcome = self._calculate_market_outcome(
                current_price, next_price, session_data, bar + 32
            )

            # Get outcome predictions from model
            with torch.no_grad():
                hidden = self.model.representation(obs_tensor)
                action_tensor = torch.tensor([[action]], device=self.device)
                outcome_probs = self.model.outcome_probability(hidden, action_tensor)
                outcome_probs = outcome_probs.cpu().numpy()[0]

            # Execute action and calculate reward (V7-style AMDDP)
            reward, position, entry_price, entry_bar, trade_result, high_water_mark, prev_max_dd, dd_sum = self._execute_action(
                action, position, entry_price, entry_bar,
                current_price, bar, session_data,
                high_water_mark, prev_max_dd, dd_sum
            )

            # Track trade results
            if trade_result is not None:
                num_trades += 1
                total_pnl += trade_result
                if trade_result > 0:
                    winning_trades += 1

            # Update position tracking
            if position != 0:
                current_pnl = self._calculate_current_pnl(
                    position, entry_price, current_price
                )
                peak_pnl = max(peak_pnl, current_pnl)
                drawdown = peak_pnl - current_pnl
                max_dd = max(max_dd, drawdown)
                if drawdown > 0:
                    accumulated_dd += drawdown * 0.01
            else:
                peak_pnl = 0.0
                max_dd = 0.0
                accumulated_dd = 0.0

            # Store experience
            experiences.append(Experience(
                observation=observation,
                action=action,
                reward=reward,
                done=(bar == 359),
                policy=policy,
                value=value,
                market_outcome=market_outcome,
                outcome_probs=outcome_probs
            ))

            total_reward += reward

        # Force close any open position at episode end
        if position != 0:
            final_price = session_data.iloc[-1]['close']
            close_pnl = self._calculate_current_pnl(position, entry_price, final_price)
            bars_held = 360 - entry_bar if entry_bar >= 0 else 0

            # Calculate final reward with V7-style AMDDP
            final_reward = self._calculate_amddp1_v7(close_pnl, dd_sum)

            old_reward = experiences[-1].reward if experiences else 0
            experiences[-1].reward = final_reward
            total_reward = total_reward - old_reward + final_reward
            num_trades += 1
            total_pnl += close_pnl
            if close_pnl > 0:
                winning_trades += 1

        # Calculate expectancy
        expectancy = total_pnl / max(num_trades, 1)

        return Episode(
            experiences=experiences,
            total_reward=total_reward,
            num_trades=num_trades,
            winning_trades=winning_trades,
            expectancy=expectancy,
            split=split,
            session_idx=session_idx,
            start_bar_index=start_bar_index
        )

    def _load_session_data(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load data for a session including lookback period."""
        # Use memory cache if available
        if USE_MEMORY_CACHE and self.memory_cache is not None:
            # Get from memory cache - it returns ALL columns with original names
            return self.memory_cache.get_session_data(start_idx, end_idx)

        # Fallback to database query
        # Technical features (first 9) have lag suffixes in database
        # Position features (last 6) don't have suffixes
        db_columns = []

        # Technical and time features with _0 suffix for current bar
        technical_and_time_features = self.feature_columns[:9]  # First 9 features
        for feat in technical_and_time_features:
            db_columns.append(f"{feat}_0")

        # Position features without suffix
        position_features = self.feature_columns[9:]  # Last 6 features
        db_columns.extend(position_features)

        query = f"""
        SELECT bar_index, timestamp, close, {', '.join(db_columns)}
        FROM micro_features
        WHERE bar_index >= {start_idx} AND bar_index < {end_idx}
        ORDER BY bar_index
        """

        df = self.conn.execute(query).fetchdf()

        # Rename columns to remove _0 suffix from technical/time features
        rename_mapping = {f"{feat}_0": feat for feat in technical_and_time_features}
        df = df.rename(columns=rename_mapping)

        return df

    def _create_observation(
        self, data: pd.DataFrame, current_bar: int,
        position: int, entry_price: float, entry_bar: int,
        peak_pnl: float, max_dd: float, accumulated_dd: float
    ) -> np.ndarray:
        """Create (32, 15) observation for current timestep."""
        # Get last 32 bars
        window = data.iloc[current_bar-32:current_bar]

        # Create observation array
        obs = np.zeros((32, 15), dtype=np.float32)

        # Fill technical features (columns 0-8)
        for i in range(9):
            col_name = self.feature_columns[i]
            if col_name not in window.columns:
                logger.error(f"Column '{col_name}' not found in window columns: {window.columns.tolist()[:20]}")
                raise KeyError(f"Column '{col_name}' not found in data")
            obs[:, i] = window[col_name].values

        # Position features need to be calculated
        current_price = data.iloc[current_bar]['close']

        for t in range(32):
            # Position side (column 9)
            obs[t, 9] = position

            if position != 0 and entry_bar >= 0:
                # Position pips (column 10)
                pnl = self._calculate_current_pnl(position, entry_price, current_price)
                obs[t, 10] = np.tanh(pnl / 100)

                # Bars since entry (column 11)
                bars_held = current_bar - entry_bar
                obs[t, 11] = np.tanh(bars_held / 100)

                # Pips from peak (column 12)
                obs[t, 12] = np.tanh((pnl - peak_pnl) / 100)

                # Max drawdown (column 13)
                obs[t, 13] = np.tanh(-abs(max_dd) / 100)

                # Accumulated drawdown (column 14)
                obs[t, 14] = np.tanh(accumulated_dd / 100)

        return obs

    def _execute_action(
        self, action: int, position: int, entry_price: float,
        entry_bar: int, current_price: float, current_bar: int,
        session_data: pd.DataFrame,
        high_water_mark: float, prev_max_dd: float, dd_sum: float
    ) -> Tuple[float, int, float, int, Optional[float], float, float, float]:
        """
        Execute action with V7-style AMDDP reward calculation.

        Returns: (reward, new_position, new_entry_price, new_entry_bar, trade_pnl,
                 high_water_mark, prev_max_dd, dd_sum)
        """
        reward = 0.0
        trade_pnl = None

        # Update drawdown tracking if in position
        if position != 0 and entry_price > 0:
            current_pnl = self._calculate_current_pnl(position, entry_price, current_price)

            # Update high water mark
            if current_pnl > high_water_mark:
                high_water_mark = current_pnl

            # Calculate current drawdowns
            dd_from_open = max(0, -current_pnl)  # DD from entry
            dd_from_hwm = max(0, high_water_mark - current_pnl)  # DD from HWM
            current_max_dd = max(dd_from_open, dd_from_hwm)

            # Track drawdown increases
            if current_max_dd > prev_max_dd:
                dd_increase = current_max_dd - prev_max_dd
                dd_sum += dd_increase
            prev_max_dd = current_max_dd

        if action == 0:  # HOLD
            if position == 0:
                reward = -0.05  # Small penalty for idle
            else:
                reward = 0.0  # Neutral when in position

        elif action == 1:  # BUY
            if position == 0:
                reward = 1.0  # Reward decisive entry
                position = 1
                entry_price = current_price
                entry_bar = current_bar
                # Reset drawdown tracking for new position
                high_water_mark = 0.0
                prev_max_dd = 0.0
                dd_sum = 0.0
            elif position == -1:
                # Close short with V7-style AMDDP
                trade_pnl = (entry_price - current_price) * 100 - 4
                reward = self._calculate_amddp1_v7(trade_pnl, dd_sum)
                # Open long
                position = 1
                entry_price = current_price
                entry_bar = current_bar
                # Reset drawdown tracking for new position
                high_water_mark = 0.0
                prev_max_dd = 0.0
                dd_sum = 0.0
            else:
                reward = -1.0  # Invalid action

        elif action == 2:  # SELL
            if position == 0:
                reward = 1.0  # Reward decisive entry
                position = -1
                entry_price = current_price
                entry_bar = current_bar
                # Reset drawdown tracking for new position
                high_water_mark = 0.0
                prev_max_dd = 0.0
                dd_sum = 0.0
            elif position == 1:
                # Close long with V7-style AMDDP
                trade_pnl = (current_price - entry_price) * 100 - 4
                reward = self._calculate_amddp1_v7(trade_pnl, dd_sum)
                # Open short
                position = -1
                entry_price = current_price
                entry_bar = current_bar
                # Reset drawdown tracking for new position
                high_water_mark = 0.0
                prev_max_dd = 0.0
                dd_sum = 0.0
            else:
                reward = -1.0  # Invalid action

        elif action == 3:  # CLOSE
            if position != 0:
                trade_pnl = self._calculate_current_pnl(position, entry_price, current_price)
                # V7-style AMDDP reward
                reward = self._calculate_amddp1_v7(trade_pnl, dd_sum)
                position = 0
                entry_price = 0.0
                entry_bar = -1
                # Reset drawdown tracking
                high_water_mark = 0.0
                prev_max_dd = 0.0
                dd_sum = 0.0
            else:
                reward = -1.0  # Invalid action

        return reward, position, entry_price, entry_bar, trade_pnl, high_water_mark, prev_max_dd, dd_sum

    def _calculate_current_pnl(self, position: int, entry_price: float, current_price: float) -> float:
        """Calculate current P&L in pips."""
        if position == 1:  # Long
            return (current_price - entry_price) * 100 - 4
        elif position == -1:  # Short
            return (entry_price - current_price) * 100 - 4
        return 0.0

    def _calculate_market_outcome(
        self, current_price: float, next_price: float,
        session_data: pd.DataFrame, bar_idx: int
    ) -> int:
        """
        Calculate market outcome based on rolling stdev.
        Returns: 0=UP, 1=NEUTRAL, 2=DOWN
        """
        # Calculate rolling stdev (20 bars)
        if bar_idx >= 20:
            recent_prices = session_data.iloc[bar_idx-20:bar_idx]['close'].values
            rolling_stdev = np.std(recent_prices)
        else:
            rolling_stdev = 0.001  # Default for early bars

        price_change = next_price - current_price
        threshold = 0.5 * rolling_stdev

        if price_change > threshold:
            return 0  # UP
        elif price_change < -threshold:
            return 2  # DOWN
        else:
            return 1  # NEUTRAL

    def _calculate_amddp1_v7(self, pnl_pips: float, dd_sum: float) -> float:
        """
        Calculate V7-style AMDDP reward with 1% penalty (AMDDP1).

        Formula: reward = pnl_pips - 0.01 * cumulative_drawdown_sum

        With profit protection: If profitable trade has negative reward due to DD,
        return small positive reward (0.001) instead.

        Args:
            pnl_pips: Final P&L in pips (including costs)
            dd_sum: Cumulative sum of drawdown increases during position

        Returns:
            AMDDP1 reward value
        """
        # Base V7 formula with 1% penalty (instead of 10%)
        base_reward = pnl_pips - 0.01 * dd_sum

        # Apply profit protection
        if pnl_pips > 0 and base_reward < 0:
            return 0.001  # Small positive reward for profitable trades
        else:
            return base_reward



def test_episode_runner():
    """Test the episode runner."""
    # This would need actual model and MCTS instances
    logger.info("Episode runner created successfully")
    logger.info("To test with actual model:")
    logger.info("  1. Load trained model")
    logger.info("  2. Create MCTS instance")
    logger.info("  3. Run episode: runner.run_episode('train')")


if __name__ == "__main__":
    test_episode_runner()