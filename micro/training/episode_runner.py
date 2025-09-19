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
        use_amddp10: bool = True  # Use enhanced AMDDP10 reward system
    ):
        self.model = model
        self.mcts = mcts
        self.device = torch.device(device)
        self.conn = duckdb.connect(db_path, read_only=True)
        self.use_amddp10 = use_amddp10

        # Load pre-calculated valid session indices
        self.session_indices = self._load_session_indices(session_indices_path)

        # Feature columns we need
        self.feature_columns = [
            'position_in_range_60', 'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
            'price_change_pips', 'dow_cos_final', 'dow_sin_final',
            'hour_cos_final', 'hour_sin_final', 'position_side',
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
        # Get valid indices for this split
        valid_indices = self.session_indices[split]
        if len(valid_indices) == 0:
            raise ValueError(f"No valid sessions for split '{split}'")

        # Select session
        if session_idx is None:
            session_idx = np.random.choice(len(valid_indices))
        start_bar_index = valid_indices[session_idx]

        # Load session data (360 bars + 32 lookback = 392 total)
        session_data = self._load_session_data(start_bar_index - 32, start_bar_index + 360)

        # Initialize episode tracking
        experiences = []
        position = 0  # -1: short, 0: flat, 1: long
        entry_price = 0.0
        entry_bar = -1
        peak_pnl = 0.0
        max_dd = 0.0
        accumulated_dd = 0.0

        total_reward = 0.0
        num_trades = 0
        winning_trades = 0
        total_pnl = 0.0

        # Run through 360 bars
        for bar in range(360):
            # Create observation from last 32 bars
            observation = self._create_observation(
                session_data, bar + 32, position, entry_price,
                entry_bar, peak_pnl, max_dd, accumulated_dd
            )

            # Get action from MCTS
            obs_tensor = torch.tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

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

            # Execute action and calculate reward
            reward, position, entry_price, entry_bar, trade_result = self._execute_action(
                action, position, entry_price, entry_bar,
                current_price, bar, session_data, max_dd, bar - entry_bar if entry_bar >= 0 else 0
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

            if self.use_amddp10:
                final_reward = self._calculate_amddp10(close_pnl, max_dd, bars_held, position)
            else:
                final_reward = self._calculate_amddp1(close_pnl)

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
            obs[:, i] = window[self.feature_columns[i]].values

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
        session_data: pd.DataFrame, max_dd: float = 0.0, bars_held: int = 0
    ) -> Tuple[float, int, float, int, Optional[float]]:
        """
        Execute action and return (reward, new_position, new_entry_price, new_entry_bar, trade_pnl).
        """
        reward = 0.0
        trade_pnl = None

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
            elif position == -1:
                # Close short, open long
                trade_pnl = (entry_price - current_price) * 100 - 4
                if self.use_amddp10:
                    reward = self._calculate_amddp10(trade_pnl, max_dd, bars_held, position)
                else:
                    reward = self._calculate_amddp1(trade_pnl)
                position = 1
                entry_price = current_price
                entry_bar = current_bar
            else:
                reward = -1.0  # Invalid action

        elif action == 2:  # SELL
            if position == 0:
                reward = 1.0  # Reward decisive entry
                position = -1
                entry_price = current_price
                entry_bar = current_bar
            elif position == 1:
                # Close long, open short
                trade_pnl = (current_price - entry_price) * 100 - 4
                if self.use_amddp10:
                    reward = self._calculate_amddp10(trade_pnl, max_dd, bars_held, position)
                else:
                    reward = self._calculate_amddp1(trade_pnl)
                position = -1
                entry_price = current_price
                entry_bar = current_bar
            else:
                reward = -1.0  # Invalid action

        elif action == 3:  # CLOSE
            if position != 0:
                trade_pnl = self._calculate_current_pnl(position, entry_price, current_price)
                if self.use_amddp10:
                    reward = self._calculate_amddp10(trade_pnl, max_dd, bars_held, position)
                else:
                    reward = self._calculate_amddp1(trade_pnl)
                position = 0
                entry_price = 0.0
                entry_bar = -1
            else:
                reward = -1.0  # Invalid action

        return reward, position, entry_price, entry_bar, trade_pnl

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

    def _calculate_amddp1(self, pnl_pips: float) -> float:
        """AMDDP1 reward calculation with asymmetric penalties."""
        if pnl_pips > 0:
            if pnl_pips < 10:
                return 1.0 + pnl_pips * 0.05
            elif pnl_pips < 30:
                return 1.5 + (pnl_pips - 10) * 0.025
            else:
                return 2.0 + np.tanh((pnl_pips - 30) / 50)
        else:
            pnl_abs = abs(pnl_pips)
            if pnl_abs < 10:
                return -1.0 - pnl_abs * 0.1
            elif pnl_abs < 30:
                return -2.0 - (pnl_abs - 10) * 0.05
            else:
                return -3.0 - np.tanh((pnl_abs - 30) / 30)

    def _calculate_amddp10(self, pnl_pips: float, max_dd: float = 0.0,
                           bars_held: int = 0, position: int = 0) -> float:
        """AMDDP10 enhanced reward with multiple factors.

        Incorporates:
        - Base P&L reward/penalty (asymmetric)
        - Drawdown penalty
        - Time decay penalty for long holds
        - Bonus for quick profitable exits
        - Position management incentives
        """
        # Base reward from P&L
        if pnl_pips > 0:
            # Profitable trade
            if pnl_pips < 5:
                base_reward = 0.5 + pnl_pips * 0.1  # Small win
            elif pnl_pips < 15:
                base_reward = 1.0 + pnl_pips * 0.08  # Medium win
            elif pnl_pips < 30:
                base_reward = 2.2 + pnl_pips * 0.05  # Good win
            elif pnl_pips < 50:
                base_reward = 3.5 + pnl_pips * 0.03  # Great win
            else:
                base_reward = 5.0 + np.tanh((pnl_pips - 50) / 100) * 2  # Exceptional

            # Quick profit bonus (reward fast profitable exits)
            if bars_held > 0 and bars_held < 30:
                speed_bonus = (30 - bars_held) / 30 * 0.5
                base_reward += speed_bonus

        else:
            # Losing trade - heavier penalties
            pnl_abs = abs(pnl_pips)
            if pnl_abs < 5:
                base_reward = -1.0 - pnl_abs * 0.2  # Small loss
            elif pnl_abs < 15:
                base_reward = -2.0 - pnl_abs * 0.15  # Medium loss
            elif pnl_abs < 30:
                base_reward = -4.25 - pnl_abs * 0.1  # Large loss
            else:
                base_reward = -7.0 - np.tanh((pnl_abs - 30) / 50) * 3  # Severe loss

        # Drawdown penalty (encourage tight risk management)
        if max_dd > 0:
            dd_penalty = np.tanh(max_dd / 20) * 2.0  # Up to -2.0 penalty
            base_reward -= dd_penalty

        # Time decay penalty for holding too long
        if bars_held > 60:  # Positions held over 1 hour
            time_penalty = np.tanh((bars_held - 60) / 120) * 1.0
            base_reward -= time_penalty

        # Flat position bonus (encourage taking breaks)
        if position == 0 and pnl_pips == 0:
            base_reward += 0.1  # Small reward for being patient

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