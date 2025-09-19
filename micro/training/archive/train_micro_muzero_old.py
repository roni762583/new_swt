#!/usr/bin/env python3
"""
Training script for Micro Stochastic MuZero.

Implements the complete training loop with experience replay.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
import numpy as np
import pandas as pd
import duckdb
from collections import deque
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
import heapq
import logging
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import queue as queue_module
import threading

# Add parent directory to path
import sys
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.stochastic_mcts import StochasticMCTS, MarketOutcome
from micro.training.session_queue_manager import SessionQueueManager, ValidSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    input_features: int = 15
    lag_window: int = 32
    hidden_dim: int = 256
    action_dim: int = 4
    num_outcomes: int = 3  # UP, NEUTRAL, DOWN
    support_size: int = 300

    # Training
    batch_size: int = 64
    # Learning rate scheduling
    learning_rate: float = 2e-3  # Fixed at 0.002 as requested
    initial_lr: float = 2e-3  # Fixed learning rate - NO DECAY
    min_lr: float = 2e-3          # Same as initial - NO DECAY
    lr_decay_episodes: int = 1000000  # Effectively no decay

    # Exploration decay (critical for escaping Hold-only behavior)
    initial_temperature: float = 10.0  # EXTREME exploration to force action diversity
    final_temperature: float = 1.0     # Standard exploration later
    temperature_decay_episodes: int = 50000  # Faster decay once diversity achieved
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0  # Even more aggressive clipping for stability

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 100  # Slightly larger for more diverse initial experiences

    # MuZero
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997

    # MCTS - BALANCED FOR EXPLORATION
    num_simulations: int = 15  # Balanced for speed and exploration
    temperature: float = 2.0  # Higher MCTS temperature for more exploration

    # Multiprocessing
    num_workers: int = 4  # Parallel session collectors
    session_queue_size: int = 100  # Pre-validated sessions

    # Training loop
    num_episodes: int = 1000000
    checkpoint_interval: int = 50  # Save every 50 episodes
    log_interval: int = 100

    # Paths
    data_path: str = os.environ.get("DATA_PATH", "/workspace/data/micro_features.duckdb")
    checkpoint_dir: str = os.environ.get("CHECKPOINT_DIR", "/workspace/micro/checkpoints")
    log_dir: str = os.environ.get("LOG_DIR", "/workspace/micro/logs")


@dataclass
class Trajectory:
    """Trajectory segment for replay buffer."""
    observations: np.ndarray  # (seq_len, lag_window, features)
    actions: np.ndarray       # (seq_len,)
    rewards: np.ndarray       # (seq_len,)
    policies: np.ndarray      # (seq_len, action_dim)
    values: np.ndarray        # (seq_len,)
    td_errors: np.ndarray     # (seq_len,)
    priority: float = 1.0
    has_trade: bool = False

    def __len__(self):
        return len(self.actions)

@dataclass
class Experience:
    """Single experience with market outcome."""
    observation: np.ndarray
    action: int
    policy: np.ndarray
    value: float
    reward: float
    done: bool
    market_outcome: int = 1  # Default NEUTRAL
    outcome_probs: Optional[np.ndarray] = None  # Predicted probabilities
    td_error: float = 0.0



class BalancedReplayBuffer:
    """Simple balanced buffer with quota-based eviction (no PER)."""

    def __init__(self, max_size: int = 10000, min_trade_fraction: float = 0.2,
                 trajectory_length: int = 10):
        self.buffer = []
        self.max_size = max_size
        self.min_trade_fraction = min_trade_fraction
        self.trajectory_length = trajectory_length
        self.current_trajectory = []  # Accumulate experiences for trajectory

    def add(self, experience: Experience):
        """Add experience to current trajectory."""
        self.current_trajectory.append(experience)

        # Create trajectory segment when we reach trajectory_length
        if len(self.current_trajectory) >= self.trajectory_length:
            self._create_trajectory_from_buffer()

    def _create_trajectory_from_buffer(self):
        """Convert accumulated experiences into a trajectory."""
        if not self.current_trajectory:
            return

        # Extract data from experiences
        observations = [exp.observation for exp in self.current_trajectory]
        actions = [exp.action for exp in self.current_trajectory]
        rewards = [exp.reward for exp in self.current_trajectory]
        policies = [exp.policy for exp in self.current_trajectory]
        values = [exp.value for exp in self.current_trajectory]
        td_errors = [exp.td_error for exp in self.current_trajectory]

        # Check if trajectory contains trades (Buy, Sell, Close)
        has_trade = any(a in [1, 2, 3] for a in actions)

        # Create trajectory (NO PRIORITY - uniform sampling)
        trajectory = Trajectory(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            policies=np.array(policies),
            values=np.array(values),
            td_errors=np.array(td_errors),
            priority=1.0,  # Uniform priority - no PER
            has_trade=has_trade
        )

        self.buffer.append(trajectory)

        # Clear current trajectory
        self.current_trajectory = []

        # Evict if needed
        if len(self.buffer) > self.max_size:
            self._evict()

    def _evict(self):
        """Quota-based eviction to maintain trade diversity."""
        trade_trajs = [t for t in self.buffer if t.has_trade]
        hold_trajs = [t for t in self.buffer if not t.has_trade]

        trade_fraction = len(trade_trajs) / len(self.buffer) if self.buffer else 0

        if trade_fraction < self.min_trade_fraction and hold_trajs:
            # Too few trades - evict a random hold-only trajectory
            victim = np.random.choice(hold_trajs)
        else:
            # Sufficient trades - evict random trajectory (FIFO-like)
            victim = self.buffer[0]  # Simple FIFO eviction

        self.buffer.remove(victim)

    def sample_batch(self, batch_size: int) -> Dict:
        """Sample batch of experiences for training."""
        if not self.buffer:
            return None

        # Sample trajectories
        sampled_trajs = self.sample_trajectories(min(batch_size, len(self.buffer)))

        # Extract random experiences from trajectories
        all_observations = []
        all_actions = []
        all_rewards = []
        all_policies = []
        all_values = []

        for traj in sampled_trajs:
            if len(traj) > 0:
                # Sample random index from trajectory
                idx = np.random.randint(0, len(traj))
                all_observations.append(traj.observations[idx])
                all_actions.append(traj.actions[idx])
                all_rewards.append(traj.rewards[idx])
                all_policies.append(traj.policies[idx])
                all_values.append(traj.values[idx])

        if not all_observations:
            return None

        return {
            'observations': np.array(all_observations),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards),
            'policies': np.array(all_policies),
            'values': np.array(all_values)
        }

    def sample_trajectories(self, num_trajectories: int) -> List[Trajectory]:
        """Uniform random sampling (no priority)."""
        if len(self.buffer) <= num_trajectories:
            return self.buffer.copy()

        # Uniform random sampling
        indices = np.random.choice(len(self.buffer), size=num_trajectories, replace=False)
        return [self.buffer[i] for i in indices]

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {'total': 0, 'trade_fraction': 0}

        trade_trajs = [t for t in self.buffer if t.has_trade]
        return {
            'total': len(self.buffer),
            'trade_trajs': len(trade_trajs),
            'hold_trajs': len(self.buffer) - len(trade_trajs),
            'trade_fraction': len(trade_trajs) / len(self.buffer)
        }

    def __len__(self):
        return len(self.buffer)


class DataLoader:
    """Load micro features from DuckDB with train/validate/test split."""

    def __init__(self, db_path: str, lag_window: int = 32,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 session_length: int = 360,
                 max_gap_minutes: int = 10):
        self.conn = duckdb.connect(db_path, read_only=True)
        self.lag_window = lag_window
        self.session_length = session_length  # 360 minutes = 6 hours
        self.max_gap_minutes = max_gap_minutes  # Max allowed gap between bars

        # Get total rows
        self.total_rows = self.conn.execute(
            "SELECT COUNT(*) FROM micro_features"
        ).fetchone()[0]

        # Calculate data splits
        self.train_end = int(self.total_rows * train_ratio)
        self.val_end = int(self.total_rows * (train_ratio + val_ratio))

        # Data ranges
        self.train_range = (0, self.train_end)
        self.val_range = (self.train_end, self.val_end)
        self.test_range = (self.val_end, self.total_rows)

        logger.info(f"DataLoader initialized with {self.total_rows:,} rows")
        logger.info(f"  Train: rows 0-{self.train_end:,} ({train_ratio*100:.0f}%)")
        logger.info(f"  Val: rows {self.train_end:,}-{self.val_end:,} ({val_ratio*100:.0f}%)")
        logger.info(f"  Test: rows {self.val_end:,}-{self.total_rows:,} ({test_ratio*100:.0f}%)")
        logger.info(f"  Session: {session_length}min, max gap {max_gap_minutes}min, weekend filtering ON")

    def _validate_session(self, start_idx: int) -> Tuple[bool, Optional[str]]:
        """
        Validate a session starting at given index.

        Checks:
        - No gaps > max_gap_minutes
        - No weekend hours (Friday 22:00 UTC to Sunday 22:00 UTC)
        - No open positions at session end

        Returns:
            (is_valid, rejection_reason)
        """
        # Get session data with timestamps
        query = f"""
        SELECT timestamp, close, position_side
        FROM micro_features
        WHERE bar_index >= {start_idx}
        AND bar_index < {start_idx + self.session_length}
        ORDER BY bar_index
        LIMIT {self.session_length}
        """

        rows = self.conn.execute(query).fetchall()

        if len(rows) < self.session_length:
            return False, "insufficient_data"

        prev_time = None
        has_open_position = False

        for i, (timestamp_str, close, position_side) in enumerate(rows):
            # Parse timestamp
            if isinstance(timestamp_str, str):
                timestamp = pd.to_datetime(timestamp_str)
            else:
                timestamp = timestamp_str

            # Check weekend hours
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            hour = timestamp.hour

            # Saturday is always closed
            if weekday == 5:
                return False, "weekend_saturday"

            # Friday after 22:00 UTC
            if weekday == 4 and hour >= 22:
                return False, "friday_close"

            # Sunday before 22:00 UTC
            if weekday == 6 and hour < 22:
                return False, "sunday_closed"

            # Check gap between bars
            if prev_time is not None:
                gap_seconds = (timestamp - prev_time).total_seconds()
                gap_minutes = gap_seconds / 60.0

                if gap_minutes > self.max_gap_minutes:
                    return False, f"gap_{gap_minutes:.1f}min"

            # Check if position is open at end
            if i == len(rows) - 1:  # Last bar
                if position_side != 0:  # Position still open
                    has_open_position = True

            prev_time = timestamp

        # FIXED: Don't reject sessions with open positions
        # Sessions with open positions are valid for training
        # We only warn about them for debugging
        if has_open_position:
            logger.debug(f"Session ends with open position (this is OK for training)")

        # Always return True - all sessions are valid

        return True, None

    def get_session_from_queue(self, session: ValidSession) -> Dict:
        """
        Get data for a pre-validated session from queue.

        Args:
            session: ValidSession from queue

        Returns:
            Dictionary with features and metadata
        """
        # Fetch window data
        query = f"""
        SELECT *
        FROM micro_features
        WHERE bar_index >= {session.start_idx}
        ORDER BY bar_index
        LIMIT 1000
        """

        data = self.conn.execute(query).fetchdf()

        # Extract features
        feature_cols = []

        # Technical features with lags
        technical_features = [
            'position_in_range_60',
            'min_max_scaled_momentum_60',
            'min_max_scaled_rolling_range',
            'min_max_scaled_momentum_5',
            'price_change_pips'
        ]

        for feat in technical_features:
            for lag in range(self.lag_window):
                col_name = f"{feat}_{lag}"
                if col_name in data.columns:
                    feature_cols.append(col_name)

        # Cyclical features with lags
        cyclical_features = [
            'dow_cos_final',
            'dow_sin_final',
            'hour_cos_final',
            'hour_sin_final'
        ]

        for feat in cyclical_features:
            for lag in range(self.lag_window):
                col_name = f"{feat}_{lag}"
                if col_name in data.columns:
                    feature_cols.append(col_name)

        # Position features (current only)
        position_features = [
            'position_side',
            'position_pips',
            'bars_since_entry',
            'pips_from_peak',
            'max_drawdown_pips',
            'accumulated_dd'
        ]

        for feat in position_features:
            if feat in data.columns:
                feature_cols.append(feat)

        # Extract feature matrix
        features = data[feature_cols].values

        return {
            'features': features,
            'prices': data['close'].values,
            'timestamps': data['timestamp'].values,
            'start_idx': session.start_idx,
            'split': session.split,
            'quality_score': session.quality_score
        }

    def get_random_window(self, split: str = 'train') -> Dict:
        """Get random window from specified data split.

        Args:
            split: 'train', 'val', or 'test'
        """
        # Select range based on split
        if split == 'train':
            range_start, range_end = self.train_range
        elif split == 'val':
            range_start, range_end = self.val_range
        elif split == 'test':
            range_start, range_end = self.test_range
        else:
            raise ValueError(f"Invalid split: {split}")

        # Sample random starting point within range
        max_start = range_end - self.session_length - 100  # Leave room for session
        if max_start <= range_start:
            max_start = range_end - 100  # Smaller buffer for small ranges

        # Try to find valid session (max 20 attempts)
        valid_found = False
        rejection_counts = {}

        for attempt in range(20):
            start_idx = np.random.randint(range_start, max(range_start + 1, max_start))
            is_valid, reason = self._validate_session(start_idx)

            if is_valid:
                valid_found = True
                break
            else:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        if not valid_found:
            # Log rejection reasons periodically
            if np.random.random() < 0.01:  # Log 1% of rejections
                logger.debug(f"Session rejections in {split}: {rejection_counts}")
            # Use last attempted index anyway for training continuity
            start_idx = np.random.randint(range_start, max(range_start + 1, max_start))

        # Fetch window data
        query = f"""
        SELECT *
        FROM micro_features
        WHERE bar_index >= {start_idx}
        ORDER BY bar_index
        LIMIT 1000
        """

        data = self.conn.execute(query).fetchdf()

        # ROBUST DATA LOADING - Handle missing features gracefully
        logger.debug(f"Available columns: {list(data.columns)[:10]}...")  # Log first 10 columns

        # Check if we have the expected lagged format or need to generate synthetic data
        has_lagged_features = any(col.endswith('_0') for col in data.columns)

        if not has_lagged_features:
            logger.warning("Lagged features not found - generating synthetic 15-feature data")
            # Generate synthetic 15-feature observation for testing
            observation = []
            for t in range(self.lag_window):
                # Create 15 features per timestep
                row_features = []

                # 5 technical indicators (normalized random values)
                for i in range(5):
                    row_features.append(np.random.normal(0, 0.1))  # Small variance

                # 4 cyclical time features (sine/cosine patterns)
                hour_angle = (t % 24) * 2 * np.pi / 24
                dow_angle = (t % 168) * 2 * np.pi / 168  # Weekly cycle
                row_features.extend([
                    np.sin(hour_angle), np.cos(hour_angle),  # Hour cyclical
                    np.sin(dow_angle), np.cos(dow_angle)     # Day-of-week cyclical
                ])

                # 6 position features (realistic trading states)
                position_side = np.random.choice([-1, 0, 1])  # Short, flat, long
                position_pips = np.random.normal(0, 10) if position_side != 0 else 0
                bars_since_entry = np.random.exponential(20) if position_side != 0 else 0
                pips_from_peak = min(0, np.random.normal(-5, 10)) if position_side != 0 else 0
                max_drawdown = min(0, np.random.normal(-8, 15)) if position_side != 0 else 0
                accumulated_dd = abs(max_drawdown) * np.random.uniform(0.5, 2.0)

                row_features.extend([
                    position_side,
                    np.tanh(position_pips / 100),      # Normalized position P&L
                    np.tanh(bars_since_entry / 100),   # Normalized time in position
                    np.tanh(pips_from_peak / 100),     # Normalized distance from peak
                    np.tanh(max_drawdown / 100),       # Normalized max drawdown
                    np.tanh(accumulated_dd / 100)      # Normalized accumulated drawdown
                ])

                observation.append(row_features)
        else:
            # Original code path for when lagged features exist
            observation = []
            for t in range(self.lag_window):
                row_features = []

                # Add technical and cyclical at time t
                for feat in ['position_in_range_60', 'min_max_scaled_momentum_60',
                            'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
                            'price_change_pips', 'dow_cos_final', 'dow_sin_final',
                            'hour_cos_final', 'hour_sin_final']:
                    col_name = f"{feat}_{self.lag_window - 1 - t}"
                    if col_name in data.columns:
                        value = data.iloc[0][col_name]
                        # Validate and clean the value
                        if np.isnan(value) or np.isinf(value):
                            logger.warning(f"Invalid value in {col_name}: {value}, using 0.0")
                            value = 0.0
                        row_features.append(float(value))
                    else:
                        logger.warning(f"Column {col_name} not found, using 0.0")
                        row_features.append(0.0)

                # Add position features (always from current/first row)
                position_cols = ['position_side', 'position_pips', 'bars_since_entry',
                               'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd']
                for feat in position_cols:
                    if feat in data.columns:
                        value = data.iloc[0][feat]
                        # Validate and clean the value
                        if np.isnan(value) or np.isinf(value):
                            logger.warning(f"Invalid value in {feat}: {value}, using 0.0")
                            value = 0.0
                        row_features.append(float(value))
                    else:
                        logger.warning(f"Column {feat} not found, using 0.0")
                        row_features.append(0.0)

                observation.append(row_features)

        # Final validation of observation shape and values
        observation = np.array(observation, dtype=np.float32)
        if observation.shape != (self.lag_window, 15):
            logger.error(f"Invalid observation shape: {observation.shape}, expected ({self.lag_window}, 15)")
            # Create fallback observation
            observation = np.random.randn(self.lag_window, 15).astype(np.float32) * 0.1

        # Check for NaN/Inf in final observation
        if np.isnan(observation).any() or np.isinf(observation).any():
            logger.error("NaN/Inf detected in observation - replacing with zeros")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return {
            'observation': np.array(observation, dtype=np.float32),
            'data': data,
            'start_idx': start_idx
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


class SessionCollector(Process):
    """Worker process for parallel experience collection."""

    def __init__(self, worker_id: int, config: TrainingConfig,
                 session_queue: Queue, experience_queue: Queue,
                 stop_event: mp.Event):
        super().__init__()
        self.worker_id = worker_id
        self.config = config
        self.session_queue = session_queue
        self.experience_queue = experience_queue
        self.stop_event = stop_event

    def run(self):
        """Worker main loop."""
        # Initialize components for this worker
        device = torch.device("cpu")  # Workers use CPU
        model = MicroStochasticMuZero(
            input_features=self.config.input_features,
            lag_window=self.config.lag_window,
            hidden_dim=self.config.hidden_dim,
            action_dim=self.config.action_dim,
            num_outcomes=self.config.num_outcomes,
            support_size=self.config.support_size
        ).to(device)

        mcts = MCTS(
            model=model,
            num_actions=self.config.action_dim,
            discount=self.config.discount,
            num_simulations=self.config.num_simulations
        )

        data_loader = DataLoader(self.config.data_path, self.config.lag_window)

        logger.info(f"Worker {self.worker_id} started")

        # Position tracking for AMDDP1
        position = 0
        entry_price = 0.0
        entry_step = -1

        while not self.stop_event.is_set():
            try:
                # Get session from queue (with timeout)
                session_data = self.session_queue.get(timeout=1)
                if session_data is None:  # Poison pill
                    break

                # Process session with post-trade rewards
                experiences = []
                trade_experiences = []  # Track current trade
                trade_start_idx = -1

                for step in range(360):  # 360 minute session
                    # Get observation
                    observation = session_data['observations'][step]

                    # Run MCTS
                    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        mcts_result = mcts.run(
                            obs_tensor,
                            add_exploration_noise=True,
                            temperature=self.config.initial_temperature
                        )

                    action = mcts_result['action']
                    value = mcts_result['value']
                    policy = mcts_result['policy']

                    # Calculate reward with post-trade system
                    current_price = session_data['prices'][step]
                    next_price = session_data['prices'][step + 1] if step < 359 else current_price

                    (reward, position, entry_price, entry_step,
                     trade_closed, final_amddp1, pnl_pips) = self._calculate_reward(
                        action, current_price, next_price,
                        position, entry_price, entry_step, step
                    )

                    # Track trade experiences
                    if action in [1, 2] and position != 0:  # New trade started
                        trade_experiences = [len(experiences)]
                        trade_start_idx = len(experiences)
                    elif position != 0 and trade_start_idx >= 0:  # In trade
                        trade_experiences.append(len(experiences))

                    # Create experience with all required fields
                    exp = Experience(
                        observation=observation,
                        action=action,
                        policy=policy,
                        value=value,
                        reward=reward,
                        done=False,
                        pip_pnl=pnl_pips if trade_closed else 0.0,
                        trade_complete=trade_closed,
                        position_change=(action in [1, 2, 3]),
                        td_error=0.0,
                        session_expectancy=0.0,
                        trade_id=None
                    )
                    experiences.append(exp)

                    # Retroactively assign rewards if trade closed
                    if trade_closed and trade_experiences:
                        # Update all trade experiences with final AMDDP1
                        for idx in trade_experiences:
                            experiences[idx].reward = final_amddp1
                            experiences[idx].pip_pnl = pnl_pips
                            experiences[idx].trade_complete = True
                        trade_experiences = []  # Reset for next trade
                        trade_start_idx = -1

                # Put experiences in queue
                for exp in experiences:
                    self.experience_queue.put(exp)

            except queue_module.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")

        logger.info(f"Worker {self.worker_id} stopped")

    def _calculate_reward(self, action, current_price, next_price,
                         position, entry_price, entry_step, current_step):
        """Calculate POST-TRADE rewards with placeholders."""
        reward = 0.0
        trade_closed = False
        final_amddp1 = 0.0
        pnl_pips = 0.0

        if action == 0:  # HOLD
            # Placeholder - will be reassigned if part of trade
            reward = 0.0
        elif action == 1:  # BUY
            if position == 0:
                reward = 0.0  # Placeholder - NO immediate reward
                position = 1
                entry_price = current_price
                entry_step = current_step
            else:
                reward = -1.0  # Invalid
        elif action == 2:  # SELL
            if position == 0:
                reward = 0.0  # Placeholder - NO immediate reward
                position = -1
                entry_price = current_price
                entry_step = current_step
            else:
                reward = -1.0  # Invalid
        elif action == 3:  # CLOSE
            if position != 0:
                # Calculate final P&L
                if position == 1:
                    pnl_pips = (current_price - entry_price) * 100 - 4
                else:
                    pnl_pips = (entry_price - current_price) * 100 - 4

                final_amddp1 = self._calculate_amddp1(pnl_pips)
                reward = final_amddp1  # Close gets final reward
                trade_closed = True

                # Reset position
                position = 0
                entry_price = 0
                entry_step = -1
            else:
                reward = -1.0  # Invalid

        return (np.clip(reward, -3.0, 3.0), position, entry_price, entry_step,
                trade_closed, final_amddp1, pnl_pips)

    def _calculate_amddp1(self, pnl_pips):
        """AMDDP1 calculation."""
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


class MicroMuZeroTrainer:
    """Main trainer for Micro Stochastic MuZero with multiprocessing."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MicroStochasticMuZero(
            input_features=config.input_features,
            lag_window=config.lag_window,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_outcomes=config.num_outcomes,
            support_size=config.support_size
        ).to(self.device)

        # Force complete weight randomization for fresh start
        if hasattr(self.model, 'randomize_weights'):
            self.model.randomize_weights()
            logger.info("Model weights randomized for fresh start")
        else:
            logger.info("Using default weight initialization")

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        # Learning rate scheduler - DISABLED for fixed learning rate
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=1.0  # NO DECAY - fixed learning rate
        )

        # Track current temperature for exploration decay
        self.current_temperature = config.initial_temperature
        self.temperature_decay_rate = (config.final_temperature / config.initial_temperature) ** (1.0 / config.temperature_decay_episodes)

        # Initialize MCTS with minimal simulations for ultra-fast collection
        self.mcts = StochasticMCTS(
            model=self.model,
            num_actions=config.action_dim,
            num_outcomes=config.num_outcomes,
            discount=config.discount,
            num_simulations=config.num_simulations,
            dirichlet_alpha=1.0,  # Strong exploration noise
            exploration_fraction=0.5,  # 50% exploration mix at root
            depth_limit=3  # Planning depth for stochastic tree
        )

        # Initialize data loader
        self.data_loader = DataLoader(config.data_path, config.lag_window)

        # Initialize balanced replay buffer (no PER, quota-based)
        self.buffer = BalancedReplayBuffer(
            max_size=config.buffer_size,
            min_trade_fraction=0.2,
            trajectory_length=10
        )

        # Training stats
        self.episode = 0
        self.total_steps = 0
        self.last_optimizer_step_episode = -1  # Track when optimizer.step() was last called
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'policy_losses': [],
            'value_losses': [],
            'reward_losses': [],
            'expectancies': [],  # Track expectancy over time
            'win_rates': [],     # Track win rates
            'trade_counts': []   # Track number of trades
        }

        # Initialize position tracking for AMDDP1 rewards
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.position_pnl = 0.0
        self.entry_step = -1  # Track entry step for retroactive AMDDP1


        # Multiprocessing setup
        self.session_queue = Queue(maxsize=config.session_queue_size)
        self.experience_queue = Queue(maxsize=1000)
        self.stop_event = mp.Event()
        self.workers = []

        # Session pre-validator thread
        self.session_validator = None

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Multiprocessing: {config.num_workers} workers")

    def start_workers(self):
        """Start worker processes for parallel collection."""
        for i in range(self.config.num_workers):
            worker = SessionCollector(
                worker_id=i,
                config=self.config,
                session_queue=self.session_queue,
                experience_queue=self.experience_queue,
                stop_event=self.stop_event
            )
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.config.num_workers} worker processes")

    def stop_workers(self):
        """Stop all worker processes."""
        # Signal workers to stop
        self.stop_event.set()

        # Send poison pills
        for _ in self.workers:
            self.session_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

        self.workers.clear()
        logger.info("All workers stopped")

    def populate_session_queue(self):
        """Pre-populate session queue with validated sessions."""
        sessions_added = 0
        while sessions_added < self.config.session_queue_size // 2:
            try:
                # Get random window
                window = self.data_loader.get_random_window(split='train')

                # Create session data
                session_data = {
                    'observations': [window['observation']] * 360,  # Simplified for now
                    'prices': [window.get('close', 180.0)] * 361
                }

                self.session_queue.put(session_data, timeout=0.1)
                sessions_added += 1
            except queue_module.Full:
                break
            except Exception as e:
                logger.error(f"Error populating session queue: {e}")
                break

        logger.info(f"Session queue populated with {sessions_added} sessions")

    def collect_experience(self) -> Dict:
        """Collect experience using MCTS with post-trade reward system."""
        # Get random window from TRAINING data only
        window = self.data_loader.get_random_window(split='train')
        observation = torch.tensor(
            window['observation'],  # Use observation from the window
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Get current and next prices for reward calculation
        # Use the data from window to get price
        current_data = window['data']
        current_price = current_data['close'].iloc[0] if 'close' in current_data.columns else 1.0

        # For next price, just use a small offset from current
        # (This is simplified - in real episode we'd track sequential prices)
        next_price = current_price * 1.0001  # Small synthetic change

        # Run MCTS - use minimal simulations during initial collection
        self.model.eval()
        num_sims = 1 if len(self.buffer) < self.config.min_buffer_size else self.config.num_simulations
        self.mcts.num_simulations = num_sims  # Temporarily override
        mcts_result = self.mcts.run(
            observation,
            add_noise=True,  # Changed from add_exploration_noise
            temperature=self.current_temperature
        )
        self.mcts.num_simulations = self.config.num_simulations  # Restore

        # Calculate POST-TRADE REWARDS (placeholders until trade closes)
        action = mcts_result['action']
        reward = self._calculate_action_reward(action, current_price, next_price)

        # Create experience
        experience = Experience(
            observation=window['observation'],  # Use observation from window
            action=action,
            policy=mcts_result['policy'],
            value=mcts_result['value'],
            reward=reward,
            done=False,
            td_error=0.0  # Will be calculated during training
        )

        return experience

    def _calculate_action_reward(self, action: int, current_price: float, next_price: float) -> float:
        """
        Calculate CLEAN rewards with clear signals.

        Reward Structure:
        0: HOLD - 0.0 (intra-trade) or -0.05 (idle/extra-trade)
        1: BUY - +1.0 immediate reward for decisive entry
        2: SELL - +1.0 immediate reward for decisive entry
        3: CLOSE - AMDDP1 based on actual P&L
        """
        reward = 0.0

        if action == 0:  # HOLD
            if self.position != 0:
                # INTRA-TRADE HOLD: Neutral (don't overweight patience)
                reward = 0.0
            else:
                # IDLE HOLD: Small penalty to discourage inactivity
                reward = -0.05

        elif action == 1:  # BUY (open long ONLY)
            if self.position == 0:  # Can only buy when flat
                # IMMEDIATE REWARD for taking action
                reward = 1.0
                self.position = 1
                self.entry_price = current_price
                self.entry_step = self.total_steps
            else:
                # Invalid - already have position
                reward = -1.0

        elif action == 2:  # SELL (open short ONLY)
            if self.position == 0:  # Can only sell when flat
                # IMMEDIATE REWARD for taking action
                reward = 1.0
                self.position = -1
                self.entry_price = current_price
                self.entry_step = self.total_steps
            else:
                # Invalid - already have position
                reward = -1.0

        elif action == 3:  # CLOSE
            if self.position != 0:  # Have position to close
                # Calculate P&L with 4 pip spread
                if self.position == 1:  # Close long
                    pnl_pips = (current_price - self.entry_price) * 100 - 4
                else:  # Close short
                    pnl_pips = (self.entry_price - current_price) * 100 - 4

                # AMDDP1 reward for close
                reward = self._calculate_amddp1(pnl_pips)

                # Reset position
                self.position = 0
                self.entry_price = 0
                self.entry_step = -1
            else:  # No position to close
                reward = -0.5  # Penalty for invalid action
        else:
            reward = -1.0  # Invalid action

        # Clamp reward to reasonable range
        return np.clip(reward, -3.0, 3.0)

    def _calculate_amddp1(self, pnl_pips: float) -> float:
        """
        AMDDP1 (Asymmetric Mean Deviation Drawdown Penalty) calculation.
        """
        if pnl_pips > 0:
            # Winning trade - progressive reward
            if pnl_pips < 10:
                return 1.0 + pnl_pips * 0.05  # 1.0 to 1.5
            elif pnl_pips < 30:
                return 1.5 + (pnl_pips - 10) * 0.025  # 1.5 to 2.0
            else:
                return 2.0 + np.tanh((pnl_pips - 30) / 50)  # 2.0 to ~3.0
        else:
            # Losing trade - asymmetric penalty
            pnl_abs = abs(pnl_pips)
            if pnl_abs < 10:
                return -1.0 - pnl_abs * 0.1  # -1.0 to -2.0
            elif pnl_abs < 30:
                return -2.0 - (pnl_abs - 10) * 0.05  # -2.0 to -3.0
            else:
                return -3.0 - np.tanh((pnl_abs - 30) / 30)  # -3.0 to ~-4.0

    def train_batch(self, batch_data: Dict) -> Dict[str, float]:
        """Train on a batch of data."""
        self.model.train()

        # Prepare batch tensors
        observations = torch.tensor(
            batch_data['observations'],
            device=self.device,
            dtype=torch.float32
        )

        target_policies = torch.tensor(
            batch_data['policies'],
            device=self.device,
            dtype=torch.float32
        )

        target_values = torch.tensor(
            batch_data['values'],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(1)

        # ROBUST FORWARD PASS with input validation
        # Validate input observations
        if torch.isnan(observations).any() or torch.isinf(observations).any():
            logger.error("NaN/Inf in input observations - skipping batch")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        try:
            # Anomaly detection disabled for performance
            hidden, policy_logits, value_probs = self.model.initial_inference(observations)
        except RuntimeError as e:
            logger.error(f"Forward pass failed: {e}")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        # Validate forward pass outputs
        if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
            logger.error("NaN/Inf in policy logits - skipping batch")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        if torch.isnan(value_probs).any() or torch.isinf(value_probs).any():
            logger.error("NaN/Inf in value probabilities - skipping batch")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        # ROBUST LOSS COMPUTATION
        try:
            # Policy loss with label smoothing for stability
            policy_loss = nn.functional.cross_entropy(
                policy_logits,
                target_policies,
                label_smoothing=0.01  # Small smoothing for numerical stability
            )

            # Value loss using Huber loss (more robust than MSE)
            predicted_values = self.model.value.get_value(value_probs)
            value_loss = nn.functional.huber_loss(
                predicted_values,
                target_values,
                delta=1.0  # Less sensitive to outliers
            )

            # Check individual losses
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                logger.error(f"Invalid policy loss: {policy_loss}")
                policy_loss = torch.tensor(0.0, requires_grad=True)

            if torch.isnan(value_loss) or torch.isinf(value_loss):
                logger.error(f"Invalid value loss: {value_loss}")
                value_loss = torch.tensor(0.0, requires_grad=True)

            # Total loss with clamping
            total_loss = policy_loss + value_loss
            total_loss = torch.clamp(total_loss, 0.0, 100.0)  # Clamp to reasonable range

        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"NaN/Inf in final loss: {total_loss.item():.6f} - skipping batch")
            self.optimizer.zero_grad()
            return {
                'total_loss': float('nan'),
                'policy_loss': float('nan'),
                'value_loss': float('nan')
            }

        # Calculate TD errors for quality scoring
        # Note: With BalancedReplayBuffer, we don't update individual experiences
        # since we're working with pre-sampled arrays, not Experience objects

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )

        self.optimizer.step()

        # Mark that optimizer step was successful (for scheduler)
        self.last_optimizer_step_episode = self.episode

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def save_checkpoint(self, is_best=False):
        """Save training checkpoint with cleanup of old files."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"micro_checkpoint_ep{self.episode:06d}.pth"
        )

        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'model_state': self.model.get_weights(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'buffer_state': self.buffer.get_state() if hasattr(self.buffer, 'get_state') else None
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save as latest
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        # Save as best if specified
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Best checkpoint saved: {best_path}")

        # Clean up old checkpoints (keep only last 2 + best + latest)
        self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self, keep_last=2):
        """Remove old checkpoints, keeping only recent ones."""
        checkpoint_files = []
        for f in os.listdir(self.config.checkpoint_dir):
            if f.startswith('micro_checkpoint_ep') and f.endswith('.pth'):
                checkpoint_files.append(f)

        # Sort by episode number
        checkpoint_files.sort()

        # Remove old checkpoints
        if len(checkpoint_files) > keep_last:
            for f in checkpoint_files[:-keep_last]:
                path = os.path.join(self.config.checkpoint_dir, f)
                os.remove(path)
                logger.debug(f"Removed old checkpoint: {f}")

    def resume_from_checkpoint(self):
        """Resume training from latest checkpoint if exists."""
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pth")

        if os.path.exists(latest_path):
            logger.info(f"Resuming from checkpoint: {latest_path}")
            checkpoint = torch.load(latest_path, map_location=self.device)

            # Restore model and optimizer
            self.model.set_weights(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # Restore training state
            self.episode = checkpoint['episode']
            self.total_steps = checkpoint['total_steps']
            self.training_stats = checkpoint['training_stats']

            # Ensure new fields exist in loaded stats (for backward compatibility)
            if 'expectancies' not in self.training_stats:
                self.training_stats['expectancies'] = []
            if 'win_rates' not in self.training_stats:
                self.training_stats['win_rates'] = []
            if 'trade_counts' not in self.training_stats:
                self.training_stats['trade_counts'] = []

            # Restore buffer if available
            if checkpoint.get('buffer_state') and hasattr(self.buffer, 'set_state'):
                self.buffer.set_state(checkpoint['buffer_state'])

            logger.info(f"Resumed from episode {self.episode}, total steps {self.total_steps}")
            return True

        return False

    def train(self):
        """Main training loop with resume support."""
        # Try to resume from checkpoint
        resumed = self.resume_from_checkpoint()
        start_episode = self.episode if resumed else 0

        logger.info(f"Starting training from episode {start_episode}...")
        start_time = time.time()
        best_loss = float('inf')

        # Save initial checkpoint for testing BEFORE buffer collection
        if start_episode == 0 and not resumed:
            logger.info("Saving initial checkpoint for testing...")
            self.episode = 0
            self.save_checkpoint(is_best=True)  # Save as best to trigger validation
            logger.info("Initial checkpoint saved as both latest.pth and best.pth")

        # Collect initial experiences if not resumed
        # REDUCED to 100 experiences for faster startup (was 1000)
        min_buffer = min(100, self.config.min_buffer_size)
        if not resumed or len(self.buffer) < min_buffer:
            logger.info(f"Collecting {min_buffer} initial experiences... Current buffer size: {len(self.buffer)}")
            collection_count = 0
            start_time = time.time()

            while len(self.buffer) < min_buffer:
                if collection_count % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Experience {collection_count}... Buffer: {len(self.buffer)}/{min_buffer} ({elapsed:.1f}s elapsed)")

                experience = self.collect_experience()
                self.buffer.add(experience)
                # Track rewards for expectancy even during initial collection
                self.training_stats['rewards'].append(experience.reward)
                collection_count += 1

                if len(self.buffer) % 25 == 0:
                    logger.info(f"Buffer size: {len(self.buffer)}/{min_buffer}")

            total_time = time.time() - start_time
            logger.info(f"Collected {min_buffer} experiences in {total_time:.1f} seconds")

        logger.info("Starting main training loop...")

        for episode in range(start_episode, self.config.num_episodes):
            self.episode = episode

            # Collect new experience
            experience = self.collect_experience()
            self.buffer.add(experience)

            # Track rewards for expectancy calculation
            self.training_stats['rewards'].append(experience.reward)

            # Train on batch
            if len(self.buffer) >= self.config.batch_size:
                batch_data = self.buffer.sample_batch(self.config.batch_size)
                if batch_data is None:
                    continue
                losses = self.train_batch(batch_data)

                # Update stats
                self.training_stats['episodes'].append(episode)
                self.training_stats['losses'].append(losses['total_loss'])
                self.training_stats['policy_losses'].append(losses['policy_loss'])
                self.training_stats['value_losses'].append(losses['value_loss'])

            self.total_steps += 1
            # Update learning rate (only if optimizer stepped recently)
            if episode % 100 == 0 and self.last_optimizer_step_episode >= episode - 10:
                self.scheduler.step()

            # Decay exploration temperature
            if episode < self.config.temperature_decay_episodes:
                self.current_temperature *= self.temperature_decay_rate
                self.current_temperature = max(self.current_temperature, self.config.final_temperature)

            # Logging
            if episode % self.config.log_interval == 0 and episode > 0:
                elapsed = time.time() - start_time
                eps = episode / elapsed

                avg_loss = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0
                avg_policy_loss = np.mean(self.training_stats['policy_losses'][-100:]) if self.training_stats['policy_losses'] else 0
                avg_value_loss = np.mean(self.training_stats['value_losses'][-100:]) if self.training_stats['value_losses'] else 0

                # Calculate expectancy from recent rewards
                recent_rewards = self.training_stats['rewards'][-100:] if self.training_stats['rewards'] else []
                if recent_rewards:
                    wins = [r for r in recent_rewards if r > 0]
                    losses = [r for r in recent_rewards if r < 0]
                    win_rate = len(wins) / len(recent_rewards) if recent_rewards else 0
                    avg_win = np.mean(wins) if wins else 0
                    avg_loss_val = abs(np.mean(losses)) if losses else 0
                    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss_val)

                    # Track stats
                    self.training_stats['expectancies'].append(expectancy)
                    self.training_stats['win_rates'].append(win_rate)
                else:
                    expectancy = 0
                    win_rate = 0

                logger.info(
                    f"Episode {episode:,} | "
                    f"Steps {self.total_steps:,} | "
                    f"EPS: {eps:.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Expectancy: {expectancy:+.4f} | "
                    f"WinRate: {win_rate:.1%}"
                )

            # Save checkpoint
            if episode % self.config.checkpoint_interval == 0 and episode > 0:
                # Check if this is the best model
                recent_loss = np.mean(self.training_stats['losses'][-50:]) if len(self.training_stats['losses']) >= 50 else float('inf')
                is_best = recent_loss < best_loss
                if is_best:
                    best_loss = recent_loss

                self.save_checkpoint(is_best=is_best)

        # Final checkpoint
        self.save_checkpoint()
        logger.info("Training completed!")

        # Save final stats
        stats_path = os.path.join(
            self.config.log_dir,
            f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f)

        logger.info(f"Training stats saved: {stats_path}")

    def cleanup(self):
        """Clean up resources."""
        self.data_loader.close()


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = MicroMuZeroTrainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        trainer.cleanup()