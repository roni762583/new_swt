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

# Add parent directory to path
import sys
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.mcts_micro import MCTS
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
    z_dim: int = 16
    support_size: int = 300

    # Training
    batch_size: int = 64
        # Learning rate scheduling
    initial_lr: float = 5e-4  # Higher initial learning rate
    min_lr: float = 1e-5      # Minimum learning rate
    lr_decay_episodes: int = 50000  # Episodes for full decay

    # Exploration decay (critical for escaping Hold-only behavior)
    initial_temperature: float = 2.0  # High exploration initially
    final_temperature: float = 0.5    # Lower exploration later
    temperature_decay_episodes: int = 20000  # Faster decay for exploration
    weight_decay: float = 1e-5
    gradient_clip: float = 10.0

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # MuZero
    num_unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997

    # MCTS
    num_simulations: int = 15
    temperature: float = 1.0

    # Training loop
    num_episodes: int = 1000000
    checkpoint_interval: int = 50  # Save every 50 episodes
    log_interval: int = 100

    # Paths
    data_path: str = os.environ.get("DATA_PATH", "/workspace/data/micro_features.duckdb")
    checkpoint_dir: str = os.environ.get("CHECKPOINT_DIR", "/workspace/micro/checkpoints")
    log_dir: str = os.environ.get("LOG_DIR", "/workspace/micro/logs")


@dataclass
class Experience:
    """Experience with quality scoring."""
    observation: np.ndarray
    action: int
    policy: np.ndarray
    value: float
    reward: float
    done: bool
    # Quality metrics
    quality_score: float = 0.0
    pip_pnl: float = 0.0
    trade_complete: bool = False
    position_change: bool = False
    td_error: float = 0.0
    session_expectancy: float = 0.0
    trade_id: Optional[int] = None

    def calculate_quality_score(self) -> float:
        """Calculate quality score with heavy emphasis on trading performance and SQN."""
        score = 0.0

        # Validate inputs to prevent NaN propagation
        if np.isnan(self.pip_pnl) or np.isnan(self.reward):
            logger.error(
                f"NaN detected in quality score inputs: "
                f"pip_pnl={self.pip_pnl}, reward={self.reward}, action={self.action}"
            )
            return 0.1  # Return minimum valid score

        # PRIMARY FACTOR: Pip P&L (heaviest weight for actual trading performance)
        if self.pip_pnl > 0:
            score += self.pip_pnl * 2.0  # Much heavier weight on profitable trades
        else:
            score += self.pip_pnl * 0.3  # Still penalize losses but proportionally

        # CRITICAL: AMDDP1 reward (risk-adjusted returns)
        # AMDDP1 already incorporates drawdown penalty, so it's crucial
        score += self.reward * 1.5  # Heavily weight risk-adjusted returns

        # Trade completion bonus (emphasize full trade cycles)
        if self.trade_complete:
            if self.pip_pnl > 0:
                score += 10.0  # Major bonus for profitable completed trades
            else:
                score += 2.0   # Small bonus for completed losing trades

        # SQN-based quality component
        # SQN = (expectancy / std_dev) * sqrt(num_trades)
        # We use session_expectancy as proxy for now
        if self.session_expectancy > 0:
            # Scale expectancy to pseudo-SQN range
            pseudo_sqn = self.session_expectancy * 2.5  # Approximate scaling

            if pseudo_sqn >= 3.0:      # Excellent system
                score += 15.0
            elif pseudo_sqn >= 2.5:    # Good system
                score += 10.0
            elif pseudo_sqn >= 2.0:    # Average system
                score += 7.0
            elif pseudo_sqn >= 1.6:    # Below average
                score += 4.0
            else:                       # Poor system
                score += 1.0

        
        # CRITICAL: Action diversity bonus (combat Hold-only behavior)
        # Heavily reward non-Hold actions to encourage trading
        if self.action == 0:  # Hold action
            score -= 2.0  # Penalty for Hold (encourage active trading)
        else:  # Active trading actions (Buy, Sell, Close)
            score += 5.0  # Strong bonus for trading actions

        # Position change (important for learning diverse actions)
        if self.position_change:
            score += 3.0  # Increased - encourages action exploration

        # Terminal state bonus (episode completion)
        if self.done:
            score += 2.0  # Increased - terminal states are valuable

        # TD error (important learning signal - low error means good value estimates)
        if abs(self.td_error) < 1.0:
            score += 5.0  # High bonus for accurate predictions
        elif abs(self.td_error) < 2.0:
            score += 3.0  # Medium bonus
        elif abs(self.td_error) < 5.0:
            score += 1.0  # Small bonus
        # High TD errors get no bonus but aren't penalized

        return max(score, 0.1)  # Minimum quality score


class QualityExperienceBuffer:
    """Experience buffer with quality-based smart eviction."""

    def __init__(self, capacity: int, eviction_batch: int = 2000):
        self.capacity = capacity
        self.eviction_batch = eviction_batch
        self.buffer: List[Experience] = []
        self.quality_heap: List[Tuple[float, int]] = []  # Min-heap for eviction
        self.trade_experiences = {}  # Map trade_id to experience indices
        self.current_trade_id = 0
        self.total_evicted = 0

    def add(self, experience: Experience):
        """Add experience with quality scoring."""
        # Calculate quality score
        experience.quality_score = experience.calculate_quality_score()

        # Add to buffer
        self.buffer.append(experience)

        # Add to quality heap for smart eviction
        heapq.heappush(self.quality_heap, (experience.quality_score, len(self.buffer) - 1))

        # Track trade experiences
        if experience.trade_id is not None:
            if experience.trade_id not in self.trade_experiences:
                self.trade_experiences[experience.trade_id] = []
            self.trade_experiences[experience.trade_id].append(len(self.buffer) - 1)

        # Smart eviction if over capacity
        if len(self.buffer) > self.capacity:
            self._evict_low_quality()

    def _evict_low_quality(self):
        """Evict lowest quality experiences."""
        # Remove eviction_batch lowest quality experiences
        to_remove = set()

        # Get indices of lowest quality experiences
        while len(to_remove) < self.eviction_batch and self.quality_heap:
            quality, idx = heapq.heappop(self.quality_heap)
            if idx < len(self.buffer):  # Valid index
                to_remove.add(idx)

        # Remove in reverse order to maintain indices
        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.buffer):
                del self.buffer[idx]
                self.total_evicted += 1

        # Rebuild heap with updated indices
        self.quality_heap = []
        for i, exp in enumerate(self.buffer):
            heapq.heappush(self.quality_heap, (exp.quality_score, i))

        logger.debug(f"Evicted {len(to_remove)} low quality experiences (total: {self.total_evicted})")

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample with priority on quality, with proper NaN detection and handling."""
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()

        # Weighted sampling based on quality scores
        weights = []
        valid_indices = []
        nan_count = 0
        negative_count = 0

        for i, exp in enumerate(self.buffer):
            score = exp.quality_score

            # Track and log NaN/Inf values to identify root causes
            if np.isnan(score) or np.isinf(score):
                nan_count += 1
                if nan_count <= 3:  # Log first few to avoid spam
                    logger.warning(
                        f"NaN/Inf quality score at index {i}: "
                        f"pip_pnl={exp.pip_pnl}, reward={exp.reward}, "
                        f"expectancy={exp.session_expectancy}, action={exp.action}"
                    )
                continue  # Skip this experience

            if score <= 0:
                negative_count += 1
                score = 0.01  # Minimum weight for valid experiences

            weights.append(score)
            valid_indices.append(i)

        if nan_count > 0:
            logger.error(f"Found {nan_count}/{len(self.buffer)} experiences with NaN/Inf quality scores")

        if negative_count > 0:
            logger.debug(f"Found {negative_count} experiences with non-positive scores")

        # Fall back to uniform sampling if too few valid experiences
        if len(valid_indices) < batch_size:
            logger.error(
                f"Insufficient valid experiences ({len(valid_indices)}) for batch size {batch_size}. "
                f"Using uniform sampling."
            )
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]

        # Normalize valid weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample from valid experiences only
        sampled_valid_idx = np.random.choice(len(valid_indices), batch_size, replace=False, p=weights)
        indices = [valid_indices[i] for i in sampled_valid_idx]

        return [self.buffer[i] for i in indices]

    def reassign_trade_rewards(self, trade_id: int, final_amddp1_reward: float, pip_pnl: float):
        """Reassign rewards for all actions in a trade (equal partners)."""
        if trade_id not in self.trade_experiences:
            return

        # Update all experiences in this trade
        for exp_idx in self.trade_experiences[trade_id]:
            if exp_idx < len(self.buffer):
                exp = self.buffer[exp_idx]
                exp.reward = final_amddp1_reward
                exp.pip_pnl = pip_pnl
                exp.trade_complete = True
                # Recalculate quality score
                exp.quality_score = exp.calculate_quality_score()

        logger.debug(f"Reassigned {len(self.trade_experiences[trade_id])} experiences for trade {trade_id}")

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

        # Extract features
        feature_cols = []

        # Technical indicators with lags
        for feat in ['position_in_range_60', 'min_max_scaled_momentum_60',
                     'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
                     'price_change_pips']:
            for lag in range(self.lag_window):
                feature_cols.append(f"{feat}_{lag}")

        # Cyclical features with lags
        for feat in ['dow_cos_final', 'dow_sin_final',
                     'hour_cos_final', 'hour_sin_final']:
            for lag in range(self.lag_window):
                feature_cols.append(f"{feat}_{lag}")

        # Position features (no lags)
        position_cols = ['position_side', 'position_pips', 'bars_since_entry',
                        'pips_from_peak', 'max_drawdown_pips', 'accumulated_dd']

        # Prepare observation (first 32 timesteps)
        observation = []
        for t in range(self.lag_window):
            row_features = []

            # Add technical and cyclical at time t
            for feat in ['position_in_range_60', 'min_max_scaled_momentum_60',
                        'min_max_scaled_rolling_range', 'min_max_scaled_momentum_5',
                        'price_change_pips', 'dow_cos_final', 'dow_sin_final',
                        'hour_cos_final', 'hour_sin_final']:
                col_name = f"{feat}_{self.lag_window - 1 - t}"
                row_features.append(data.iloc[0][col_name])

            # Add position features (always from current/first row)
            for feat in position_cols:
                row_features.append(data.iloc[0][feat])

            observation.append(row_features)

        return {
            'observation': np.array(observation, dtype=np.float32),
            'data': data,
            'start_idx': start_idx
        }

    def close(self):
        """Close database connection."""
        self.conn.close()


class MicroMuZeroTrainer:
    """Main trainer for Micro Stochastic MuZero."""

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
            z_dim=config.z_dim,
            support_size=config.support_size
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        # Learning rate scheduler (exponential decay)
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.9999  # Slow decay
        )

        # Track current temperature for exploration decay
        self.current_temperature = config.initial_temperature
        self.temperature_decay_rate = (config.final_temperature / config.initial_temperature) ** (1.0 / config.temperature_decay_episodes)

        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            num_actions=config.action_dim,
            discount=config.discount,
            num_simulations=config.num_simulations
        )

        # Initialize data loader
        self.data_loader = DataLoader(config.data_path, config.lag_window)

        # Initialize quality experience buffer with smart eviction
        self.buffer = QualityExperienceBuffer(
            capacity=config.buffer_size,
            eviction_batch=max(100, config.buffer_size // 50)  # Evict 2% when full
        )

        # Training stats
        self.episode = 0
        self.total_steps = 0
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'policy_losses': [],
            'value_losses': [],
            'reward_losses': []
        }

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def collect_experience(self) -> Dict:
        """Collect experience using MCTS with AMDDP1 reward."""
        # Get random window from TRAINING data only
        window = self.data_loader.get_random_window(split='train')
        observation = torch.tensor(
            window['observation'],
            device=self.device
        ).unsqueeze(0)  # Add batch dimension

        # Run MCTS
        self.model.eval()
        mcts_result = self.mcts.run(
            observation,
            add_exploration_noise=True,
            temperature=self.current_temperature
        )

        # Calculate AMDDP1 reward (simplified for training)
        # In production, this comes from environment
        base_reward = np.random.randn() * 10  # Simulated P&L
        drawdown_penalty = abs(np.random.randn()) * 0.01  # 1% drawdown penalty
        amddp1_reward = base_reward - drawdown_penalty

        # Apply profit protection (matching main system)
        if base_reward > 0 and amddp1_reward < 0:
            amddp1_reward = 0.01  # Min protected reward

        # Create experience with quality metrics
        experience = Experience(
            observation=window['observation'],
            action=mcts_result['action'],
            policy=mcts_result['policy'],
            value=mcts_result['value'],
            reward=amddp1_reward,
            done=False,
            pip_pnl=base_reward,  # Simplified P&L tracking
            trade_complete=False,
            position_change=(mcts_result['action'] in [1, 2, 3]),  # Buy/Sell/Close
            td_error=0.0,  # Will be calculated during training
            session_expectancy=0.0,  # Will be updated per session
            trade_id=None
        )

        return experience

    def train_batch(self, batch: List[Experience]) -> Dict[str, float]:
        """Train on a batch of experiences."""
        self.model.train()

        # Prepare batch tensors
        observations = torch.tensor(
            np.array([e.observation for e in batch]),
            device=self.device,
            dtype=torch.float32
        )

        target_policies = torch.tensor(
            np.array([e.policy for e in batch]),
            device=self.device,
            dtype=torch.float32
        )

        target_values = torch.tensor(
            np.array([e.value for e in batch]),
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(1)

        # Forward pass
        hidden, policy_logits, value_probs = self.model.initial_inference(observations)

        # Calculate losses
        policy_loss = nn.functional.cross_entropy(
            policy_logits,
            target_policies
        )

        # Value loss (using scalar value for simplicity)
        predicted_values = self.model.value.get_value(value_probs)
        value_loss = nn.functional.mse_loss(predicted_values, target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Calculate TD errors for quality scoring
        with torch.no_grad():
            td_errors = (predicted_values - target_values).abs().cpu().numpy()
            for i, exp in enumerate(batch):
                exp.td_error = float(td_errors[i])
                exp.quality_score = exp.calculate_quality_score()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip
        )

        self.optimizer.step()

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
        if not resumed or len(self.buffer) < self.config.min_buffer_size:
            logger.info("Collecting initial experiences...")
            while len(self.buffer) < self.config.min_buffer_size:
                experience = self.collect_experience()
                self.buffer.add(experience)

                if len(self.buffer) % 100 == 0:
                    logger.info(f"Buffer size: {len(self.buffer)}/{self.config.min_buffer_size}")

        logger.info("Starting main training loop...")

        for episode in range(start_episode, self.config.num_episodes):
            self.episode = episode

            # Collect new experience
            experience = self.collect_experience()
            self.buffer.add(experience)

            # Train on batch
            if len(self.buffer) >= self.config.batch_size:
                batch = self.buffer.sample(self.config.batch_size)
                losses = self.train_batch(batch)

                # Update stats
                self.training_stats['episodes'].append(episode)
                self.training_stats['losses'].append(losses['total_loss'])
                self.training_stats['policy_losses'].append(losses['policy_loss'])
                self.training_stats['value_losses'].append(losses['value_loss'])

            self.total_steps += 1
            # Update learning rate
            if episode % 100 == 0:  # Every 100 episodes
                self.scheduler.step()

            # Decay exploration temperature
            if episode < self.config.temperature_decay_episodes:
                self.current_temperature *= self.temperature_decay_rate
                self.current_temperature = max(self.current_temperature, self.config.final_temperature)

            # Logging
            if episode % self.config.log_interval == 0 and episode > 0:
                elapsed = time.time() - start_time
                eps = episode / elapsed

                avg_loss = np.mean(self.training_stats['losses'][-100:])
                avg_policy_loss = np.mean(self.training_stats['policy_losses'][-100:])
                avg_value_loss = np.mean(self.training_stats['value_losses'][-100:])

                logger.info(
                    f"Episode {episode:,} | "
                    f"Steps {self.total_steps:,} | "
                    f"EPS: {eps:.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Policy: {avg_policy_loss:.4f} | "
                    f"Value: {avg_value_loss:.4f}"
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