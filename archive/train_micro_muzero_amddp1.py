#!/usr/bin/env python3
"""
Training script for Micro Stochastic MuZero with AMDDP1 rewards and multiprocessing.

Key Features:
- AMDDP1 reward system with retroactive assignment
- Multiprocessing for parallel session collection
- Pre-validated session queue
- Single position enforcement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import duckdb
from collections import deque
import time
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime, timezone
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager, Pool
import queue
import random
import threading

# Add parent directory to path
import sys
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.mcts_micro import MCTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration with AMDDP1 and multiprocessing."""
    # Model
    input_features: int = 15
    lag_window: int = 32
    hidden_dim: int = 256
    action_dim: int = 4
    z_dim: int = 16
    support_size: int = 300

    # Training
    batch_size: int = 64
    learning_rate: float = 2e-3  # Fixed at 0.002 as requested
    weight_decay: float = 1e-5
    gradient_clip: float = 10.0

    # Exploration (1% decay per 1000 episodes)
    initial_temperature: float = 4.0
    final_temperature: float = 2.0
    temperature_decay_per_1000: float = 0.01  # 1% per 1000 episodes

    # Replay buffer
    buffer_size: int = 10000
    min_buffer_size: int = 100

    # MCTS
    num_simulations: int = 15  # As specified
    discount: float = 0.997

    # Multiprocessing
    num_workers: int = 4
    session_queue_size: int = 100
    session_length: int = 360  # 360 minutes per session

    # Training loop
    num_episodes: int = 1000000
    save_interval: int = 50
    validate_interval: int = 100
    log_interval: int = 100

    # Data
    data_path: str = "/workspace/micro/data/micro_features.duckdb"
    checkpoint_dir: str = "/workspace/micro/checkpoints"
    log_dir: str = "/workspace/micro/logs"


@dataclass
class Experience:
    """Experience tuple with AMDDP1 support."""
    observation: np.ndarray
    action: int
    reward: float  # Can be updated retroactively
    value: float
    policy: np.ndarray
    episode_idx: int  # Track which episode this came from
    step_idx: int  # Track step within episode for retroactive rewards


@dataclass
class TradingSession:
    """Represents a validated 360-minute trading session."""
    start_idx: int
    end_idx: int
    timestamp: datetime
    has_gaps: bool = False


class SessionCollector(Process):
    """Worker process for parallel session collection with MCTS."""

    def __init__(self, worker_id: int, config: TrainingConfig,
                 session_queue: Queue, experience_queue: Queue,
                 model_weights: Dict):
        super().__init__()
        self.worker_id = worker_id
        self.config = config
        self.session_queue = session_queue
        self.experience_queue = experience_queue
        self.model_weights = model_weights

    def run(self):
        """Main worker loop."""
        # Initialize model for this worker
        device = torch.device("cpu")  # Workers use CPU
        model = MicroStochasticMuZero(
            input_features=self.config.input_features,
            lag_window=self.config.lag_window,
            hidden_dim=self.config.hidden_dim,
            action_dim=self.config.action_dim,
            z_dim=self.config.z_dim,
            support_size=self.config.support_size
        ).to(device)

        # Initialize MCTS
        mcts = MCTS(
            model=model,
            num_actions=self.config.action_dim,
            discount=self.config.discount,
            num_simulations=self.config.num_simulations
        )

        # Connect to database
        conn = duckdb.connect(self.config.data_path, read_only=True)

        logger.info(f"Worker {self.worker_id} started")

        while True:
            try:
                # Get session from queue
                session = self.session_queue.get(timeout=1)
                if session is None:  # Poison pill
                    break

                # Update model weights if available
                if self.model_weights:
                    model.set_weights(self.model_weights)

                # Collect experiences from session
                experiences = self._collect_session_experiences(
                    conn, session, model, mcts
                )

                # Put experiences in queue
                for exp in experiences:
                    self.experience_queue.put(exp)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")

        conn.close()
        logger.info(f"Worker {self.worker_id} stopped")

    def _collect_session_experiences(self, conn, session: TradingSession,
                                      model, mcts) -> List[Experience]:
        """Collect experiences from a single session with AMDDP1."""
        experiences = []

        # Load session data
        query = f"""
        SELECT * FROM micro_features
        WHERE bar_index >= {session.start_idx}
        AND bar_index < {session.end_idx}
        ORDER BY bar_index
        LIMIT {self.config.session_length}
        """
        data = conn.execute(query).fetchdf()

        # Trading state
        position = 0  # -1: short, 0: flat, 1: long
        entry_price = 0.0
        entry_step = -1
        pending_rewards = []  # For retroactive AMDDP1

        # Process each bar in session
        for step in range(len(data) - 1):
            # Get observation (32, 15)
            observation = self._extract_observation(data, step)

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

            # Calculate reward with AMDDP1
            current_price = data.iloc[step]['close']
            next_price = data.iloc[step + 1]['close']

            reward, position, entry_price, entry_step = self._calculate_amddp1_reward(
                action, current_price, next_price,
                position, entry_price, entry_step, step
            )

            # Create experience
            exp = Experience(
                observation=observation,
                action=action,
                reward=reward,
                value=value,
                policy=policy,
                episode_idx=session.start_idx,
                step_idx=step
            )
            experiences.append(exp)

            # Handle retroactive rewards for trade closes
            if action == 3 and position != 0:  # Successful close
                # Calculate final AMDDP1 for the trade
                if position == 1:  # Closed long
                    pnl_pips = (current_price - entry_price) * 100 - 4
                else:  # Closed short
                    pnl_pips = (entry_price - current_price) * 100 - 4

                amddp1_reward = self._compute_amddp1(pnl_pips)

                # Update entry action reward retroactively
                if 0 <= entry_step < len(experiences):
                    experiences[entry_step].reward = amddp1_reward

                # Reset position
                position = 0
                entry_price = 0
                entry_step = -1

        return experiences

    def _extract_observation(self, data, step: int) -> np.ndarray:
        """Extract (32, 15) observation from data."""
        observation = np.zeros((self.config.lag_window, 15), dtype=np.float32)

        # Fill with lagged features
        for lag in range(self.config.lag_window):
            if step - lag >= 0:
                row = data.iloc[step - lag]

                # Technical features (5)
                observation[lag, 0] = row.get('position_in_range_60', 0)
                observation[lag, 1] = row.get('min_max_scaled_momentum_60', 0)
                observation[lag, 2] = row.get('min_max_scaled_rolling_range', 0)
                observation[lag, 3] = row.get('min_max_scaled_momentum_5', 0)
                observation[lag, 4] = row.get('price_change_pips', 0)

                # Cyclical features (4)
                observation[lag, 5] = row.get('dow_cos_final', 0)
                observation[lag, 6] = row.get('dow_sin_final', 0)
                observation[lag, 7] = row.get('hour_cos_final', 0)
                observation[lag, 8] = row.get('hour_sin_final', 0)

                # Position features (6) - same for all lags
                observation[lag, 9] = row.get('position_side', 0)
                observation[lag, 10] = row.get('position_pips', 0)
                observation[lag, 11] = row.get('bars_since_entry', 0)
                observation[lag, 12] = row.get('pips_from_peak', 0)
                observation[lag, 13] = row.get('max_drawdown_pips', 0)
                observation[lag, 14] = row.get('accumulated_dd', 0)

        return observation

    def _calculate_amddp1_reward(self, action: int, current_price: float,
                                  next_price: float, position: int,
                                  entry_price: float, entry_step: int,
                                  current_step: int) -> Tuple[float, int, float, int]:
        """
        Calculate AMDDP1 reward with proper position management.

        Returns: (reward, new_position, new_entry_price, new_entry_step)
        """
        reward = 0.0

        if action == 0:  # HOLD
            if position != 0:
                reward = 0.1  # Intra-trade hold bonus
            else:
                reward = 0.0  # Extra-trade hold neutral

        elif action == 1:  # BUY (open long only)
            if position == 0:
                reward = 1.0  # Initial reward, updated on close
                position = 1
                entry_price = current_price
                entry_step = current_step
            else:
                reward = -1.0  # Invalid - already have position

        elif action == 2:  # SELL (open short only)
            if position == 0:
                reward = 1.0  # Initial reward, updated on close
                position = -1
                entry_price = current_price
                entry_step = current_step
            else:
                reward = -1.0  # Invalid - already have position

        elif action == 3:  # CLOSE
            if position != 0:
                # Calculate P&L
                if position == 1:  # Close long
                    pnl_pips = (current_price - entry_price) * 100 - 4
                else:  # Close short
                    pnl_pips = (entry_price - current_price) * 100 - 4

                reward = self._compute_amddp1(pnl_pips)
                position = 0
                entry_price = 0
                entry_step = -1
            else:
                reward = -1.0  # Invalid - no position

        return reward, position, entry_price, entry_step

    def _compute_amddp1(self, pnl_pips: float) -> float:
        """Compute AMDDP1 reward value."""
        if pnl_pips > 0:
            # Winning trade
            if pnl_pips < 10:
                return 1.0 + pnl_pips * 0.05  # Small win
            elif pnl_pips < 30:
                return 1.5 + (pnl_pips - 10) * 0.025  # Medium win
            else:
                return 2.0 + np.tanh((pnl_pips - 30) / 50)  # Large win, capped
        else:
            # Losing trade - asymmetric penalty
            if pnl_pips > -10:
                return -1.0 + pnl_pips * 0.1  # Small loss
            elif pnl_pips > -30:
                return -2.0 + (pnl_pips + 10) * 0.05  # Medium loss
            else:
                return -3.0 - np.tanh((abs(pnl_pips) - 30) / 30)  # Large loss


class SessionQueueManager:
    """Manages pre-validated session queue."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.conn = duckdb.connect(config.data_path, read_only=True)
        self.session_queue = Queue(maxsize=config.session_queue_size)
        self.validation_thread = None
        self.stop_flag = False

        # Get data ranges
        result = self.conn.execute("""
            SELECT MIN(bar_index) as min_idx, MAX(bar_index) as max_idx
            FROM micro_features
        """).fetchone()

        self.min_idx = result[0]
        self.max_idx = result[1]

        # Split data 70/15/15
        total_bars = self.max_idx - self.min_idx
        self.train_end = self.min_idx + int(total_bars * 0.7)
        self.val_end = self.min_idx + int(total_bars * 0.85)

    def start(self):
        """Start background session validation."""
        self.stop_flag = False
        self.validation_thread = threading.Thread(target=self._validation_loop)
        self.validation_thread.start()

    def stop(self):
        """Stop background validation."""
        self.stop_flag = True
        if self.validation_thread:
            self.validation_thread.join()

    def _validation_loop(self):
        """Continuously validate and queue sessions."""
        while not self.stop_flag:
            if self.session_queue.qsize() < self.config.session_queue_size // 2:
                # Need more sessions
                session = self._find_valid_session()
                if session:
                    try:
                        self.session_queue.put(session, timeout=1)
                    except queue.Full:
                        pass
            else:
                time.sleep(0.1)

    def _find_valid_session(self) -> Optional[TradingSession]:
        """Find and validate a random session."""
        max_attempts = 20

        for _ in range(max_attempts):
            # Random start in training range
            start_idx = random.randint(
                self.min_idx,
                self.train_end - self.config.session_length - 100
            )

            # Check session validity
            query = f"""
            SELECT bar_index, timestamp
            FROM micro_features
            WHERE bar_index >= {start_idx}
            AND bar_index < {start_idx + self.config.session_length}
            ORDER BY bar_index
            LIMIT {self.config.session_length}
            """

            rows = self.conn.execute(query).fetchall()

            if len(rows) < self.config.session_length:
                continue

            # Check for gaps
            has_gaps = False
            prev_time = None

            for idx, timestamp_str in rows:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Check weekend
                if timestamp.weekday() == 4 and timestamp.hour >= 21:  # Friday close
                    has_gaps = True
                    break
                if timestamp.weekday() == 6 and timestamp.hour < 22:  # Sunday open
                    has_gaps = True
                    break

                # Check time gaps
                if prev_time:
                    gap_minutes = (timestamp - prev_time).total_seconds() / 60
                    if gap_minutes > 10:
                        has_gaps = True
                        break

                prev_time = timestamp

            if not has_gaps:
                return TradingSession(
                    start_idx=start_idx,
                    end_idx=start_idx + self.config.session_length,
                    timestamp=rows[0][1],
                    has_gaps=False
                )

        return None


class MicroMuZeroTrainer:
    """Main trainer with AMDDP1 and multiprocessing."""

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

        # Initialize optimizer (fixed learning rate)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Experience buffer
        self.buffer = deque(maxlen=config.buffer_size)

        # Multiprocessing setup
        self.manager = Manager()
        self.model_weights = self.manager.dict()
        self.experience_queue = Queue(maxsize=1000)

        # Session queue manager
        self.session_manager = SessionQueueManager(config)

        # Worker processes
        self.workers = []

        # Training stats
        self.episode = 0
        self.total_steps = 0
        self.current_temperature = config.initial_temperature

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def start_workers(self):
        """Start worker processes."""
        self.session_manager.start()

        for i in range(self.config.num_workers):
            worker = SessionCollector(
                worker_id=i,
                config=self.config,
                session_queue=self.session_manager.session_queue,
                experience_queue=self.experience_queue,
                model_weights=self.model_weights
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.config.num_workers} workers")

    def stop_workers(self):
        """Stop all workers."""
        # Send poison pills
        for _ in self.workers:
            self.session_manager.session_queue.put(None)

        # Wait for workers
        for worker in self.workers:
            worker.join()

        self.session_manager.stop()
        logger.info("All workers stopped")

    def update_model_weights(self):
        """Share model weights with workers."""
        self.model_weights.update(self.model.get_weights())

    def collect_experiences(self, num_experiences: int):
        """Collect experiences from workers."""
        collected = []
        timeout = 0.1
        max_wait = 30  # Maximum wait time

        start_time = time.time()
        while len(collected) < num_experiences:
            if time.time() - start_time > max_wait:
                logger.warning(f"Timeout collecting experiences: got {len(collected)}/{num_experiences}")
                break

            try:
                exp = self.experience_queue.get(timeout=timeout)
                collected.append(exp)
            except queue.Empty:
                continue

        return collected

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
            target_policies,
            label_smoothing=0.01
        )

        predicted_values = self.model.value.get_value(value_probs)
        value_loss = nn.functional.huber_loss(
            predicted_values,
            target_values,
            delta=1.0
        )

        total_loss = policy_loss + value_loss

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

    def update_temperature(self):
        """Update exploration temperature with 1% decay per 1000 episodes."""
        # Calculate decay based on episodes
        decay_factor = (1 - self.config.temperature_decay_per_1000) ** (self.episode / 1000)
        self.current_temperature = (
            self.config.initial_temperature * decay_factor +
            self.config.final_temperature * (1 - decay_factor)
        )

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'model_state': self.model.get_weights(),
            'optimizer_state': self.optimizer.state_dict(),
            'temperature': self.current_temperature,
            'buffer_size': len(self.buffer)
        }

        # Save latest and numbered
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"micro_checkpoint_ep{self.episode:06d}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, "latest.pth"))

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """Main training loop with multiprocessing."""
        logger.info("Starting training with AMDDP1 and multiprocessing...")

        # Start workers
        self.start_workers()

        try:
            while self.episode < self.config.num_episodes:
                # Update model weights for workers
                if self.episode % 10 == 0:
                    self.update_model_weights()

                # Collect experiences in parallel
                new_experiences = self.collect_experiences(100)
                self.buffer.extend(new_experiences)

                # Train when buffer is ready
                if len(self.buffer) >= self.config.min_buffer_size:
                    # Sample batch
                    batch_indices = np.random.choice(
                        len(self.buffer),
                        size=min(self.config.batch_size, len(self.buffer)),
                        replace=False
                    )
                    batch = [self.buffer[i] for i in batch_indices]

                    # Train
                    losses = self.train_batch(batch)

                    # Update temperature
                    self.update_temperature()

                    # Logging
                    if self.episode % self.config.log_interval == 0:
                        logger.info(
                            f"Episode {self.episode} | "
                            f"Loss: {losses['total_loss']:.4f} | "
                            f"Policy: {losses['policy_loss']:.4f} | "
                            f"Value: {losses['value_loss']:.4f} | "
                            f"Temp: {self.current_temperature:.3f} | "
                            f"Buffer: {len(self.buffer)}"
                        )

                    # Save checkpoint
                    if self.episode % self.config.save_interval == 0:
                        self.save_checkpoint()

                self.episode += 1
                self.total_steps += len(new_experiences)

        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            self.stop_workers()
            self.save_checkpoint()

        logger.info("Training complete")


if __name__ == "__main__":
    config = TrainingConfig()
    trainer = MicroMuZeroTrainer(config)
    trainer.train()