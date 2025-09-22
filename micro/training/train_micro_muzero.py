#!/usr/bin/env python3
"""
Fixed training script for Micro Stochastic MuZero with proper episode collection.
Runs full 360-bar episodes instead of single experiences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time
import os
import json
import logging
import multiprocessing as mp

# Performance optimizations
torch.set_num_threads(1)  # Better for multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent thread oversubscription
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append('/workspace')

from micro.models.micro_networks import MicroStochasticMuZero
from micro.training.stochastic_mcts import StochasticMCTS
from micro.training.parallel_episode_collector import ParallelEpisodeCollector
from micro.training.episode_runner import Episode
from micro.utils.session_index_calculator import SessionIndexCalculator
from micro.training.checkpoint_manager import cleanup_old_checkpoints

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Calculate optimal number of workers based on CPU count."""
    n_cores = mp.cpu_count()
    # Use most cores but leave some for OS/main thread
    # Generally n-1 or 75% of cores, whichever is larger (min 4)
    optimal = min(n_cores - 1, max(4, int(n_cores * 0.75)))
    logger.info(f"CPU cores available: {n_cores}, optimal workers: {optimal}")
    return optimal


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    temporal_features: int = 9  # Market (5) + Time (4)
    static_features: int = 6  # Position features
    lag_window: int = 32
    hidden_dim: int = 256
    action_dim: int = 4
    num_outcomes: int = 3  # UP, NEUTRAL, DOWN
    support_size: int = 300

    # MCTS - 1 simulation for fastest testing
    num_simulations: int = 1  # Minimal for testing (1 sim × 360 steps = 360 sims/episode)
    discount: float = 0.997
    depth_limit: int = 3  # FIXED at 3 - sweet spot for trading

    # Training
    learning_rate: float = 0.002  # FIXED - no decay
    weight_decay: float = 1e-5
    batch_size: int = 64
    gradient_clip: float = 1.0
    buffer_size: int = 10000  # As per README
    min_buffer_size: int = 100  # As per README

    # Episode collection
    num_workers: int = get_optimal_workers()  # Dynamically set based on CPU
    episodes_per_iteration: int = 2

    # Temperature
    initial_temperature: float = 10.0
    final_temperature: float = 1.0
    temperature_decay_episodes: float = 50000

    # Paths
    data_path: str = "/workspace/data/micro_features.duckdb"
    checkpoint_dir: str = "/workspace/micro/checkpoints"
    log_dir: str = "/workspace/micro/logs"
    cache_dir: str = "/workspace/micro/cache"

    # Training schedule
    num_episodes: int = 1000000  # Run for 1 million episodes
    save_interval: int = 50  # Save every 50 episodes
    log_interval: int = 10
    validation_interval: int = 100


@dataclass
class Experience:
    """Single experience for replay buffer."""
    observation: np.ndarray  # Can be either (32,15) or tuple of ((32,9), (6,))
    action: int
    reward: float
    policy: np.ndarray
    value: float
    done: bool
    market_outcome: int
    outcome_probs: np.ndarray
    # Note: td_error, priority, visit_count removed - no longer using priority replay


class BalancedReplayBuffer:
    """Simple FIFO replay buffer with quota-based eviction to maintain trade diversity.

    NO priority experience replay - just simple random sampling with trade quota.
    As described in README: maintains minimum 30% trading trajectories.
    """

    def __init__(self, capacity: int = 10000, trade_quota: float = 0.3):
        """
        Initialize balanced replay buffer.

        Args:
            capacity: Maximum buffer size (10000 as per README)
            trade_quota: Minimum fraction of trading experiences (30%)
        """
        self.capacity = capacity
        self.buffer = []
        self.trade_quota = trade_quota

    def add(self, experience: Experience):
        """Add experience to buffer using quota-based eviction."""
        if len(self.buffer) >= self.capacity:
            self._evict_with_quota()
        self.buffer.append(experience)

    def _evict_with_quota(self):
        """Evict experience while maintaining trade quota.

        If below quota: evict random hold trajectory
        If above quota: standard FIFO eviction
        """
        trade_fraction = self._get_trade_fraction()

        if trade_fraction < self.trade_quota:
            # Below quota - try to evict a hold experience
            hold_indices = [i for i, exp in enumerate(self.buffer)
                          if exp.action == 0]  # HOLD action
            if hold_indices:
                evict_idx = np.random.choice(hold_indices)
            else:
                evict_idx = 0  # FIFO fallback if no holds
        else:
            # Above quota or balanced - standard FIFO
            evict_idx = 0

        self.buffer.pop(evict_idx)

    def _get_trade_fraction(self) -> float:
        """Calculate fraction of trading experiences in buffer."""
        if not self.buffer:
            return 0.0
        trade_count = sum(1 for exp in self.buffer if exp.action in [1, 2, 3])
        return trade_count / len(self.buffer)

    def sample(self, batch_size: int) -> List[Experience]:
        """Simple random sampling without priorities or weights.

        Returns:
            List of experiences (no importance weights)
        """
        if len(self.buffer) < batch_size:
            # Return all experiences if buffer is smaller than batch
            return self.buffer.copy()

        # Simple random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        # Ensure minimum trade diversity in sampled batch
        trade_count = sum(1 for exp in batch if exp.action in [1, 2, 3])
        if trade_count < batch_size * self.trade_quota:
            # Need more trade experiences in batch
            trade_pool = [i for i, exp in enumerate(self.buffer)
                         if exp.action in [1, 2, 3] and i not in indices]
            hold_in_batch = [i for i, exp in enumerate(batch) if exp.action == 0]

            if trade_pool and hold_in_batch:
                n_needed = int(batch_size * self.trade_quota) - trade_count
                n_replace = min(n_needed, len(trade_pool), len(hold_in_batch))

                # Replace some holds with trades
                replace_indices = np.random.choice(hold_in_batch, n_replace, replace=False)
                new_trade_indices = np.random.choice(trade_pool, n_replace, replace=False)

                for batch_idx, buffer_idx in zip(replace_indices, new_trade_indices):
                    batch[batch_idx] = self.buffer[buffer_idx]

        return batch

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.buffer:
            return {}

        action_counts = {i: sum(1 for exp in self.buffer if exp.action == i)
                        for i in range(4)}
        trade_fraction = self._get_trade_fraction()

        return {
            'buffer_size': len(self.buffer),
            'action_distribution': action_counts,
            'trade_ratio': trade_fraction,
            'hold_ratio': 1.0 - trade_fraction,
            'quota_satisfied': trade_fraction >= self.trade_quota
        }


class MicroMuZeroTrainer:
    """Fixed trainer with proper episode collection."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_expectancy = -float('inf')  # Initialize for first checkpoint

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model with separated architecture
        self.model = MicroStochasticMuZero(
            temporal_features=config.temporal_features,
            static_features=config.static_features,
            lag_window=config.lag_window,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_outcomes=config.num_outcomes,
            support_size=config.support_size
        ).to(self.device)

        # Optimize with torch.compile for faster execution (2-3x speedup)
        if not self.device.type == 'cuda':  # CPU optimization
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ Model optimized with torch.compile for CPU")
            except Exception as e:
                logger.info(f"⚠️ torch.compile not available: {e}")

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize MCTS with 3x3 configuration
        self.mcts = StochasticMCTS(
            model=self.model,
            num_simulations=config.num_simulations,
            discount=config.discount,
            depth_limit=config.depth_limit,  # Use config value (3)
            dirichlet_alpha=1.0,
            exploration_fraction=0.5
        )

        # Initialize balanced replay buffer (NO priority replay)
        self.buffer = BalancedReplayBuffer(
            capacity=config.buffer_size,  # 10000
            trade_quota=0.3  # Maintain 30% trade experiences
        )

        # Temperature for exploration
        self.current_temperature = config.initial_temperature
        self.temperature_decay_rate = (
            config.final_temperature / config.initial_temperature
        ) ** (1.0 / config.temperature_decay_episodes)

        # Training state
        self.episode = 0
        self.total_steps = 0

        # Initialize episode collector
        self.episode_collector = None

        # Training statistics
        self.training_stats = {
            'episodes': [],
            'losses': [],
            'expectancies': [],
            'win_rates': [],
            'trade_counts': [],
            'action_distributions': []
        }

        # Pre-calculate valid session indices
        self.ensure_session_indices()

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def ensure_session_indices(self):
        """Ensure valid session indices are pre-calculated."""
        calculator = SessionIndexCalculator(
            db_path=self.config.data_path,
            cache_path=os.path.join(self.config.cache_dir, "valid_session_indices.pkl")
        )
        indices = calculator.get_or_calculate_indices()
        logger.info(f"Session indices ready - Train: {len(indices['train'])}, "
                   f"Val: {len(indices['val'])}, Test: {len(indices['test'])}")

    def initialize_episode_collector(self):
        """Initialize parallel episode collector."""
        model_config = {
            'temporal_features': self.config.temporal_features,
            'static_features': self.config.static_features,
            'lag_window': self.config.lag_window,
            'hidden_dim': self.config.hidden_dim,
            'action_dim': self.config.action_dim,
            'num_outcomes': self.config.num_outcomes,
            'support_size': self.config.support_size
        }

        mcts_config = {
            'num_simulations': self.config.num_simulations,
            'discount': self.config.discount,
            'depth_limit': 3,
            'dirichlet_alpha': 1.0,
            'exploration_fraction': 0.5
        }

        # Save current model for workers
        temp_model_path = os.path.join(self.config.checkpoint_dir, "temp_model.pth")
        self.save_temp_model(temp_model_path)

        self.episode_collector = ParallelEpisodeCollector(
            model_path=temp_model_path,
            model_config=model_config,
            mcts_config=mcts_config,
            num_workers=self.config.num_workers,
            db_path=self.config.data_path,
            session_indices_path=os.path.join(self.config.cache_dir, "valid_session_indices.pkl")
        )
        self.episode_collector.start()
        logger.info(f"Episode collector started with {self.config.num_workers} workers")

    def save_temp_model(self, path: str):
        """Save model weights for workers."""
        torch.save({
            'model_state': self.model.get_weights(),
            'episode': self.episode,
            'temperature': self.current_temperature
        }, path)

    def collect_episodes(self, num_episodes: int, split: str = 'train') -> int:
        """Collect episodes and add to buffer."""
        # Update model for workers
        temp_model_path = os.path.join(self.config.checkpoint_dir, "temp_model.pth")
        self.save_temp_model(temp_model_path)

        # Collect episodes
        episodes = self.episode_collector.collect_episodes(
            num_episodes=num_episodes,
            split=split,
            temperature=self.current_temperature,
            add_noise=(split == 'train')
        )

        # Add experiences to buffer and track statistics
        experiences_added = 0
        total_trades = 0
        total_wins = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for episode in episodes:
            for exp in episode.experiences:
                # Convert to buffer experience
                buffer_exp = Experience(
                    observation=exp.observation,
                    action=exp.action,
                    reward=exp.reward,
                    policy=exp.policy,
                    value=exp.value,
                    done=exp.done,
                    market_outcome=exp.market_outcome,
                    outcome_probs=exp.outcome_probs
                )
                self.buffer.add(buffer_exp)
                experiences_added += 1
                action_counts[exp.action] += 1

            # Track episode statistics
            self.training_stats['expectancies'].append(episode.expectancy)
            self.training_stats['win_rates'].append(
                (episode.winning_trades / max(episode.num_trades, 1)) * 100
            )
            self.training_stats['trade_counts'].append(episode.num_trades)

            total_trades += episode.num_trades
            total_wins += episode.winning_trades

        # Log action distribution
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_dist = {
                'HOLD': action_counts[0] / total_actions,
                'BUY': action_counts[1] / total_actions,
                'SELL': action_counts[2] / total_actions,
                'CLOSE': action_counts[3] / total_actions
            }
            self.training_stats['action_distributions'].append(action_dist)

            logger.info(f"Action distribution - HOLD: {action_dist['HOLD']:.1%}, "
                       f"BUY: {action_dist['BUY']:.1%}, "
                       f"SELL: {action_dist['SELL']:.1%}, "
                       f"CLOSE: {action_dist['CLOSE']:.1%}")

        # Calculate metrics
        avg_expectancy = np.mean(self.training_stats['expectancies'][-num_episodes:])
        avg_win_rate = total_wins / max(total_trades, 1) * 100

        logger.info(f"Collected {len(episodes)} episodes, {experiences_added} experiences")
        logger.info(f"Expectancy: {avg_expectancy:.4f} pips, Win Rate: {avg_win_rate:.1f}%")

        return experiences_added

    def train_step(self) -> Dict[str, float]:
        """Single training step without priority replay."""
        if self.buffer.size() < self.config.batch_size:
            return {}

        # Simple random sampling (no importance weights)
        batch = self.buffer.sample(self.config.batch_size)

        # Prepare batch tensors - handle separated inputs (temporal, static)
        # Extract temporal and static components from observations
        temporal_batch = []
        static_batch = []
        for e in batch:
            temporal, static = e.observation
            temporal_batch.append(temporal)
            static_batch.append(static)

        temporal_obs = torch.tensor(
            np.array(temporal_batch),
            dtype=torch.float32, device=self.device
        )
        static_obs = torch.tensor(
            np.array(static_batch),
            dtype=torch.float32, device=self.device
        )
        observations = (temporal_obs, static_obs)
        actions = torch.tensor(
            [e.action for e in batch],
            dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [e.reward for e in batch],
            dtype=torch.float32, device=self.device
        )
        policies = torch.tensor(
            np.array([e.policy for e in batch]),
            dtype=torch.float32, device=self.device
        )
        values = torch.tensor(
            [e.value for e in batch],
            dtype=torch.float32, device=self.device
        )
        outcomes = torch.tensor(
            [e.market_outcome for e in batch],
            dtype=torch.long, device=self.device
        )
        outcome_probs = torch.tensor(
            np.array([e.outcome_probs for e in batch]),
            dtype=torch.float32, device=self.device
        )

        # Forward pass using initial_inference
        hidden, policy_logits, value_probs = self.model.initial_inference(observations)
        # Convert value distribution to scalar
        value_pred = self.model.value.get_value(value_probs)

        # Outcome predictions
        outcome_pred = self.model.outcome_probability(hidden, actions.unsqueeze(1))

        # Calculate losses (no weighting)
        policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(), values)
        outcome_loss = F.cross_entropy(outcome_pred, outcomes)

        # Combined loss with equal weighting
        total_loss = policy_loss + value_loss + outcome_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        # Get buffer stats
        buffer_stats = self.buffer.get_stats()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'outcome_loss': outcome_loss.item(),
            'buffer_size': buffer_stats.get('buffer_size', 0),
            'trade_ratio': buffer_stats.get('trade_ratio', 0.0),
            'quota_satisfied': buffer_stats.get('quota_satisfied', False)
        }

    def update_temperature(self):
        """Update exploration temperature."""
        self.current_temperature *= self.temperature_decay_rate
        self.current_temperature = max(
            self.current_temperature,
            self.config.final_temperature
        )

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'model_state': self.model.get_weights(),
            'optimizer_state': self.optimizer.state_dict(),
            'temperature': self.current_temperature,
            'training_stats': self.training_stats
        }

        # Save latest
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # Save periodic
        if self.episode % self.config.save_interval == 0:
            episode_path = os.path.join(
                self.config.checkpoint_dir,
                f'micro_checkpoint_ep{self.episode:06d}.pth'
            )
            torch.save(checkpoint, episode_path)
            logger.info(f"Checkpoint saved: {episode_path}")

            # Clean up old checkpoints (keep only last 2 + best + latest)
            cleanup_old_checkpoints(self.config.checkpoint_dir, keep_recent=2)

        # Save best based on expectancy
        # Always save first checkpoint as best for testing
        best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')

        if self.episode == 0 or self.episode == self.config.save_interval:
            # First checkpoint - always save as best
            torch.save(checkpoint, best_path)
            logger.info(f"First checkpoint saved as best: {best_path}")
            if len(self.training_stats['expectancies']) > 0:
                self.best_expectancy = np.mean(self.training_stats['expectancies'][-min(100, len(self.training_stats['expectancies'])):])
        elif len(self.training_stats['expectancies']) > 10:
            # After first checkpoint, use running average
            recent_expectancy = np.mean(self.training_stats['expectancies'][-min(100, len(self.training_stats['expectancies'])):])
            if recent_expectancy > self.best_expectancy:
                self.best_expectancy = recent_expectancy
                torch.save(checkpoint, best_path)
                logger.info(f"Best checkpoint saved: {best_path} (expectancy: {recent_expectancy:.4f})")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        # Initialize episode collector
        self.initialize_episode_collector()

        try:
            # Initial buffer filling - skip if resuming and already past initial episodes
            if self.episode == 0:
                logger.info(f"Filling initial buffer (need {self.config.min_buffer_size} experiences)...")
            while self.buffer.size() < self.config.min_buffer_size and self.episode == 0:
                episodes_needed = max(1, (self.config.min_buffer_size - self.buffer.size()) // 360)
                episodes_to_collect = min(10, episodes_needed)
                self.collect_episodes(episodes_to_collect, split='train')
                logger.info(f"Buffer size: {self.buffer.size()}/{self.config.min_buffer_size}")

            logger.info("Initial buffer filled, starting training...")

            # Main training loop
            while self.episode < self.config.num_episodes:
                try:
                    logger.info(f"Starting episode {self.episode}/{self.config.num_episodes}")
                    # Collect new episodes
                    self.collect_episodes(self.config.episodes_per_iteration, split='train')
                    logger.info(f"Collected episodes, now training...")

                    # Train on batch
                    losses = self.train_step()
                    logger.info(f"Training step completed with losses: {losses}")
                except Exception as e:
                    logger.error(f"Error in training loop: {e}", exc_info=True)
                    raise

                # Update temperature
                self.update_temperature()

                # Update episode count
                self.episode += self.config.episodes_per_iteration
                self.total_steps += self.config.episodes_per_iteration * 360

                # Enhanced logging with TD error and trade metrics
                if self.episode % self.config.log_interval == 0 and losses:
                    recent_expectancy = np.mean(self.training_stats['expectancies'][-100:])
                    recent_win_rate = np.mean(self.training_stats['win_rates'][-100:])

                    elapsed = time.time() - start_time
                    eps = self.episode / elapsed

                    logger.info(
                        f"Episode {self.episode} | Steps {self.total_steps} | "
                        f"EPS: {eps:.1f} | Loss: {losses['total_loss']:.4f} | "
                        f"TD: {losses.get('mean_td_error', 0):.3f} | "
                        f"Exp: {recent_expectancy:.2f} | WR: {recent_win_rate:.1f}% | "
                        f"TradeRatio: {losses.get('trade_ratio', 0):.1%}"
                    )

                # Save checkpoint with smart system
                self.save_checkpoint()  # Handles both frequent and periodic saves

                # Validation
                if self.episode % self.config.validation_interval == 0:
                    self.validate()

        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            # Clean up
            if self.episode_collector:
                self.episode_collector.stop()
            self.save_checkpoint()
            logger.info("Training complete")

    def validate(self):
        """Run validation episodes."""
        logger.info("Running validation...")

        # Collect validation episodes
        val_episodes = self.episode_collector.collect_episodes(
            num_episodes=10,
            split='val',
            temperature=0.0,  # Deterministic
            add_noise=False
        )

        # Calculate validation metrics
        total_reward = 0
        total_trades = 0
        winning_trades = 0
        total_pnl = 0

        for episode in val_episodes:
            total_reward += episode.total_reward
            total_trades += episode.num_trades
            winning_trades += episode.winning_trades
            total_pnl += episode.expectancy * episode.num_trades

        avg_expectancy = total_pnl / max(total_trades, 1)
        win_rate = (winning_trades / max(total_trades, 1)) * 100

        logger.info(f"Validation - Expectancy: {avg_expectancy:.4f} pips, "
                   f"Win Rate: {win_rate:.1f}%, Trades: {total_trades}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.set_weights(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint.get('total_steps', self.episode * 360)
        self.current_temperature = checkpoint.get('temperature', self.config.initial_temperature)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)

        logger.info(f"Resumed from episode {self.episode}, total steps {self.total_steps}")


def main():
    """Main entry point."""
    logger.info("Starting MicroMuZero training main()")

    try:
        config = TrainingConfig()

        # Check for resume
        latest_path = os.path.join(config.checkpoint_dir, 'latest.pth')

        trainer = MicroMuZeroTrainer(config)

        if os.path.exists(latest_path):
            logger.info(f"Found existing checkpoint at {latest_path}")
            trainer.resume_from_checkpoint(latest_path)
        else:
            logger.info("Starting fresh training - no checkpoint found")

        trainer.train()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        sys.exit(1)