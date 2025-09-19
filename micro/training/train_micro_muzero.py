#!/usr/bin/env python3
"""
Fixed training script for Micro Stochastic MuZero with proper episode collection.
Runs full 360-bar episodes instead of single experiences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import time
import os
import json
import logging
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

    # MCTS
    num_simulations: int = 25
    discount: float = 0.997

    # Training
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    batch_size: int = 64
    gradient_clip: float = 1.0
    buffer_size: int = 50000
    min_buffer_size: int = 3600  # 10 episodes worth

    # Episode collection
    num_workers: int = 4
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
    num_episodes: int = 100000
    save_interval: int = 50  # Save every 50 episodes
    log_interval: int = 10
    validation_interval: int = 100


@dataclass
class Experience:
    """Single experience for replay buffer."""
    observation: np.ndarray
    action: int
    reward: float
    policy: np.ndarray
    value: float
    done: bool
    market_outcome: int
    outcome_probs: np.ndarray
    td_error: float = 0.0  # TD error for prioritization
    priority: float = 1.0  # Sampling priority
    visit_count: int = 0  # Number of times sampled


class QualityExperienceBuffer:
    """Enhanced replay buffer with TD error tracking and smart eviction."""

    def __init__(self, capacity: int = 50000, priority_alpha: float = 0.7, priority_beta: float = 0.5):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priority_alpha = priority_alpha  # How much to prioritize high TD error
        self.priority_beta = priority_beta  # Importance sampling correction
        self.max_priority = 1.0
        self.trade_diversity_threshold = 0.3  # Ensure 30% have trades

    def add(self, experience: Experience):
        """Add experience to buffer with initial high priority."""
        # New experiences get max priority to ensure they're trained on
        experience.priority = self.max_priority
        experience.visit_count = 0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # Smart eviction: prefer removing low-priority, high-visit experiences
            eviction_scores = np.array([
                exp.priority / (1 + exp.visit_count * 0.1) for exp in self.buffer
            ])
            evict_idx = np.argmin(eviction_scores)
            self.buffer[evict_idx] = experience

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray]:
        """Sample batch with priority-based sampling and importance weights."""
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
            batch = self.buffer.copy()
            weights = np.ones(len(self.buffer))
        else:
            # Calculate sampling probabilities
            priorities = np.array([exp.priority for exp in self.buffer])
            probs = priorities ** self.priority_alpha
            probs = probs / probs.sum()

            # Sample with priority
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            batch = [self.buffer[idx] for idx in indices]

            # Calculate importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.priority_beta)
            weights = weights / weights.max()  # Normalize

            # Ensure trade diversity (at least 30% should have trades)
            trade_experiences = [i for i, exp in enumerate(batch)
                               if exp.action in [1, 2, 3]]  # BUY, SELL, CLOSE
            if len(trade_experiences) < batch_size * self.trade_diversity_threshold:
                # Add more trade experiences
                trade_pool = [i for i, exp in enumerate(self.buffer)
                            if exp.action in [1, 2, 3] and i not in indices]
                if trade_pool:
                    n_needed = int(batch_size * self.trade_diversity_threshold) - len(trade_experiences)
                    n_replace = min(n_needed, len(trade_pool))
                    replace_indices = np.random.choice(
                        [i for i, exp in enumerate(batch) if exp.action == 0][:n_replace],
                        n_replace, replace=False
                    )
                    new_trade_indices = np.random.choice(trade_pool, n_replace, replace=False)
                    for i, new_idx in zip(replace_indices, new_trade_indices):
                        batch[i] = self.buffer[new_idx]
                        weights[i] = 1.0  # Reset weight for diversity samples

        # Update visit counts
        for exp in batch:
            exp.visit_count += 1

        return batch, weights

    def update_priorities(self, experiences: List[Experience], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for exp, td_error in zip(experiences, td_errors):
            exp.td_error = abs(td_error)
            exp.priority = (abs(td_error) + 1e-6) ** self.priority_alpha
            self.max_priority = max(self.max_priority, exp.priority)

    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.buffer:
            return {}

        td_errors = [exp.td_error for exp in self.buffer]
        priorities = [exp.priority for exp in self.buffer]
        visit_counts = [exp.visit_count for exp in self.buffer]
        action_counts = {i: sum(1 for exp in self.buffer if exp.action == i) for i in range(4)}

        return {
            'mean_td_error': np.mean(td_errors),
            'max_td_error': np.max(td_errors),
            'mean_priority': np.mean(priorities),
            'mean_visits': np.mean(visit_counts),
            'action_distribution': action_counts,
            'trade_ratio': (action_counts[1] + action_counts[2] + action_counts[3]) / len(self.buffer)
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

        # Initialize model
        self.model = MicroStochasticMuZero(
            input_features=config.input_features,
            lag_window=config.lag_window,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            num_outcomes=config.num_outcomes,
            support_size=config.support_size
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Initialize MCTS
        self.mcts = StochasticMCTS(
            model=self.model,
            num_simulations=config.num_simulations,
            discount=config.discount,
            depth_limit=3,
            dirichlet_alpha=1.0,
            exploration_fraction=0.5
        )

        # Initialize quality experience buffer with TD error tracking
        self.buffer = QualityExperienceBuffer(
            capacity=config.buffer_size,
            priority_alpha=0.7,
            priority_beta=0.5
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
            'input_features': self.config.input_features,
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
        """Single training step with TD error calculation."""
        if self.buffer.size() < self.config.batch_size:
            return {}

        # Sample batch with importance weights
        batch, importance_weights = self.buffer.sample(self.config.batch_size)

        # Prepare batch tensors
        observations = torch.tensor(
            np.array([e.observation for e in batch]),
            dtype=torch.float32, device=self.device
        )
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

        # Forward pass
        hidden = self.model.representation(observations)
        policy_logits, value_pred = self.model.prediction(hidden)

        # Outcome predictions
        outcome_pred = self.model.outcome_probability(hidden, actions.unsqueeze(1))

        # Convert importance weights to tensor
        weights = torch.tensor(importance_weights, dtype=torch.float32, device=self.device)

        # Calculate TD errors for priority updates
        with torch.no_grad():
            td_errors = (rewards + self.config.discount * values - value_pred.squeeze()).cpu().numpy()

        # Update priorities in buffer
        self.buffer.update_priorities(batch, td_errors)

        # Calculate weighted losses
        policy_loss = -(weights * (policies * torch.log_softmax(policy_logits, dim=1)).sum(1)).mean()
        value_loss = (weights * (value_pred.squeeze() - values) ** 2).mean()
        outcome_loss = (weights * nn.CrossEntropyLoss(reduction='none')(outcome_pred, outcomes)).mean()

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
            'mean_td_error': buffer_stats.get('mean_td_error', 0.0),
            'max_td_error': buffer_stats.get('max_td_error', 0.0),
            'trade_ratio': buffer_stats.get('trade_ratio', 0.0)
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
            # Initial buffer filling
            logger.info(f"Filling initial buffer (need {self.config.min_buffer_size} experiences)...")
            while self.buffer.size() < self.config.min_buffer_size:
                episodes_needed = max(1, (self.config.min_buffer_size - self.buffer.size()) // 360)
                episodes_to_collect = min(10, episodes_needed)
                self.collect_episodes(episodes_to_collect, split='train')
                logger.info(f"Buffer size: {self.buffer.size()}/{self.config.min_buffer_size}")

            logger.info("Initial buffer filled, starting training...")

            # Main training loop
            while self.episode < self.config.num_episodes:
                # Collect new episodes
                self.collect_episodes(self.config.episodes_per_iteration, split='train')

                # Train on batch
                losses = self.train_step()

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
    config = TrainingConfig()

    # Check for resume
    latest_path = os.path.join(config.checkpoint_dir, 'latest.pth')

    trainer = MicroMuZeroTrainer(config)

    if os.path.exists(latest_path):
        trainer.resume_from_checkpoint(latest_path)

    trainer.train()


if __name__ == "__main__":
    main()