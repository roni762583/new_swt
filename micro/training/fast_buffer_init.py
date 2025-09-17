#!/usr/bin/env python3
"""
Fast initial buffer collection using random/simple policy instead of expensive MCTS.

This dramatically speeds up initial buffer filling from hours to seconds.
"""

import torch
import numpy as np
from typing import Dict, List
import logging
from dataclasses import dataclass

from micro.training.train_micro_muzero import Experience

logger = logging.getLogger(__name__)


class FastBufferInitializer:
    """Fast initial experience collection without MCTS overhead."""

    def __init__(self, model, data_loader, device='cpu'):
        """
        Initialize fast buffer collector.

        Args:
            model: MicroStochasticMuZero model
            data_loader: DataLoader for getting windows
            device: Computation device
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def collect_random_experiences(self, num_experiences: int) -> List[Experience]:
        """
        Collect experiences using random actions (no MCTS).

        10-100x faster than MCTS for initial buffer.
        """
        experiences = []

        logger.info(f"Fast collecting {num_experiences} initial experiences...")

        for i in range(num_experiences):
            # Get random window
            window = self.data_loader.get_random_window(split='train')

            observation = torch.tensor(
                window['observation'],
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(0)

            # Get model predictions WITHOUT MCTS
            self.model.eval()
            with torch.no_grad():
                hidden, policy_logits, value = self.model.initial_inference(observation)

                # Add temperature for exploration
                temperature = 2.0  # Higher temperature for more exploration
                policy = torch.softmax(policy_logits / temperature, dim=1)

                # Sample action from policy
                action = torch.multinomial(policy, 1).item()

            # Create experience with simulated rewards
            # These will be replaced with real rewards during training
            base_reward = np.random.randn() * 10
            amddp1_reward = self._calculate_simple_amddp1(base_reward)

            experience = Experience(
                observation=observation.squeeze(0).cpu().numpy(),
                action=action,
                reward=amddp1_reward,
                value=value.item() if value.dim() == 0 else value.squeeze().item(),
                policy=policy.squeeze().cpu().numpy(),
                done=np.random.random() < 0.1,  # 10% chance of episode end
                pip_pnl=base_reward,
                position_change=action != 0,  # Any non-hold is position change
                session_expectancy=window.get('expectancy', 0.0)
            )

            # Calculate quality score
            experience.quality_score = experience.calculate_quality_score()
            experiences.append(experience)

            if (i + 1) % 100 == 0:
                logger.info(f"  Collected {i + 1}/{num_experiences}")

        logger.info(f"Fast collection complete! Collected {len(experiences)} experiences")
        return experiences

    def collect_guided_experiences(self, num_experiences: int) -> List[Experience]:
        """
        Collect experiences using simple heuristic policy.

        Slightly slower than random but produces better initial experiences.
        """
        experiences = []

        logger.info(f"Guided collecting {num_experiences} initial experiences...")

        for i in range(num_experiences):
            window = self.data_loader.get_random_window(split='train')

            observation = torch.tensor(
                window['observation'],
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(0)

            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                hidden, policy_logits, value = self.model.initial_inference(observation)

                # Use model's policy but with high temperature
                temperature = 1.5
                policy = torch.softmax(policy_logits / temperature, dim=1)

                # Bias towards hold (action 0) for safety
                policy[0, 0] *= 1.5  # Increase hold probability
                policy = policy / policy.sum()

                action = torch.multinomial(policy, 1).item()

            # Better reward simulation based on action
            if action == 0:  # Hold
                base_reward = np.random.randn() * 2  # Small variance
            elif action in [1, 2]:  # Buy/Sell
                base_reward = np.random.randn() * 15  # Higher variance
            else:  # Close
                base_reward = np.random.randn() * 10

            amddp1_reward = self._calculate_simple_amddp1(base_reward)

            experience = Experience(
                observation=observation.squeeze(0).cpu().numpy(),
                action=action,
                reward=amddp1_reward,
                value=value.item() if value.dim() == 0 else value.squeeze().item(),
                policy=policy.squeeze().cpu().numpy(),
                done=action == 3 or np.random.random() < 0.05,  # Close or 5% random
                pip_pnl=base_reward,
                position_change=action != 0,
                session_expectancy=window.get('expectancy', 0.0),
                trade_complete=action == 3  # Close completes trade
            )

            experience.quality_score = experience.calculate_quality_score()
            experiences.append(experience)

            if (i + 1) % 100 == 0:
                logger.info(f"  Collected {i + 1}/{num_experiences}")

        logger.info(f"Guided collection complete! Avg quality: {np.mean([e.quality_score for e in experiences]):.2f}")
        return experiences

    def _calculate_simple_amddp1(self, base_reward: float) -> float:
        """
        Simple AMDDP1 calculation without full environment.

        AMDDP1 = profit - drawdown_penalty
        """
        # Simulate drawdown (1% of absolute reward)
        drawdown_penalty = abs(base_reward) * 0.01

        # Add random noise
        noise = np.random.randn() * 0.5

        return base_reward - drawdown_penalty + noise


def initialize_buffer_fast(trainer, use_guided: bool = True):
    """
    Replace slow MCTS initial collection with fast collection.

    Args:
        trainer: MicroMuZeroTrainer instance
        use_guided: Use guided policy instead of pure random
    """
    initializer = FastBufferInitializer(
        model=trainer.model,
        data_loader=trainer.data_loader,
        device=trainer.device
    )

    # Collect experiences
    if use_guided:
        experiences = initializer.collect_guided_experiences(
            trainer.config.min_buffer_size
        )
    else:
        experiences = initializer.collect_random_experiences(
            trainer.config.min_buffer_size
        )

    # Add to buffer
    for exp in experiences:
        trainer.buffer.add(exp)

    logger.info(f"Buffer initialized with {len(trainer.buffer)} experiences")
    logger.info(f"Average quality score: {np.mean([e.quality_score for e in experiences]):.2f}")