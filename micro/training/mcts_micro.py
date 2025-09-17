#!/usr/bin/env python3
"""
MCTS (Monte Carlo Tree Search) for Micro Stochastic MuZero.

Optimized for 15-feature micro variant with faster search.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class MinMaxStats:
    """Statistics for value normalization."""
    minimum: float = float('inf')
    maximum: float = float('-inf')

    def update(self, value: float):
        """Update min/max statistics."""
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    """MCTS tree node."""

    def __init__(
        self,
        prior: float,
        hidden_state: torch.Tensor,
        reward: float = 0.0,
        to_play: int = 0
    ):
        """
        Initialize node.

        Args:
            prior: Prior probability from policy network
            hidden_state: Hidden state representation
            reward: Immediate reward
            to_play: Player to play (0 for single-player)
        """
        self.prior = prior
        self.hidden_state = hidden_state
        self.reward = reward
        self.to_play = to_play

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> Node
        self.is_expanded = False

    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        """Check if node has been expanded."""
        return self.is_expanded

    def expand(
        self,
        actions: int,
        priors: np.ndarray,
        hidden_states: List[torch.Tensor],
        rewards: List[float]
    ):
        """
        Expand node with children.

        Args:
            actions: Number of actions
            priors: Prior probabilities for each action
            hidden_states: Hidden states after each action
            rewards: Rewards for each action
        """
        self.is_expanded = True
        for action in range(actions):
            self.children[action] = Node(
                prior=priors[action],
                hidden_state=hidden_states[action],
                reward=rewards[action],
                to_play=self.to_play
            )

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """Add Dirichlet noise to priors for exploration."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = (
                self.children[a].prior * (1 - exploration_fraction) +
                n * exploration_fraction
            )


class MCTS:
    """
    Monte Carlo Tree Search for Micro MuZero.

    Optimized for fast search with 15 features.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int = 4,
        discount: float = 0.997,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        num_simulations: int = 25  # Increased for better search,
        dirichlet_alpha: float = 0.5,   # More exploration noise
        exploration_fraction: float = 0.4
    ):
        """
        Initialize MCTS.

        Args:
            model: MicroStochasticMuZero model
            num_actions: Number of actions (4: Hold, Buy, Sell, Close)
            discount: Discount factor
            pb_c_base: Base for UCB formula
            pb_c_init: Init for UCB formula
            num_simulations: Number of MCTS simulations (reduced to 15 for speed)
            dirichlet_alpha: Dirichlet noise parameter
            exploration_fraction: Fraction of exploration noise
        """
        self.model = model
        self.num_actions = num_actions
        self.discount = discount
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction

    def run(
        self,
        observation: torch.Tensor,
        add_exploration_noise: bool = True,
        temperature: float = 1.0
    ) -> Dict:
        """
        Run MCTS from given observation.

        Args:
            observation: Input observation (batch=1, 32, 15)
            add_exploration_noise: Whether to add exploration noise
            temperature: Temperature for action selection

        Returns:
            Dictionary with:
                - action: Selected action
                - policy: Action probabilities
                - value: Root value
                - tree_depth: Maximum tree depth
        """
        # Initial inference
        with torch.no_grad():
            hidden, policy_logits, value_probs = self.model.initial_inference(observation)

            # Convert to numpy
            priors = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            root_value = self.model.value.get_value(value_probs).item()

        # Initialize root node
        root = Node(
            prior=0,
            hidden_state=hidden,
            reward=0,
            to_play=0
        )

        # Expand root with policy predictions
        hidden_states = [hidden.clone() for _ in range(self.num_actions)]
        rewards = [0.0] * self.num_actions
        root.expand(self.num_actions, priors, hidden_states, rewards)

        # Add exploration noise to root
        if add_exploration_noise:
            root.add_exploration_noise(
                self.dirichlet_alpha,
                self.exploration_fraction
            )

        # Initialize statistics
        min_max_stats = MinMaxStats()

        # Run simulations
        max_depth = 0
        for _ in range(self.num_simulations):
            # Selection
            path = [root]
            node = root
            depth = 0

            while node.expanded():
                action, node = self.select_action(node, min_max_stats)
                path.append(node)
                depth += 1

            # Track maximum depth
            max_depth = max(max_depth, depth)

            # Expansion (if not already expanded)
            parent = path[-2] if len(path) >= 2 else root
            action = self.get_action_from_parent(parent, node)

            if action is not None and not node.expanded():
                # Get action as one-hot
                action_one_hot = torch.zeros(
                    1, self.num_actions,
                    device=parent.hidden_state.device
                )
                action_one_hot[0, action] = 1

                # Recurrent inference
                with torch.no_grad():
                    next_hidden, reward, policy_logits, value_probs = \
                        self.model.recurrent_inference(
                            parent.hidden_state,
                            action_one_hot
                        )

                    # Convert to numpy
                    priors = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    value = self.model.value.get_value(value_probs).item()

                # Expand node
                hidden_states = [next_hidden.clone() for _ in range(self.num_actions)]
                rewards = [reward.item()] * self.num_actions
                node.expand(self.num_actions, priors, hidden_states, rewards)

            else:
                # Leaf node value
                with torch.no_grad():
                    value_probs = self.model.value(node.hidden_state)
                    value = self.model.value.get_value(value_probs).item()

            # Backup
            self.backup(path, value, min_max_stats)

        # Select action based on visit counts
        visits = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(self.num_actions)
        ])

        # Apply temperature
        if temperature == 0:
            # Deterministic (argmax)
            action = np.argmax(visits)
            policy = np.zeros(self.num_actions)
            policy[action] = 1.0
        else:
            # Stochastic with temperature
            visits_temp = np.power(visits, 1.0 / temperature)
            policy = visits_temp / visits_temp.sum()
            action = np.random.choice(self.num_actions, p=policy)

        return {
            'action': action,
            'policy': policy,
            'value': root.value(),
            'tree_depth': max_depth,
            'visit_counts': visits
        }

    def select_action(
        self,
        node: Node,
        min_max_stats: MinMaxStats
    ) -> Tuple[int, Node]:
        """
        Select action using UCB formula.

        Args:
            node: Current node
            min_max_stats: Value normalization statistics

        Returns:
            Selected action and child node
        """
        # Calculate UCB scores for all actions
        ucb_scores = []
        for action in range(self.num_actions):
            if action in node.children:
                child = node.children[action]
                ucb = self.ucb_score(
                    node,
                    child,
                    min_max_stats
                )
            else:
                ucb = float('inf')
            ucb_scores.append(ucb)

        # Select action with highest UCB
        action = np.argmax(ucb_scores)
        return action, node.children[action]

    def ucb_score(
        self,
        parent: Node,
        child: Node,
        min_max_stats: MinMaxStats
    ) -> float:
        """
        Calculate UCB score for child node.

        UCB = Q + P * sqrt(parent_visits) / (1 + child_visits) * c

        Args:
            parent: Parent node
            child: Child node
            min_max_stats: Value normalization statistics

        Returns:
            UCB score
        """
        pb_c = math.log(
            (parent.visit_count + self.pb_c_base + 1) / self.pb_c_base
        ) + self.pb_c_init

        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        # Normalize value
        value_score = min_max_stats.normalize(child.value())

        return value_score + child.prior * pb_c

    def backup(
        self,
        path: List[Node],
        value: float,
        min_max_stats: MinMaxStats
    ):
        """
        Backup value through path.

        Args:
            path: Path from root to leaf
            value: Leaf value
            min_max_stats: Value normalization statistics
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(node.value())

            # Apply discount and add reward
            value = node.reward + self.discount * value

    def get_action_from_parent(
        self,
        parent: Node,
        child: Node
    ) -> Optional[int]:
        """
        Get action that led from parent to child.

        Args:
            parent: Parent node
            child: Child node

        Returns:
            Action or None if not found
        """
        for action, node in parent.children.items():
            if node is child:
                return action
        return None