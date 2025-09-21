#!/usr/bin/env python3
"""
Stochastic MCTS with Market Outcome Modeling.

Implements chance nodes for market uncertainty using 3 discrete outcomes:
- UP: price change > 0.33 * rolling_stdev
- NEUTRAL: price change within ±0.33 * rolling_stdev
- DOWN: price change < -0.33 * rolling_stdev
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketOutcome:
    """Market outcome enumeration."""
    UP: int = 0
    NEUTRAL: int = 1
    DOWN: int = 2

    @staticmethod
    def to_onehot(outcome: int, device: torch.device) -> torch.Tensor:
        """Convert outcome index to one-hot tensor."""
        onehot = torch.zeros(3, device=device)
        onehot[outcome] = 1.0
        return onehot


class DecisionNode:
    """
    Decision node in MCTS tree where agent chooses action.
    """

    def __init__(
        self,
        prior: float,
        hidden_state: torch.Tensor,
        reward: float = 0.0,
        to_play: int = 0
    ):
        """
        Initialize decision node.

        Args:
            prior: Prior probability from parent's policy
            hidden_state: Hidden state representation
            reward: Reward received reaching this node
            to_play: Player to play (0 for single-player)
        """
        self.prior = prior
        self.hidden_state = hidden_state
        self.reward = reward
        self.to_play = to_play

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> ChanceNode
        self.is_expanded = False

    def value(self) -> float:
        """Get mean value of node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        actions: int,
        priors: np.ndarray,
        outcome_probs: Dict[int, np.ndarray]
    ):
        """
        Expand node with chance nodes for each action.

        Args:
            actions: Number of actions
            priors: Prior probabilities for each action
            outcome_probs: Dict[action] -> outcome probabilities
        """
        self.is_expanded = True
        for action in range(actions):
            self.children[action] = ChanceNode(
                prior=priors[action],
                action=action,
                outcome_probabilities=outcome_probs[action],
                parent_hidden=self.hidden_state
            )

    def add_exploration_noise(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float
    ):
        """Add Dirichlet noise to action priors for exploration."""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = (
                self.children[a].prior * (1 - exploration_fraction) +
                n * exploration_fraction
            )


class ChanceNode:
    """
    Chance node representing market uncertainty after action.
    """

    def __init__(
        self,
        prior: float,
        action: int,
        outcome_probabilities: np.ndarray,
        parent_hidden: Optional[torch.Tensor] = None
    ):
        """
        Initialize chance node.

        Args:
            prior: Prior probability of taking this action
            action: Action that led to this chance node
            outcome_probabilities: Probabilities for [UP, NEUTRAL, DOWN]
            parent_hidden: Hidden state from parent decision node
        """
        self.prior = prior
        self.action = action
        self.outcome_probabilities = outcome_probabilities
        self.parent_hidden = parent_hidden

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # outcome -> DecisionNode
        self.is_expanded = False

    def value(self) -> float:
        """Get mean value of chance node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        hidden_states: List[torch.Tensor],
        rewards: List[float]
    ):
        """
        Expand chance node with decision nodes for each outcome.

        Args:
            hidden_states: Hidden states for each outcome
            rewards: Rewards for each outcome
        """
        self.is_expanded = True
        for outcome in range(3):  # UP, NEUTRAL, DOWN
            self.children[outcome] = DecisionNode(
                prior=self.outcome_probabilities[outcome],
                hidden_state=hidden_states[outcome],
                reward=rewards[outcome],
                to_play=0
            )


class StochasticMCTS:
    """
    Stochastic Monte Carlo Tree Search with market outcome modeling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int = 4,
        num_outcomes: int = 3,
        discount: float = 0.997,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        num_simulations: int = 25,
        dirichlet_alpha: float = 1.0,  # Strong exploration
        exploration_fraction: float = 0.5,  # High exploration
        depth_limit: int = 3  # Planning depth
    ):
        """
        Initialize Stochastic MCTS.

        Args:
            model: MicroStochasticMuZero model
            num_actions: Number of actions (4: Hold, Buy, Sell, Close)
            num_outcomes: Number of market outcomes (3: UP, NEUTRAL, DOWN)
            discount: Discount factor
            pb_c_base: Base for UCB formula
            pb_c_init: Init for UCB formula
            num_simulations: Number of MCTS simulations
            dirichlet_alpha: Dirichlet noise parameter
            exploration_fraction: Fraction of exploration noise
            depth_limit: Maximum planning depth
        """
        self.model = model
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.discount = discount
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.depth_limit = depth_limit

    def run(
        self,
        observation: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> Dict:
        """
        Run MCTS simulations from given observation.

        Args:
            observation: Either:
                - Legacy: Single tensor (batch, 32, 15)
                - Separated: Tuple of (temporal, static) tensors
                    temporal: (batch, 32, 9) - market + time features
                    static: (batch, 6) - position features
            temperature: Temperature for action selection
            add_noise: Whether to add exploration noise

        Returns:
            Dict with:
                - action: Selected action
                - policy: Action probabilities
                - value: Root value estimate
                - tree_stats: Search statistics
        """
        if isinstance(observation, tuple):
            logger.debug(f"MCTS.run starting with separated obs: temporal {observation[0].shape}, static {observation[1].shape}")
        else:
            logger.debug(f"MCTS.run starting with obs shape {observation.shape}")

        # Ensure we're in eval mode
        self.model.eval()

        # Use inference_mode instead of no_grad for better multiprocessing compatibility
        with torch.inference_mode():
            # Initial inference
            logger.debug("MCTS: Running initial inference...")
            try:
                # Ensure observation is on the correct device
                device = next(self.model.parameters()).device

                if isinstance(observation, tuple):
                    # Separated inputs
                    temporal, static = observation
                    if temporal.device != device:
                        temporal = temporal.to(device)
                    if static.device != device:
                        static = static.to(device)
                    observation = (temporal, static)
                else:
                    # Legacy single tensor
                    if observation.device != device:
                        observation = observation.to(device)

                hidden, policy_logits, value_probs = self.model.initial_inference(
                    observation
                )
            except Exception as e:
                logger.error(f"MCTS initial inference failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Return a random action as fallback
                return {
                    'action': np.random.randint(0, self.num_actions),
                    'policy': np.ones(self.num_actions) / self.num_actions,
                    'value': 0.0,
                    'tree_stats': {'error': str(e)}
                }
            logger.debug(f"MCTS: Initial inference done - hidden shape: {hidden.shape}")

            # Convert to numpy
            priors = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            root_value = self.model.value.get_value(value_probs).item()

            # Create root node
            logger.debug("MCTS: Creating root node...")
            root = DecisionNode(
                prior=1.0,
                hidden_state=hidden[0],
                reward=0.0,
                to_play=0
            )

            # Expand root with outcome predictions
            logger.debug("MCTS: Predicting outcomes for root expansion...")
            outcome_probs = {}
            for action in range(self.num_actions):
                action_onehot = torch.zeros(1, self.num_actions, device=hidden.device)
                action_onehot[0, action] = 1
                outcome_prob = self.model.predict_outcome(hidden, action_onehot)
                outcome_probs[action] = outcome_prob.cpu().numpy()[0]
            logger.debug(f"MCTS: Got outcome probs for {self.num_actions} actions")

            root.expand(self.num_actions, priors, outcome_probs)
            logger.debug("MCTS: Root expansion complete")

            # Add exploration noise at root
            if add_noise:
                root.add_exploration_noise(
                    self.dirichlet_alpha,
                    self.exploration_fraction
                )

            # Run simulations
            logger.debug(f"MCTS: Starting {self.num_simulations} simulations...")
            for sim in range(self.num_simulations):
                logger.debug(f"MCTS: Simulation {sim+1}/{self.num_simulations}")
                self._simulate(root, 0)
            logger.debug("MCTS: All simulations complete")

            # Extract policy from visit counts
            visits = [
                root.children[a].visit_count if a in root.children else 0
                for a in range(self.num_actions)
            ]

            # Apply temperature
            if temperature == 0:
                # Greedy selection
                action = int(np.argmax(visits))
                policy = np.zeros(self.num_actions)
                policy[action] = 1.0
            else:
                # Boltzmann selection
                visits_temp = np.array(visits) ** (1 / temperature)
                visits_sum = visits_temp.sum()
                if visits_sum > 0:
                    policy = visits_temp / visits_sum
                else:
                    # Uniform if no visits
                    policy = np.ones(self.num_actions) / self.num_actions
                action = int(np.random.choice(self.num_actions, p=policy))

            return {
                'action': action,
                'policy': policy,
                'value': root.value(),
                'tree_stats': {
                    'simulations': self.num_simulations,
                    'root_visits': visits,
                    'root_value': root_value,
                    'tree_value': root.value()
                }
            }

    def _simulate(self, node: Union[DecisionNode, ChanceNode], depth: int) -> float:
        """
        Simulate a single MCTS trajectory.

        Args:
            node: Current node (Decision or Chance)
            depth: Current depth in tree

        Returns:
            Value backup from simulation
        """
        # Terminal depth - use value network
        if depth >= self.depth_limit:
            if isinstance(node, DecisionNode):
                with torch.inference_mode():
                    value_probs = self.model.value(
                        node.hidden_state.unsqueeze(0)
                    )
                    value = self.model.value.get_value(value_probs).item()
                    return value
            else:
                # For chance nodes at terminal depth, average over outcomes
                return node.value()

        if isinstance(node, DecisionNode):
            # Decision node - select action
            if not node.is_expanded:
                # Expand and evaluate
                return self._expand_decision_node(node)

            # Select action using UCB
            action = self._select_action(node)
            chance_node = node.children[action]

            # Recurse through chance node
            value = self._simulate(chance_node, depth)

            # Backup
            node.visit_count += 1
            node.value_sum += value

            return value

        else:  # ChanceNode
            # Chance node - sample outcome
            if not node.is_expanded:
                # Expand and evaluate
                return self._expand_chance_node(node)

            # Sample outcome based on probabilities
            outcome = np.random.choice(3, p=node.outcome_probabilities)
            decision_node = node.children[outcome]

            # Recurse through decision node
            reward = decision_node.reward
            value = reward + self.discount * self._simulate(decision_node, depth + 1)

            # Backup
            node.visit_count += 1
            node.value_sum += value

            return value

    def _expand_decision_node(self, node: DecisionNode) -> float:
        """
        Expand a decision node and return its value.
        """
        with torch.inference_mode():
            hidden = node.hidden_state.unsqueeze(0)

            # Get policy and value
            policy_logits = self.model.policy(hidden)
            value_probs = self.model.value(hidden)

            priors = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = self.model.value.get_value(value_probs).item()

            # Get outcome predictions for each action
            outcome_probs = {}
            for action in range(self.num_actions):
                action_onehot = torch.zeros(1, self.num_actions, device=hidden.device)
                action_onehot[0, action] = 1
                outcome_prob = self.model.predict_outcome(hidden, action_onehot)
                outcome_probs[action] = outcome_prob.cpu().numpy()[0]

            node.expand(self.num_actions, priors, outcome_probs)

            # Update node value
            node.visit_count += 1
            node.value_sum += value

            return value

    def _expand_chance_node(self, node: ChanceNode) -> float:
        """
        Expand a chance node and return its expected value.
        """
        with torch.inference_mode():
            if node.parent_hidden is None:
                # Shouldn't happen but fallback to 0
                return 0.0

            hidden = node.parent_hidden.unsqueeze(0)
            action_onehot = torch.zeros(1, self.num_actions, device=hidden.device)
            action_onehot[0, node.action] = 1

            # Compute next states for each outcome
            hidden_states = []
            rewards = []

            for outcome in range(3):
                outcome_onehot = MarketOutcome.to_onehot(outcome, hidden.device).unsqueeze(0)
                next_hidden, reward = self.model.dynamics(
                    hidden, action_onehot, outcome_onehot
                )
                hidden_states.append(next_hidden[0])
                rewards.append(reward.item())

            node.expand(hidden_states, rewards)

            # Expected value over outcomes
            expected_value = 0.0
            for outcome in range(3):
                child = node.children[outcome]
                value_probs = self.model.value(
                    child.hidden_state.unsqueeze(0)
                )
                value = self.model.value.get_value(value_probs).item()
                expected_value += node.outcome_probabilities[outcome] * (
                    child.reward + self.discount * value
                )

            # Update node
            node.visit_count += 1
            node.value_sum += expected_value

            return expected_value

    def _select_action(self, node: DecisionNode) -> int:
        """
        Select action using UCB formula.
        """
        total_visits = sum(
            child.visit_count for child in node.children.values()
        )

        best_action = -1
        best_ucb = -float('inf')

        for action, chance_node in node.children.items():
            if chance_node.visit_count == 0:
                ucb = float('inf')  # Explore unvisited
            else:
                # UCB formula
                exploration_term = (
                    self.pb_c_init +
                    np.log((total_visits + self.pb_c_base + 1) / self.pb_c_base)
                ) * chance_node.prior

                exploration = exploration_term * np.sqrt(total_visits) / (1 + chance_node.visit_count)
                ucb = chance_node.value() + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action