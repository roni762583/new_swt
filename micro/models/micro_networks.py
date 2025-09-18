#!/usr/bin/env python3
"""
Micro Stochastic MuZero Neural Networks with TCN Integration.

5 networks for the micro variant:
1. Representation (with embedded TCN)
2. Dynamics
3. Policy
4. Value
5. Afterstate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

from .tcn_block import TCNBlock


class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class MLPResidualBlock(nn.Module):
    """MLP residual block for non-convolutional networks."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.bn1 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.bn2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class RepresentationNetwork(nn.Module):
    """
    Representation Network with embedded TCN encoder.

    Converts observation (32, 15) -> hidden state (256).
    """

    def __init__(
        self,
        input_features: int = 15,
        lag_window: int = 32,
        hidden_dim: int = 256,
        tcn_channels: int = 48,
        dropout: float = 0.1
    ):
        super().__init__()

        # TCN Front-End (integrated for end-to-end learning)
        self.tcn_encoder = TCNBlock(
            in_channels=input_features,
            out_channels=tcn_channels,  # Optimal compression to 48D
            kernel_size=3,
            dilations=[1, 2, 4],  # Receptive field = 15 (perfect for 32 lag)
            dropout=dropout,
            causal=True
        )

        # Temporal attention pooling (learns which timesteps matter)
        self.time_attention = nn.Linear(tcn_channels, 1)

        # Skip connection projection (48D TCN + 15D raw = 63D)
        self.projection = nn.Linear(tcn_channels + input_features, hidden_dim)

        # Standard residual blocks with dropout
        self.residual_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 32, 15)

        Returns:
            Hidden state (batch, 256)
        """
        batch_size = x.size(0)

        # TCN encoding with multi-scale temporal patterns
        tcn_out = self.tcn_encoder(x)  # (batch, 48, 32)
        tcn_out = self.dropout(tcn_out)  # Dropout after TCN

        # Attention-weighted temporal pooling
        attention_logits = self.time_attention(
            tcn_out.transpose(1, 2)
        )  # (batch, 32, 1)
        attention_weights = F.softmax(attention_logits, dim=1)
        pooled = (
            tcn_out * attention_weights.transpose(1, 2)
        ).sum(dim=2)  # (batch, 48)

        # Skip connection: combine temporal features with current state
        current_features = x[:, -1, :]  # Last timestep raw features (batch, 15)
        combined = torch.cat([pooled, current_features], dim=1)  # (batch, 63)

        # Project to hidden dimension
        hidden = self.projection(combined)  # (batch, 256)
        hidden = self.layer_norm(hidden)
        hidden = F.relu(hidden)

        # Apply residual blocks
        for block in self.residual_blocks:
            hidden = block(hidden)

        return hidden  # (batch, 256)


class OutcomeProbabilityNetwork(nn.Module):
    """
    Predicts market outcome probabilities given state and action.

    Outcomes based on rolling stdev:
    - UP: price change > 0.5 * rolling_stdev
    - DOWN: price change < -0.5 * rolling_stdev
    - NEUTRAL: price change within ±0.5 * rolling_stdev
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 4,
        num_outcomes: int = 3,  # UP, NEUTRAL, DOWN
        dropout: float = 0.1
    ):
        super().__init__()

        # Project state + action to hidden
        self.input_projection = nn.Linear(hidden_dim + action_dim, hidden_dim)

        # 2 residual blocks for outcome modeling
        self.outcome_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(2)
        ])

        # Output head for outcome probabilities
        self.outcome_head = nn.Linear(hidden_dim, num_outcomes)

    def forward(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict outcome probabilities.

        Args:
            hidden: Current hidden state (batch, 256)
            action: Action one-hot (batch, 4)

        Returns:
            Outcome probabilities (batch, 3) - softmax over [UP, NEUTRAL, DOWN]
        """
        # Concatenate state and action
        x = torch.cat([hidden, action], dim=-1)
        x = self.input_projection(x)

        # Process through residual blocks
        for block in self.outcome_blocks:
            x = block(x)

        # Get outcome probabilities
        logits = self.outcome_head(x)
        probs = F.softmax(logits, dim=-1)

        return probs


class DynamicsNetwork(nn.Module):
    """
    Stochastic Dynamics Network for state transition.

    Predicts next state and reward given current state, action, and market outcome.
    Now conditions on discrete market outcomes instead of continuous z.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 4,
        outcome_dim: int = 3,  # 3 market outcomes: UP, NEUTRAL, DOWN
        dropout: float = 0.1
    ):
        super().__init__()

        # Input: hidden(256) + action(4) + outcome(3) = 263D
        self.input_projection = nn.Linear(
            hidden_dim + action_dim + outcome_dim,
            hidden_dim
        )

        # 3 residual blocks for transition modeling
        self.dynamics_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])

        # Separate heads for next state and reward
        self.next_state_head = nn.Linear(hidden_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        outcome: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with market outcome conditioning.

        Args:
            hidden: Current hidden state (batch, 256)
            action: One-hot action (batch, 4)
            outcome: Market outcome one-hot or probs (batch, 3) [UP, NEUTRAL, DOWN]

        Returns:
            next_hidden: Next state (batch, 256)
            reward: Predicted reward (batch, 1)
        """
        # Combine inputs - now with outcome instead of z
        x = torch.cat([hidden, action, outcome], dim=1)  # (batch, 263)
        x = self.input_projection(x)
        x = F.relu(x)

        # Apply dynamics blocks
        for block in self.dynamics_blocks:
            x = block(x)

        # Generate next state and reward
        next_hidden = self.next_state_head(x)
        next_hidden = self.layer_norm(next_hidden)

        reward = self.reward_head(x)

        return next_hidden, reward


class PolicyNetwork(nn.Module):
    """
    Policy Network for action prediction.

    Outputs action logits with temperature scaling.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.temperature = temperature
        self.action_dim = action_dim

        # 2 residual blocks for policy
        self.policy_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(2)
        ])

        # Action head: 4 actions [Hold, Buy, Sell, Close]
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden: Hidden state (batch, 256)

        Returns:
            Action logits (batch, 4)
        """
        x = hidden

        # Apply policy blocks
        for block in self.policy_blocks:
            x = block(x)

        # Generate action logits with temperature scaling
        logits = self.action_head(x) / self.temperature

        return logits  # (batch, 4)


class ValueNetwork(nn.Module):
    """
    Value Network with categorical distribution.

    Predicts value as a distribution over support [-300, +300] pips.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        support_size: int = 300,
        dropout: float = 0.1
    ):
        super().__init__()

        # Categorical value distribution: [-300, +300] pips in 601 bins
        self.support_size = support_size
        self.num_atoms = 2 * support_size + 1  # 601

        # 2 residual blocks for value estimation
        self.value_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(2)
        ])

        # Value distribution head
        self.value_head = nn.Linear(hidden_dim, self.num_atoms)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden: Hidden state (batch, 256)

        Returns:
            Value distribution probs (batch, 601)
        """
        x = hidden

        # Apply value blocks
        for block in self.value_blocks:
            x = block(x)

        # Generate value distribution logits
        value_logits = self.value_head(x)  # (batch, 601)
        value_probs = F.softmax(value_logits, dim=1)

        return value_probs

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Convert distribution to scalar value."""
        support = torch.arange(
            -self.support_size,
            self.support_size + 1,
            device=probs.device,
            dtype=torch.float32
        )
        value = (probs * support).sum(dim=1, keepdim=True)
        return value


class AfterstateNetwork(nn.Module):
    """
    Afterstate Network for deterministic transition.

    Computes state after action but before stochastic transition.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        action_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input: hidden(256) + action(4) = 260D
        self.input_projection = nn.Linear(
            hidden_dim + action_dim,
            hidden_dim
        )

        # 2 residual blocks for afterstate modeling
        self.afterstate_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout=dropout) for _ in range(2)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden: Current hidden state (batch, 256)
            action: One-hot action (batch, 4)

        Returns:
            Afterstate (batch, 256)
        """
        # Combine hidden state with action
        x = torch.cat([hidden, action], dim=1)  # (batch, 260)
        x = self.input_projection(x)
        x = F.relu(x)

        # Apply afterstate blocks
        for block in self.afterstate_blocks:
            x = block(x)

        # Generate afterstate
        afterstate = self.layer_norm(x)

        return afterstate  # (batch, 256)


class MicroStochasticMuZero(nn.Module):
    """
    Complete Micro Stochastic MuZero model.

    Combines all networks with market outcome modeling for stochastic planning.
    """

    def __init__(
        self,
        input_features: int = 15,
        lag_window: int = 32,
        hidden_dim: int = 256,
        action_dim: int = 4,
        num_outcomes: int = 3,  # Market outcomes: UP, NEUTRAL, DOWN
        support_size: int = 300,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        # Initialize all networks
        self.representation = RepresentationNetwork(
            input_features=input_features,
            lag_window=lag_window,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.dynamics = DynamicsNetwork(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            outcome_dim=num_outcomes,
            dropout=dropout
        )

        self.outcome_predictor = OutcomeProbabilityNetwork(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_outcomes=num_outcomes,
            dropout=dropout
        )

        self.policy = PolicyNetwork(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            temperature=temperature,
            dropout=dropout
        )

        self.value = ValueNetwork(
            hidden_dim=hidden_dim,
            support_size=support_size,
            dropout=dropout
        )

        self.afterstate = AfterstateNetwork(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            dropout=dropout
        )

        # Store dimensions
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_outcomes = num_outcomes

        # Initialize weights properly to prevent NaN
        self._initialize_weights()

    def initial_inference(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial inference from observation.

        Args:
            observation: Input (batch, 32, 15)

        Returns:
            hidden: Hidden state (batch, 256)
            policy_logits: Action logits (batch, 4)
            value_probs: Value distribution (batch, 601)
        """
        hidden = self.representation(observation)
        policy_logits = self.policy(hidden)
        value_probs = self.value(hidden)

        return hidden, policy_logits, value_probs

    def recurrent_inference(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        outcome: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent inference for planning with market outcomes.

        Args:
            hidden: Current state (batch, 256)
            action: One-hot action (batch, 4)
            outcome: Market outcome probs (batch, 3) or None to predict

        Returns:
            next_hidden: Next state (batch, 256)
            reward: Predicted reward (batch, 1)
            policy_logits: Next action logits (batch, 4)
            value_probs: Next value distribution (batch, 601)
        """
        # Predict outcome if not provided
        if outcome is None:
            outcome = self.outcome_predictor(hidden, action)

        # Dynamics transition with market outcome
        next_hidden, reward = self.dynamics(hidden, action, outcome)

        # Policy and value for next state
        policy_logits = self.policy(next_hidden)
        value_probs = self.value(next_hidden)

        return next_hidden, reward, policy_logits, value_probs

    def predict_outcome(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict market outcome probabilities.

        Args:
            hidden: Current state (batch, 256)
            action: One-hot action (batch, 4)

        Returns:
            Outcome probabilities (batch, 3) [UP, NEUTRAL, DOWN]
        """
        return self.outcome_predictor(hidden, action)

    def compute_afterstate(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute afterstate for given state and action.

        Args:
            hidden: Current state (batch, 256)
            action: One-hot action (batch, 4)

        Returns:
            Afterstate (batch, 256)
        """
        return self.afterstate(hidden, action)

    def randomize_weights(self):
        """Force complete weight re-initialization for fresh start."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Randomizing all network weights for fresh start...")

        # Re-initialize all weights
        self._initialize_weights()

        # Extra randomization for policy network to encourage exploration
        with torch.no_grad():
            # Add noise to policy head
            if hasattr(self.policy, 'action_head'):
                self.policy.action_head.weight.add_(torch.randn_like(self.policy.action_head.weight) * 0.1)
                if self.policy.action_head.bias is not None:
                    self.policy.action_head.bias.add_(torch.randn_like(self.policy.action_head.bias) * 0.05)

        logger.info("Weight randomization complete")

    def _initialize_weights(self):
        """Initialize weights with aggressive randomization to break symmetry."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # More aggressive random initialization for policy head
                if module == self.policy.action_head:
                    # Random uniform for action head to encourage exploration
                    nn.init.uniform_(module.weight, -0.5, 0.5)
                    if module.bias is not None:
                        # Small random bias to break initial symmetry
                        nn.init.uniform_(module.bias, -0.1, 0.1)
                else:
                    # Xavier initialization for other linear layers
                    nn.init.xavier_uniform_(module.weight, gain=1.5)  # Higher gain
                    if module.bias is not None:
                        nn.init.normal_(module.bias, 0.0, 0.01)  # Small random bias
            elif isinstance(module, nn.Conv1d):
                # Kaiming initialization for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0.0, 0.01)
            elif isinstance(module, nn.BatchNorm1d):
                # Batch norm initialization
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                # LSTM initialization
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.0)

    def get_weights(self) -> dict:
        """Get model weights for checkpointing."""
        return {
            'representation': self.representation.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'afterstate': self.afterstate.state_dict()
        }

    def set_weights(self, weights: dict):
        """Load model weights from checkpoint."""
        self.representation.load_state_dict(weights['representation'])
        self.dynamics.load_state_dict(weights['dynamics'])
        self.policy.load_state_dict(weights['policy'])
        self.value.load_state_dict(weights['value'])
        self.afterstate.load_state_dict(weights['afterstate'])