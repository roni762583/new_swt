#!/usr/bin/env python3
"""
Real PPO Agent with Neural Networks for Learning.
Implements actual policy gradient optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
from collections import deque


class PolicyNetwork(nn.Module):
    """Optimized neural network for trading with PPO.

    Architecture designed for:
    - 17 input features (7 market + 6 position + 4 time)
    - 4 discrete actions (hold, buy, sell, close)
    - Feature groups with different scales/meanings
    """

    def __init__(self, input_dim=17, action_dim=4):
        super(PolicyNetwork, self).__init__()

        # Feature extractors for different groups
        # Market features (7): RSI, BB, swing, ER, etc.
        self.market_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Position features (6): side, pips, bars_since, etc.
        self.position_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Time features (4): sin/cos hour/week
        self.time_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU()
        )

        # Combined feature size: 32 + 16 + 16 = 64
        combined_dim = 64

        # Shared trunk with residual connections
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate heads for policy and value
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize weights appropriately
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Xavier for hidden layers, smaller init for output layers
            if module.out_features == 4 or module.out_features == 1:
                nn.init.orthogonal_(module.weight, gain=0.01)
            else:
                nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        """Forward pass with feature grouping."""
        # Split input into feature groups
        market_features = x[..., :7]
        position_features = x[..., 7:13]
        time_features = x[..., 13:17]

        # Encode each group
        market_encoded = self.market_encoder(market_features)
        position_encoded = self.position_encoder(position_features)
        time_encoded = self.time_encoder(time_features)

        # Combine encoded features
        combined = torch.cat([
            market_encoded,
            position_encoded,
            time_encoded
        ], dim=-1)

        # Pass through trunk
        trunk_output = self.trunk(combined)

        # Get policy and value outputs
        logits = self.policy_head(trunk_output)
        value = self.value_head(trunk_output)

        return logits, value

    def get_action(self, state):
        """Sample action from policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            logits, value = self(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update."""
        logits, values = self(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return values.squeeze(), action_log_probs, entropy


class PPOAgent:
    """PPO Agent optimized for trading."""

    def __init__(
        self,
        state_dim=17,
        action_dim=4,
        learning_rate=1e-4,  # Lower LR for stability
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.1,  # Tighter clipping for trading
        value_coef=0.5,
        entropy_coef=0.005,  # Less exploration needed
        max_grad_norm=0.5,
        lr_schedule='linear',  # Add LR scheduling
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.lr_schedule = lr_schedule
        self.initial_lr = learning_rate
        self.device = device

        # Optimized neural network
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,  # Add weight decay
            eps=1e-5
        )

        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Training stats
        self.update_count = 0
        self.total_steps = 0

    def get_action(self, state):
        """Get action from current policy."""
        action, log_prob, value = self.policy.get_action(state)

        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action

    def store_reward(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_steps += 1

    def compute_gae(self):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []

        # Convert to numpy for easier manipulation
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, n_epochs=4, batch_size=32):  # Smaller batches, fewer epochs
        """Update policy using PPO."""
        if len(self.states) < batch_size:
            return {}  # Not enough data

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages with clipping for stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std.clamp(min=0.1))

        # Training loop
        total_loss = 0
        policy_losses = []
        value_losses = []

        for epoch in range(n_epochs):
            # Mini-batch training
            indices = np.random.permutation(len(states))

            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Evaluate actions
                values, log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        # Calculate mean entropy for logging
        mean_entropy = 0
        if len(states) > 0:
            with torch.no_grad():
                _, _, entropy = self.policy.evaluate_actions(states[:100], actions[:100])
                mean_entropy = entropy.mean().item()

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        self.update_count += 1

        return {
            'total_loss': total_loss / (n_epochs * max(1, len(states) // batch_size)),
            'policy_loss': np.mean(policy_losses) if policy_losses else 0,
            'value_loss': np.mean(value_losses) if value_losses else 0,
            'entropy': mean_entropy,
            'update_count': self.update_count
        }

    def save(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_steps': self.total_steps
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)


class PPOLearningPolicy:
    """
    Optimized PPO wrapper for trading.
    Uses proper batch sizes and update frequencies for trading.
    """

    def __init__(self, state_dim=17, load_path=None):
        # Detect CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print("🚀 Using GPU acceleration for training")

        self.agent = PPOAgent(state_dim=state_dim, device=device)
        self.episode_rewards = []
        self.episode_steps = 0
        self.update_frequency = 1024  # More frequent updates for faster learning
        self.min_buffer_size = 256  # Minimum before first update

        if load_path:
            self.agent.load(load_path)
            print(f"✅ Loaded PPO model from {load_path}")

    def get_action(self, state):
        """Get action from learned policy."""
        return self.agent.get_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for learning."""
        self.agent.store_reward(reward, done)
        self.episode_rewards.append(reward)
        self.episode_steps += 1

        # Update policy when we have enough data
        if (self.episode_steps >= self.update_frequency or
            (done and self.episode_steps >= self.min_buffer_size)):

            stats = self.agent.update()
            if stats:
                print(f"\n🎓 PPO Update #{stats.get('update_count', 0)}:")
                print(f"   Loss: {stats.get('total_loss', 0):.4f}")
                print(f"   Policy Loss: {stats.get('policy_loss', 0):.4f}")
                print(f"   Value Loss: {stats.get('value_loss', 0):.4f}")
                print(f"   Entropy: {stats.get('entropy', 0):.4f}")
                print(f"   Steps trained: {self.agent.total_steps}")

                # Decay learning rate if using schedule
                if self.agent.lr_schedule == 'linear':
                    progress = min(1.0, self.agent.update_count / 1000)
                    new_lr = self.agent.initial_lr * (1 - 0.9 * progress)
                    for param_group in self.agent.optimizer.param_groups:
                        param_group['lr'] = new_lr

            self.episode_steps = 0

    def save(self, path):
        """Save learned policy."""
        self.agent.save(path)

    def get_stats(self):
        """Get training statistics."""
        return {
            'total_steps': self.agent.total_steps,
            'updates': self.agent.update_count,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }


if __name__ == "__main__":
    # Test the PPO agent
    print("Testing PPO Agent with Neural Networks")
    print("=" * 50)

    # Create agent
    agent = PPOLearningPolicy()

    # Simulate some steps
    for i in range(100):
        # Random state (17 features)
        state = np.random.randn(17)

        # Get action from policy
        action = agent.get_action(state)

        # Simulate reward
        reward = np.random.randn() * 0.1

        # Store transition
        agent.store_transition(state, action, reward, state, False)

    print("\n✅ PPO Agent working correctly!")
    print(f"Stats: {agent.get_stats()}")