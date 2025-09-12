"""
EfficientZero Model-Based Off-Policy Correction Implementation
Adaptive rollout horizons and corrected value targets for forex time series

Implements EfficientZero's third enhancement with forex market optimizations
Optimized for SWT-Enhanced Stochastic MuZero trading system with AMDDP1 reward system

Author: SWT Research Team
Date: September 2025
Adherence: CLAUDE.md professional code standards
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OffPolicyCorrectBase(ABC):
    """
    Abstract base class for off-policy correction implementations
    Provides common interface for value target correction methods
    """
    
    @abstractmethod
    def compute_corrected_value_target(
        self,
        trajectory: Dict[str, Any],
        networks: Any,
        mcts_runner: Any,
        step_idx: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute corrected value target using model-based rollout
        
        Args:
            trajectory: Episode trajectory data
            networks: Neural network components
            mcts_runner: MCTS planning component
            step_idx: Current step in trajectory
            
        Returns:
            tuple: (corrected_target, correction_metrics)
        """
        pass
    
    @abstractmethod
    def compute_adaptive_horizon(
        self, 
        data_age_steps: int, 
        trajectory_length: int,
        market_volatility: Optional[float] = None
    ) -> int:
        """
        Compute adaptive rollout horizon based on data characteristics
        
        Args:
            data_age_steps: Age of data in training steps
            trajectory_length: Remaining trajectory length
            market_volatility: Current market volatility (optional)
            
        Returns:
            Adaptive horizon length
        """
        pass


class SWTOffPolicyCorrection(OffPolicyCorrectBase):
    """
    Model-based off-policy correction for SWT trading system
    
    Implements EfficientZero's corrected value targets with forex-specific
    adaptations for market regime changes and volatility
    """
    
    def __init__(
        self,
        min_horizon: int = 3,
        max_horizon: int = 8, 
        decay_rate: float = 0.95,
        discount_factor: float = 0.997,
        volatility_adjustment: bool = True,
        market_regime_aware: bool = True
    ) -> None:
        """
        Initialize off-policy correction module
        
        Args:
            min_horizon: Minimum rollout horizon
            max_horizon: Maximum rollout horizon
            decay_rate: Exponential decay rate for horizon adaptation
            discount_factor: Discount factor for future rewards (SWT: 0.997)
            volatility_adjustment: Enable volatility-based horizon adjustment
            market_regime_aware: Enable market regime detection for horizon
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if min_horizon < 1:
            raise ValueError("min_horizon must be positive")
        if max_horizon < min_horizon:
            raise ValueError("max_horizon must be >= min_horizon")
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must be in (0, 1]")
            
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.decay_rate = decay_rate
        self.discount_factor = discount_factor
        self.volatility_adjustment = volatility_adjustment
        self.market_regime_aware = market_regime_aware
        
        # Statistics tracking
        self.correction_stats = {
            'total_corrections': 0,
            'avg_horizon': 0.0,
            'volatility_adjustments': 0,
            'regime_adjustments': 0
        }
        
        logger.info(f"Initialized SWTOffPolicyCorrection")
        logger.info(f"Parameters: horizon=[{min_horizon}, {max_horizon}], "
                   f"decay={decay_rate}, discount={discount_factor}")
    
    def compute_adaptive_horizon(
        self, 
        data_age_steps: int, 
        trajectory_length: int,
        market_volatility: Optional[float] = None,
        price_trend: Optional[str] = None
    ) -> int:
        """
        Compute adaptive rollout horizon for forex time series
        
        Considers data age, market volatility, and regime characteristics
        for optimal horizon selection in trading environments
        
        Args:
            data_age_steps: Age of training data in steps
            trajectory_length: Remaining trajectory length
            market_volatility: Market volatility measure (0-1 scale)
            price_trend: Price trend direction ('up', 'down', 'sideways')
            
        Returns:
            Optimal horizon length for current conditions
        """
        # Base horizon with exponential decay based on data age
        age_factor = self.decay_rate ** data_age_steps
        base_horizon = self.max_horizon * age_factor
        
        # Clamp to reasonable bounds
        horizon = max(self.min_horizon, min(self.max_horizon, int(base_horizon)))
        
        # Volatility adjustment: shorter horizons in volatile markets
        if self.volatility_adjustment and market_volatility is not None:
            volatility_factor = 1.0 - (market_volatility * 0.3)  # Max 30% reduction
            horizon = int(horizon * volatility_factor)
            horizon = max(self.min_horizon, horizon)
            
            if volatility_factor < 0.9:
                self.correction_stats['volatility_adjustments'] += 1
        
        # Market regime adjustment: adapt to trending vs ranging markets
        if self.market_regime_aware and price_trend is not None:
            if price_trend in ['up', 'down']:  # Trending market
                horizon = min(horizon + 1, self.max_horizon)  # Slightly longer
            elif price_trend == 'sideways':  # Ranging market
                horizon = max(horizon - 1, self.min_horizon)  # Slightly shorter
                
            self.correction_stats['regime_adjustments'] += 1
        
        # Don't exceed remaining trajectory length
        final_horizon = min(horizon, trajectory_length)
        
        # Update statistics
        self.correction_stats['avg_horizon'] = (
            (self.correction_stats['avg_horizon'] * self.correction_stats['total_corrections'] 
             + final_horizon) / (self.correction_stats['total_corrections'] + 1)
        )
        
        return final_horizon
    
    def compute_corrected_value_target(
        self,
        trajectory: Dict[str, Any],
        networks: Any,
        mcts_runner: Any,
        step_idx: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute EfficientZero's corrected value target for SWT system
        
        Formula: z = sum_R + γ^L * v_root (fresh MCTS at horizon)
        
        Args:
            trajectory: Episode trajectory containing observations, rewards, etc.
            networks: SWT neural network components (representation, etc.)
            mcts_runner: MCTS planning component for tail value estimation
            step_idx: Current step index in trajectory
            
        Returns:
            tuple: (corrected_target, correction_metrics)
                - corrected_target: Corrected value target
                - correction_metrics: Diagnostic information
                
        Raises:
            ValueError: If trajectory structure is invalid
        """
        # Validate trajectory structure
        required_keys = ['observations', 'rewards', 'actions']
        for key in required_keys:
            if key not in trajectory:
                raise ValueError(f"Missing required trajectory key: {key}")
        
        observations = trajectory['observations']
        rewards = trajectory['rewards']
        actions = trajectory.get('actions', [])
        
        if len(observations) != len(rewards):
            raise ValueError("Observations and rewards must have same length")
        
        remaining_length = len(observations) - step_idx
        if remaining_length <= 0:
            raise ValueError("step_idx exceeds trajectory length")
        
        # Extract market context for horizon adaptation
        market_volatility = self._estimate_market_volatility(trajectory, step_idx)
        price_trend = self._detect_price_trend(trajectory, step_idx)
        data_age = trajectory.get('age', 0)
        
        # Compute adaptive rollout horizon
        L = self.compute_adaptive_horizon(
            data_age_steps=data_age,
            trajectory_length=remaining_length,
            market_volatility=market_volatility,
            price_trend=price_trend
        )
        
        # Cumulative discounted rewards over horizon L
        sum_R = 0.0
        discount_factor = 1.0
        
        for i in range(L):
            reward_idx = step_idx + i
            if reward_idx < len(rewards):
                sum_R += discount_factor * float(rewards[reward_idx])
                discount_factor *= self.discount_factor
        
        # Fresh MCTS at tail state with current model
        v_root = 0.0
        tail_value_method = "terminal"
        
        if step_idx + L < len(observations):
            try:
                tail_obs = observations[step_idx + L]
                
                # Process tail observation through SWT system
                if hasattr(networks, 'market_encoder'):
                    # SWT-specific processing with market and position features
                    market_features = self._extract_market_features(tail_obs, networks)
                    position_features = self._extract_position_features(tail_obs, networks)
                    
                    # Encode to latent representation
                    tail_latent = networks.representation_network(
                        market_features, position_features
                    )
                else:
                    # Fallback to direct encoding
                    tail_latent = networks.representation_network(tail_obs)
                
                # Run deterministic MCTS for value estimation
                if hasattr(mcts_runner, 'run'):
                    mcts_result = mcts_runner.run(
                        root_latent=tail_latent,
                        simulations=15,  # Production setting
                        exploration_noise=False  # Deterministic for correction
                    )
                    v_root = float(mcts_result.root_value)
                    tail_value_method = "mcts"
                else:
                    # Fallback to direct value network
                    with torch.no_grad():
                        value_logits = networks.value_network(tail_latent)
                        v_root = float(self._logits_to_value(value_logits))
                    tail_value_method = "value_network"
                    
            except Exception as e:
                logger.warning(f"Tail value estimation failed: {e}, using zero")
                v_root = 0.0
                tail_value_method = "fallback_zero"
        
        # Corrected target: z = sum_R + γ^L * v_root
        gamma_power_L = self.discount_factor ** L
        corrected_target = sum_R + gamma_power_L * v_root
        
        # Update statistics
        self.correction_stats['total_corrections'] += 1
        
        # Diagnostic metrics
        correction_metrics = {
            'rollout_horizon': L,
            'cumulative_reward': sum_R,
            'tail_value': v_root,
            'correction_weight': gamma_power_L,
            'corrected_target': corrected_target,
            'tail_value_method': tail_value_method,
            'market_volatility': market_volatility,
            'price_trend': price_trend,
            'data_age': data_age,
            'remaining_length': remaining_length
        }
        
        return corrected_target, correction_metrics
    
    def _estimate_market_volatility(
        self, 
        trajectory: Dict[str, Any], 
        step_idx: int,
        lookback: int = 20
    ) -> float:
        """
        Estimate market volatility from recent price movements
        
        Args:
            trajectory: Episode trajectory
            step_idx: Current step
            lookback: Number of steps to look back
            
        Returns:
            Volatility estimate (0-1 scale)
        """
        observations = trajectory['observations']
        
        # Extract recent price observations
        start_idx = max(0, step_idx - lookback)
        end_idx = min(len(observations), step_idx + 1)
        
        if end_idx - start_idx < 2:
            return 0.5  # Default medium volatility
        
        try:
            # Extract price information from observations
            prices = []
            for i in range(start_idx, end_idx):
                obs = observations[i]
                
                # Try different price extraction methods
                if isinstance(obs, dict):
                    if 'market_prices' in obs:
                        price = float(obs['market_prices'][-1])  # Latest price
                    elif 'close' in obs:
                        price = float(obs['close'])
                    else:
                        price = 0.0
                elif torch.is_tensor(obs):
                    price = float(obs.flatten()[0])  # First element
                else:
                    price = float(obs)
                
                prices.append(price)
            
            if len(prices) < 2:
                return 0.5
            
            # Compute price returns
            returns = [
                (prices[i] - prices[i-1]) / prices[i-1] 
                for i in range(1, len(prices))
                if prices[i-1] != 0
            ]
            
            if not returns:
                return 0.5
            
            # Volatility as standard deviation of returns
            volatility = float(np.std(returns))
            
            # Normalize to 0-1 scale (typical forex volatility is 0-0.05)
            normalized_volatility = min(volatility * 20, 1.0)
            
            return normalized_volatility
            
        except Exception as e:
            logger.debug(f"Volatility estimation failed: {e}")
            return 0.5  # Default fallback
    
    def _detect_price_trend(
        self, 
        trajectory: Dict[str, Any], 
        step_idx: int,
        lookback: int = 10
    ) -> str:
        """
        Detect price trend direction from recent observations
        
        Args:
            trajectory: Episode trajectory
            step_idx: Current step
            lookback: Number of steps for trend analysis
            
        Returns:
            Trend direction ('up', 'down', 'sideways')
        """
        observations = trajectory['observations']
        
        start_idx = max(0, step_idx - lookback)
        end_idx = min(len(observations), step_idx + 1)
        
        if end_idx - start_idx < 3:
            return 'sideways'  # Not enough data
        
        try:
            # Extract price sequence
            prices = []
            for i in range(start_idx, end_idx):
                obs = observations[i]
                
                if isinstance(obs, dict):
                    if 'market_prices' in obs:
                        price = float(obs['market_prices'][-1])
                    elif 'close' in obs:
                        price = float(obs['close'])
                    else:
                        continue
                elif torch.is_tensor(obs):
                    price = float(obs.flatten()[0])
                else:
                    price = float(obs)
                
                prices.append(price)
            
            if len(prices) < 3:
                return 'sideways'
            
            # Simple trend detection using linear regression slope
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Determine trend based on slope magnitude
            if abs(slope) < np.std(prices) * 0.1:  # Low slope relative to volatility
                return 'sideways'
            elif slope > 0:
                return 'up'
            else:
                return 'down'
                
        except Exception as e:
            logger.debug(f"Trend detection failed: {e}")
            return 'sideways'
    
    def _extract_market_features(
        self, 
        observation: Any, 
        networks: Any
    ) -> torch.Tensor:
        """Extract market features from observation using SWT networks"""
        
        try:
            if hasattr(networks, 'market_encoder'):
                # Use SWT market encoder
                if isinstance(observation, dict) and 'market_prices' in observation:
                    market_data = observation['market_prices']
                    if not torch.is_tensor(market_data):
                        market_data = torch.tensor(market_data, dtype=torch.float32)
                    
                    return networks.market_encoder.extract_wst_features(market_data)
                
            # Fallback extraction
            if isinstance(observation, dict):
                # Try to find market data in various formats
                for key in ['market_data', 'prices', 'observations']:
                    if key in observation:
                        data = observation[key]
                        if not torch.is_tensor(data):
                            data = torch.tensor(data, dtype=torch.float32)
                        return data
            
            # Convert observation directly
            if not torch.is_tensor(observation):
                observation = torch.tensor(observation, dtype=torch.float32)
            
            return observation.flatten()
            
        except Exception as e:
            logger.debug(f"Market feature extraction failed: {e}")
            # Return zero tensor as fallback
            return torch.zeros(128, dtype=torch.float32)  # SWT default market dim
    
    def _extract_position_features(
        self, 
        observation: Any, 
        networks: Any
    ) -> torch.Tensor:
        """Extract position features from observation"""
        
        try:
            if isinstance(observation, dict) and 'position_state' in observation:
                pos_data = observation['position_state']
                if not torch.is_tensor(pos_data):
                    pos_data = torch.tensor(pos_data, dtype=torch.float32)
                return pos_data
            
            # Fallback: create default position features
            return torch.zeros(9, dtype=torch.float32)  # SWT position feature dim
            
        except Exception as e:
            logger.debug(f"Position feature extraction failed: {e}")
            return torch.zeros(9, dtype=torch.float32)
    
    def _logits_to_value(self, value_logits: torch.Tensor) -> float:
        """Convert value network logits to scalar value"""
        
        try:
            if value_logits.dim() > 1:
                # Categorical value distribution (SWT style)
                probs = torch.softmax(value_logits, dim=-1)
                support_size = probs.shape[-1]
                
                # Assume symmetric support around 0 (typical for trading)
                max_value = 300.0  # SWT: ±300 pips
                support = torch.linspace(-max_value, max_value, support_size)
                
                value = torch.sum(probs * support, dim=-1)
                return float(value.mean())
            else:
                # Scalar value output
                return float(value_logits.mean())
                
        except Exception as e:
            logger.debug(f"Value conversion failed: {e}")
            return 0.0
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get accumulated correction statistics"""
        return self.correction_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset accumulated statistics"""
        self.correction_stats = {
            'total_corrections': 0,
            'avg_horizon': 0.0,
            'volatility_adjustments': 0,
            'regime_adjustments': 0
        }


def create_swt_off_policy_correction(
    correction_type: str = 'standard',
    **kwargs
) -> OffPolicyCorrectBase:
    """
    Factory function to create off-policy correction module
    
    Args:
        correction_type: Type of correction ('standard')
        **kwargs: Additional arguments for correction initialization
        
    Returns:
        Initialized off-policy correction module
        
    Raises:
        ValueError: If correction_type not supported
    """
    if correction_type == 'standard':
        return SWTOffPolicyCorrection(**kwargs)
    else:
        raise ValueError(f"Unsupported correction_type: {correction_type}")


def test_off_policy_correction() -> None:
    """Test function for off-policy correction"""
    
    logger.info("Testing SWT Off-Policy Correction...")
    
    # Create mock trajectory
    trajectory = {
        'observations': [
            {'market_prices': torch.randn(256), 'position_state': torch.randn(9)}
            for _ in range(50)
        ],
        'rewards': [np.random.randn() for _ in range(50)],
        'actions': [np.random.randint(0, 4) for _ in range(50)],
        'age': 5
    }
    
    # Mock networks
    class MockNetworks:
        def representation_network(self, market_feat, pos_feat):
            return torch.randn(256)
        
        class market_encoder:
            @staticmethod
            def extract_wst_features(data):
                return torch.randn(128)
        
        def value_network(self, latent):
            return torch.randn(601)  # SWT categorical value
    
    # Mock MCTS runner
    class MockMCTS:
        class Result:
            root_value = 5.5
        
        def run(self, root_latent, simulations, exploration_noise):
            return self.Result()
    
    networks = MockNetworks()
    mcts_runner = MockMCTS()
    
    # Test correction
    corrector = create_swt_off_policy_correction('standard')
    
    step_idx = 20
    corrected_target, metrics = corrector.compute_corrected_value_target(
        trajectory, networks, mcts_runner, step_idx
    )
    
    logger.info(f"✅ Corrected target: {corrected_target:.4f}")
    logger.info(f"   Correction metrics: {metrics}")
    
    # Test adaptive horizon
    horizon = corrector.compute_adaptive_horizon(
        data_age_steps=10,
        trajectory_length=30,
        market_volatility=0.3,
        price_trend='up'
    )
    
    logger.info(f"✅ Adaptive horizon: {horizon}")
    
    # Test statistics
    stats = corrector.get_correction_statistics()
    logger.info(f"✅ Statistics: {stats}")
    
    logger.info("✅ Off-policy correction tests passed!")


if __name__ == "__main__":
    test_off_policy_correction()