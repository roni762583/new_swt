"""
Position Feature Extraction
CRITICAL: Exact replication of training environment position features

This module implements the EXACT position feature calculations used in
the training environment to eliminate the feature mismatch that caused
live trading failures in the original system.

The 9 position features MUST match training exactly:
1. current_equity_pips - arctan scaled by 150
2. bars_since_entry - arctan scaled by 2000
3. position_efficiency - already in [-1, 1]
4. pips_from_peak - arctan scaled by 150
5. max_drawdown_pips - arctan scaled by 150
6. amddp_reward - arctan scaled by 150
7. is_long - binary flag
8. is_short - binary flag
9. has_position - binary flag
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from swt_core.types import PositionState, PositionType
from swt_core.exceptions import FeatureProcessingError

logger = logging.getLogger(__name__)


class PositionFeatureExtractor:
    """
    Position feature extractor with EXACT training environment compatibility

    CRITICAL: This implementation matches stochastic_forex_gym.py exactly to
    prevent the dimension mismatch that rendered live trading ineffective.
    """

    def __init__(self, reward_type: str = "amddp1"):
        """
        Initialize position feature extractor

        Args:
            reward_type: Type of reward calculation ("amddp5" or "amddp10")
        """
        self.reward_type = reward_type
        logger.info(f"📊 PositionFeatureExtractor initialized with {reward_type} reward")

    @staticmethod
    def arctan_scale(value: float, scale: float = 150) -> float:
        """Scale value to [-1, 1] using arctangent (EXACT training formula)"""
        return (2 / np.pi) * np.arctan(value / scale)

    def extract_features(self, position_info: Dict[str, Any]) -> np.ndarray:
        """
        Extract 9D position features EXACTLY matching training environment

        Args:
            position_info: Position information dictionary with keys:
                - direction: Position direction (-1, 0, 1)
                - current_equity_pips: Current P&L in pips
                - bars_since_entry: Bars since position entry
                - position_efficiency: Position efficiency metric
                - pips_from_peak: Pips from equity peak
                - max_drawdown_pips: Maximum drawdown in pips
                - amddp_reward: AMDDP reward value
                - dd_sum: Cumulative drawdown sum (for AMDDP calculation)

        Returns:
            9D numpy array matching training environment format exactly
        """
        # Initialize 9D feature array (EXACT training format)
        features = np.zeros(9, dtype=np.float32)

        # Get position direction
        direction = position_info.get('direction', 0)

        # Features 0-5: Scaled position metrics (arctan normalization to [-1, 1])
        features[0] = self.arctan_scale(position_info.get('current_equity_pips', 0.0), 150)
        features[1] = self.arctan_scale(position_info.get('bars_since_entry', 0), 2000)
        features[2] = position_info.get('position_efficiency', 0.0)  # Already in [-1, 1]
        features[3] = self.arctan_scale(position_info.get('pips_from_peak', 0.0), 150)
        features[4] = self.arctan_scale(position_info.get('max_drawdown_pips', 0.0), 150)

        # Calculate AMDDP reward if not provided
        if 'amddp_reward' in position_info:
            amddp_reward = position_info['amddp_reward']
        else:
            # Calculate based on reward type
            current_equity = position_info.get('current_equity_pips', 0.0)
            dd_sum = position_info.get('dd_sum', 0.0)
            if self.reward_type == "amddp1":
                amddp_reward = current_equity - 0.01 * dd_sum
            elif self.reward_type == "amddp5":
                amddp_reward = current_equity - 0.05 * dd_sum
            elif self.reward_type == "amddp10":
                amddp_reward = current_equity - 0.1 * dd_sum
            else:
                amddp_reward = current_equity  # Pure pips

        features[5] = self.arctan_scale(amddp_reward, 150)

        # Features 6-8: Binary flags
        features[6] = float(direction > 0)   # is_long
        features[7] = float(direction < 0)   # is_short
        features[8] = float(direction != 0)  # has_position

        return features

    def create_position_info_from_state(self, position_state: PositionState,
                                       current_price: float) -> Dict[str, Any]:
        """
        Convert PositionState to position_info dict matching training format

        Args:
            position_state: Position state object
            current_price: Current market price

        Returns:
            Position info dictionary matching training environment
        """
        # Initialize position info
        position_info = {
            'direction': 0,
            'bars_since_entry': 0,
            'current_equity_pips': 0.0,
            'position_efficiency': 0.0,
            'pips_from_peak': 0.0,
            'max_drawdown_pips': 0.0,
            'dd_sum': 0.0,
            'amddp_reward': 0.0
        }

        # If flat position, return zeros
        if position_state.is_flat():
            return position_info

        # Set direction
        if position_state.position_type == PositionType.LONG:
            position_info['direction'] = 1
        elif position_state.position_type == PositionType.SHORT:
            position_info['direction'] = -1

        # Set metrics from position state
        position_info['bars_since_entry'] = position_state.bars_since_entry
        position_info['current_equity_pips'] = position_state.unrealized_pnl_pips

        # Calculate position efficiency
        if hasattr(position_state, 'peak_equity') and position_state.peak_equity > 0:
            position_info['position_efficiency'] = position_state.unrealized_pnl_pips / position_state.peak_equity
        else:
            position_info['position_efficiency'] = 1.0 if position_state.unrealized_pnl_pips > 0 else 0.0

        # Pips from peak
        if hasattr(position_state, 'peak_equity'):
            position_info['pips_from_peak'] = position_state.unrealized_pnl_pips - position_state.peak_equity
        else:
            position_info['pips_from_peak'] = min(0, position_state.unrealized_pnl_pips)

        # Max drawdown
        if hasattr(position_state, 'max_drawdown_pips'):
            position_info['max_drawdown_pips'] = abs(position_state.max_drawdown_pips)
        elif hasattr(position_state, 'max_adverse_pips'):
            position_info['max_drawdown_pips'] = abs(position_state.max_adverse_pips)

        # Cumulative drawdown sum
        if hasattr(position_state, 'dd_sum'):
            position_info['dd_sum'] = position_state.dd_sum
        elif hasattr(position_state, 'accumulated_drawdown'):
            position_info['dd_sum'] = position_state.accumulated_drawdown

        # Calculate AMDDP reward
        if self.reward_type == "amddp1":
            position_info['amddp_reward'] = position_info['current_equity_pips'] - 0.01 * position_info['dd_sum']
        elif self.reward_type == "amddp5":
            position_info['amddp_reward'] = position_info['current_equity_pips'] - 0.05 * position_info['dd_sum']
        elif self.reward_type == "amddp10":
            position_info['amddp_reward'] = position_info['current_equity_pips'] - 0.1 * position_info['dd_sum']
        else:
            position_info['amddp_reward'] = position_info['current_equity_pips']

        return position_info

    def get_feature_names(self) -> List[str]:
        """Get feature names for logging and debugging"""
        return [
            "current_equity_pips",     # 0: arctan scaled by 150
            "bars_since_entry",        # 1: arctan scaled by 2000
            "position_efficiency",     # 2: already in [-1, 1]
            "pips_from_peak",         # 3: arctan scaled by 150
            "max_drawdown_pips",      # 4: arctan scaled by 150
            "amddp_reward",           # 5: arctan scaled by 150
            "is_long",                # 6: binary flag
            "is_short",               # 7: binary flag
            "has_position"            # 8: binary flag
        ]

    def get_feature_description(self, feature_index: int) -> str:
        """Get detailed description of specific feature"""
        descriptions = {
            0: "Current equity in pips (arctan scaled by 150)",
            1: "Bars since position entry (arctan scaled by 2000)",
            2: "Position efficiency metric [-1, 1]",
            3: "Pips from equity peak (arctan scaled by 150)",
            4: "Maximum drawdown in pips (arctan scaled by 150)",
            5: f"AMDDP reward ({self.reward_type}) (arctan scaled by 150)",
            6: "Is long position (binary)",
            7: "Is short position (binary)",
            8: "Has position (binary)"
        }

        return descriptions.get(feature_index, f"Feature {feature_index}")

    def get_diagnostics(self, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get diagnostic information for debugging"""
        features = self.extract_features(position_info)

        return {
            "features": features.tolist(),
            "feature_names": self.get_feature_names(),
            "position_info": position_info,
            "reward_type": self.reward_type,
            "feature_descriptions": [self.get_feature_description(i) for i in range(9)]
        }


class PositionTracker:
    """
    Position tracker that maintains position state matching training environment

    This class tracks position metrics exactly as in stochastic_forex_gym.py
    """

    def __init__(self, pip_value: float = 0.01):
        """Initialize position tracker

        Args:
            pip_value: Value of 1 pip (0.01 for JPY pairs)
        """
        self.pip_value = pip_value
        self.reset()

    def reset(self):
        """Reset position to flat"""
        self.direction = 0  # -1: short, 0: flat, 1: long
        self.entry_price = None
        self.entry_time = None
        self.bars_since_entry = 0

        # For tracking equity and efficiency
        self.peak_equity = 0
        self.min_equity = 0

        # For tracking drawdowns
        self.max_dd = 0  # Maximum drawdown seen
        self.dd_sum = 0  # Cumulative sum of drawdown increases

    def update(self, current_price: float) -> Dict[str, Any]:
        """Update position metrics and return position info"""
        position_info = {
            'direction': self.direction,
            'bars_since_entry': self.bars_since_entry,
            'current_equity_pips': 0,
            'position_efficiency': 1.0,
            'pips_from_peak': 0,
            'max_drawdown_pips': 0,
            'amddp_reward': 0,
            'dd_sum': self.dd_sum
        }

        if self.direction != 0:
            # Calculate P&L in pips
            if self.direction > 0:  # Long
                pnl_pips = (current_price - self.entry_price) / self.pip_value
            else:  # Short
                pnl_pips = (self.entry_price - current_price) / self.pip_value

            position_info['current_equity_pips'] = pnl_pips

            # Update high water mark
            if pnl_pips > self.peak_equity:
                self.peak_equity = pnl_pips

            # Update low water mark
            if pnl_pips < self.min_equity:
                self.min_equity = pnl_pips

            # Calculate drawdown from peak
            current_dd = self.peak_equity - pnl_pips

            # Track maximum drawdown
            if current_dd > self.max_dd:
                dd_increase = current_dd - self.max_dd
                self.dd_sum += dd_increase
                self.max_dd = current_dd

            # Reverse drawdown (position recovering)
            current_max_dd = self.peak_equity - self.min_equity

            # Position metrics
            position_info['pips_from_peak'] = pnl_pips - self.peak_equity
            position_info['max_drawdown_pips'] = -current_max_dd

            # Position efficiency
            if self.peak_equity > 0:
                position_info['position_efficiency'] = pnl_pips / self.peak_equity

            # AMDDP reward calculation
            amddp1_reward = pnl_pips - 0.01 * self.dd_sum  # AMDDP1
            position_info['amddp_reward'] = amddp1_reward

            # Update bars since entry
            self.bars_since_entry += 1
            position_info['bars_since_entry'] = self.bars_since_entry

        return position_info

    def open_position(self, direction: int, entry_price: float, timestamp: datetime):
        """Open a new position"""
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = timestamp
        self.bars_since_entry = 0
        self.peak_equity = 0
        self.min_equity = 0
        self.max_dd = 0
        self.dd_sum = 0

    def close_position(self) -> Dict[str, Any]:
        """Close position and return final metrics"""
        final_info = {
            'direction': self.direction,
            'bars_since_entry': self.bars_since_entry,
            'dd_sum': self.dd_sum,
            'peak_equity': self.peak_equity,
            'min_equity': self.min_equity
        }
        self.reset()
        return final_info