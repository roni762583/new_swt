"""
Critical tests for position feature calculations
MUST match training environment exactly!
"""
import pytest
import numpy as np
import torch
from swt_features.position_features import calculate_position_features


class TestPositionFeatures:
    """Test position feature calculations match training exactly"""

    def test_no_position_features(self):
        """Test features when no position is held"""
        features = calculate_position_features(
            has_position=False,
            is_long=False,
            is_short=False,
            entry_price=0.0,
            current_price=185.0,
            bars_since_entry=0,
            current_equity_pips=0.0,
            max_drawdown_pips=0.0,
            pips_from_peak=0.0,
            position_efficiency=0.0
        )

        # Should return 9 features
        assert len(features) == 9
        
        # When no position, most features should be 0
        assert features[0] == 0.0  # current_equity_pips (scaled)
        assert features[1] == 0.0  # bars_since_entry (scaled)
        assert features[2] == 0.0  # position_efficiency
        assert features[3] == 0.0  # pips_from_peak (scaled)
        assert features[4] == 0.0  # max_drawdown_pips (scaled)
        assert features[5] == 0.0  # amddp_reward (scaled)
        assert features[6] == 0.0  # is_long
        assert features[7] == 0.0  # is_short
        assert features[8] == 0.0  # has_position

    def test_long_position_features(self):
        """Test features for a profitable long position"""
        features = calculate_position_features(
            has_position=True,
            is_long=True,
            is_short=False,
            entry_price=185.0,
            current_price=185.25,  # 25 pips profit
            bars_since_entry=100,
            current_equity_pips=25.0,
            max_drawdown_pips=-10.0,
            pips_from_peak=-5.0,
            position_efficiency=0.6
        )

        assert len(features) == 9
        
        # Check arctan scaling formula: (2/π) * arctan(value/scale)
        # For 25 pips with scale 150: (2/π) * arctan(25/150) ≈ 0.105
        expected_equity = (2/np.pi) * np.arctan(25.0/150.0)
        assert abs(features[0] - expected_equity) < 0.01
        
        # Bars since entry: scale 2000
        expected_bars = (2/np.pi) * np.arctan(100/2000)
        assert abs(features[1] - expected_bars) < 0.01
        
        # Position efficiency already in [-1, 1]
        assert features[2] == 0.6
        
        # Binary flags
        assert features[6] == 1.0  # is_long
        assert features[7] == 0.0  # is_short
        assert features[8] == 1.0  # has_position

    def test_short_position_features(self):
        """Test features for a losing short position"""
        features = calculate_position_features(
            has_position=True,
            is_long=False,
            is_short=True,
            entry_price=185.0,
            current_price=185.15,  # 15 pips loss for short
            bars_since_entry=50,
            current_equity_pips=-15.0,
            max_drawdown_pips=-20.0,
            pips_from_peak=-15.0,
            position_efficiency=-0.3
        )

        assert len(features) == 9
        
        # Check negative equity scaling
        expected_equity = (2/np.pi) * np.arctan(-15.0/150.0)
        assert abs(features[0] - expected_equity) < 0.01
        
        # Binary flags
        assert features[6] == 0.0  # is_long
        assert features[7] == 1.0  # is_short
        assert features[8] == 1.0  # has_position

    def test_feature_bounds(self):
        """Test that all features stay within expected bounds"""
        # Test extreme values
        features = calculate_position_features(
            has_position=True,
            is_long=True,
            is_short=False,
            entry_price=185.0,
            current_price=190.0,  # 500 pips profit (extreme)
            bars_since_entry=10000,  # Very long position
            current_equity_pips=500.0,
            max_drawdown_pips=-200.0,
            pips_from_peak=-100.0,
            position_efficiency=1.0
        )

        # All scaled features should be in [-1, 1] due to arctan
        for i in range(6):  # First 6 are scaled features
            assert -1.0 <= features[i] <= 1.0, f"Feature {i} out of bounds: {features[i]}"
        
        # Binary features should be 0 or 1
        for i in range(6, 9):
            assert features[i] in [0.0, 1.0], f"Binary feature {i} not 0 or 1: {features[i]}"

    def test_amddp_reward_calculation(self):
        """Test AMDDP reward with 1% drawdown penalty"""
        features = calculate_position_features(
            has_position=True,
            is_long=True,
            is_short=False,
            entry_price=185.0,
            current_price=185.30,
            bars_since_entry=60,
            current_equity_pips=30.0,
            max_drawdown_pips=-15.0,  # 15 pips drawdown
            pips_from_peak=-5.0,
            position_efficiency=0.5
        )

        # AMDDP1 formula: equity_pips - 0.01 * abs(max_drawdown_pips)
        expected_amddp = 30.0 - 0.01 * 15.0  # 30 - 0.15 = 29.85
        expected_scaled = (2/np.pi) * np.arctan(expected_amddp/150.0)
        
        assert abs(features[5] - expected_scaled) < 0.01, \
            f"AMDDP reward mismatch: got {features[5]}, expected {expected_scaled}"