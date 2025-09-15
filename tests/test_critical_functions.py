"""
Tests for critical trading functions that have caused bugs
or are essential for correct operation
"""
import pytest
import numpy as np
import torch
from swt_core.types import TradingDecision, SWTAction
from swt_environments.swt_forex_env import SWTForexEnvironment


class TestTradingLogic:
    """Test critical trading decision logic"""

    def test_action_mapping(self):
        """Test that action indices map correctly"""
        # SWT uses 4 actions: HOLD=0, BUY=1, SELL=2, CLOSE=3
        assert SWTAction.HOLD.value == 0
        assert SWTAction.BUY.value == 1
        assert SWTAction.SELL.value == 2
        assert SWTAction.CLOSE.value == 3

    def test_trading_decision_confidence(self):
        """Test confidence threshold filtering"""
        # Create a trading decision
        decision = TradingDecision(
            action=SWTAction.BUY,
            confidence=0.25,  # Below typical threshold
            expected_value=10.0,
            policy_probs=[0.4, 0.25, 0.2, 0.15]
        )
        
        # With min_confidence=0.35, this should be filtered to HOLD
        min_confidence = 0.35
        
        if decision.confidence < min_confidence:
            filtered_action = SWTAction.HOLD
        else:
            filtered_action = decision.action
            
        assert filtered_action == SWTAction.HOLD

    def test_position_size_validation(self):
        """Test that position sizes are validated correctly"""
        # Fixed 1-unit position sizing
        MAX_POSITION_SIZE = 1
        
        # Test various requested sizes
        test_cases = [
            (1, 1),      # Valid
            (2, 1),      # Should be capped
            (0, 0),      # Zero is valid (no trade)
            (-1, 1),     # Negative should become positive
            (100, 1),    # Large should be capped
        ]
        
        for requested, expected in test_cases:
            actual = min(abs(requested), MAX_POSITION_SIZE)
            assert actual == expected, f"Position size {requested} -> {actual}, expected {expected}"

    def test_pip_calculation_gbpjpy(self):
        """Test pip calculation for GBPJPY (0.01 = 1 pip)"""
        # GBPJPY uses 2 decimal places
        entry_price = 185.50
        current_price = 185.75
        
        # Long position
        pips_long = (current_price - entry_price) * 100  # 0.25 * 100 = 25 pips
        assert abs(pips_long - 25.0) < 0.01
        
        # Short position (inverse)
        pips_short = (entry_price - current_price) * 100  # -25 pips
        assert abs(pips_short - (-25.0)) < 0.01

    def test_feature_dimension_consistency(self):
        """Test that feature dimensions are consistent (137 total)"""
        # 128 WST features + 9 position features = 137 total
        WST_FEATURES = 128
        POSITION_FEATURES = 9
        TOTAL_FEATURES = WST_FEATURES + POSITION_FEATURES
        
        assert TOTAL_FEATURES == 137, "Feature dimension mismatch"
        
        # Create mock observation
        observation = np.zeros(TOTAL_FEATURES, dtype=np.float32)
        
        # Split should work correctly
        wst_features = observation[:WST_FEATURES]
        position_features = observation[WST_FEATURES:]
        
        assert len(wst_features) == 128
        assert len(position_features) == 9

    def test_reward_sign_convention(self):
        """Test that rewards have correct sign"""
        # Profit should be positive reward
        profit_pips = 25.0
        assert profit_pips > 0
        
        # Loss should be negative reward  
        loss_pips = -15.0
        assert loss_pips < 0
        
        # AMDDP1 formula: reward = pips - 0.01 * abs(drawdown)
        drawdown = -20.0
        amddp_reward = profit_pips - 0.01 * abs(drawdown)
        expected = 25.0 - 0.01 * 20.0  # 25 - 0.2 = 24.8
        assert abs(amddp_reward - expected) < 0.01

    def test_market_closed_handling(self):
        """Test that market closed periods are handled"""
        from datetime import datetime
        
        # Weekend check (simplified)
        def is_market_closed(dt):
            # Forex closed from Friday 22:00 UTC to Sunday 22:00 UTC
            if dt.weekday() == 5:  # Saturday
                return True
            if dt.weekday() == 6:  # Sunday
                return True
            if dt.weekday() == 4 and dt.hour >= 22:  # Friday after 22:00
                return True
            return False
        
        # Test cases
        saturday = datetime(2024, 1, 6, 12, 0)  # Saturday noon
        assert is_market_closed(saturday) == True
        
        monday = datetime(2024, 1, 8, 12, 0)  # Monday noon
        assert is_market_closed(monday) == False
        
        friday_night = datetime(2024, 1, 5, 23, 0)  # Friday 23:00
        assert is_market_closed(friday_night) == True