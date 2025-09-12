"""
Position Feature Extraction
CRITICAL: Exact replication of training environment position features

This module implements the EXACT position feature calculations used in 
the training environment to eliminate the feature mismatch that caused
live trading failures in the original system.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import logging

from swt_core.types import PositionState, PositionType
from swt_core.exceptions import FeatureProcessingError

logger = logging.getLogger(__name__)


class PositionFeatureExtractor:
    """
    Position feature extractor with EXACT training environment compatibility
    
    CRITICAL: This implementation must match swt_forex_env.py exactly to 
    prevent the dimension mismatch that rendered live trading ineffective.
    """
    
    def __init__(self, normalization_config: Dict[str, float]):
        """
        Initialize position feature extractor
        
        Args:
            normalization_config: Normalization parameters matching training
        """
        self.norm_config = normalization_config
        
        # Validation: ensure all required normalization parameters are present
        required_params = {
            "duration_scale", "pnl_scale", "price_change_scale", 
            "drawdown_scale", "accumulated_dd_scale", "bars_since_dd_scale"
        }
        
        missing_params = required_params - set(normalization_config.keys())
        if missing_params:
            raise FeatureProcessingError(
                f"Missing normalization parameters: {missing_params}",
                context={"provided_params": list(normalization_config.keys())}
            )
        
        # Price tracking for recent price change calculation
        self._price_history: List[float] = []
        self._max_history_length = 10
        
        logger.info("📊 PositionFeatureExtractor initialized with training-compatible parameters")
    
    def extract_features(self, position_state: PositionState,
                        current_price: float,
                        market_metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract 9D position features EXACTLY matching training environment
        
        Args:
            position_state: Current position state
            current_price: Current market price
            market_metadata: Additional market information
            
        Returns:
            9D numpy array matching training environment format
        """
        try:
            # Initialize 9D feature array (EXACT training format)
            features = np.zeros(9, dtype=np.float32)
            
            # Update price history for momentum calculation
            self._update_price_history(current_price)
            
            if position_state.is_flat():
                # FLAT POSITION: All features are zero (training behavior)
                return features
            
            # NON-FLAT POSITION: Calculate all 9 features exactly as in training
            
            # Feature 0: Position side (-1/0/+1) - EXACT training encoding
            if position_state.position_type == PositionType.LONG:
                features[0] = 1.0
            elif position_state.position_type == PositionType.SHORT:
                features[0] = -1.0
            else:
                features[0] = 0.0
            
            # Feature 1: Duration in bars (normalized by 720.0) - EXACT training
            features[1] = min(position_state.bars_since_entry / self.norm_config["duration_scale"], 1.0)
            
            # Feature 2: Unrealized PnL (normalized by 100.0) - EXACT training  
            features[2] = position_state.unrealized_pnl_pips / self.norm_config["pnl_scale"]
            
            # Feature 3: Entry price relative to current - EXACT training calculation
            if position_state.entry_price is not None and position_state.entry_price > 0:
                features[3] = (position_state.entry_price - current_price) / current_price
            else:
                features[3] = 0.0
            
            # Feature 4: Recent price change (normalized by 50.0) - EXACT training
            recent_change = self._calculate_recent_price_change()
            features[4] = recent_change / self.norm_config["price_change_scale"]
            
            # Feature 5: Max drawdown pips (normalized by 50.0) - EXACT training
            features[5] = position_state.max_adverse_pips / self.norm_config["drawdown_scale"]
            
            # Feature 6: Accumulated drawdown (normalized by 100.0) - EXACT training
            features[6] = position_state.accumulated_drawdown / self.norm_config["accumulated_dd_scale"]
            
            # Feature 7: Bars since max drawdown (normalized by 60.0) - EXACT training
            features[7] = min(position_state.bars_since_max_drawdown / self.norm_config["bars_since_dd_scale"], 1.0)
            
            # Feature 8: Risk flags (0.0 to 1.0) - EXACT training risk calculation
            features[8] = self._calculate_risk_flags(position_state, current_price, market_metadata)
            
            # Final validation
            self._validate_features(features)
            
            return features
            
        except Exception as e:
            raise FeatureProcessingError(
                f"Position feature extraction failed: {str(e)}",
                context={
                    "position_type": position_state.position_type.name,
                    "current_price": current_price,
                    "bars_since_entry": position_state.bars_since_entry
                },
                original_error=e
            )
    
    def _update_price_history(self, current_price: float) -> None:
        """Update price history for momentum calculations"""
        self._price_history.append(current_price)
        
        # Maintain maximum history length
        if len(self._price_history) > self._max_history_length:
            self._price_history.pop(0)
    
    def _calculate_recent_price_change(self) -> float:
        """
        Calculate recent price change EXACTLY as in training environment
        
        Returns:
            Recent price change in pips (positive = price increased)
        """
        if len(self._price_history) < 2:
            return 0.0
        
        # Calculate change over last few bars (training behavior)
        lookback = min(5, len(self._price_history))
        recent_prices = self._price_history[-lookback:]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Simple price momentum: current - average of recent
        current_price = recent_prices[-1]
        avg_recent = sum(recent_prices[:-1]) / len(recent_prices[:-1])
        
        # Convert to pips (for GBP/JPY: multiply by 100)
        price_change_pips = (current_price - avg_recent) * 100.0
        
        return price_change_pips
    
    def _calculate_risk_flags(self, position_state: PositionState, 
                            current_price: float,
                            market_metadata: Optional[Dict[str, Any]]) -> float:
        """
        Calculate risk flags EXACTLY as in training environment
        
        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        risk_components = []
        
        # Component 1: Duration risk (longer positions = higher risk)
        duration_risk = min(position_state.bars_since_entry / 720.0, 1.0)
        risk_components.append(duration_risk * 0.3)  # 30% weight
        
        # Component 2: Drawdown risk 
        if position_state.max_adverse_pips > 0:
            drawdown_risk = min(position_state.max_adverse_pips / 100.0, 1.0)  # 100 pip normalization
            risk_components.append(drawdown_risk * 0.4)  # 40% weight
        else:
            risk_components.append(0.0)
        
        # Component 3: PnL volatility risk
        if abs(position_state.unrealized_pnl_pips) > 50:  # High PnL volatility
            pnl_risk = min(abs(position_state.unrealized_pnl_pips) / 200.0, 1.0)
            risk_components.append(pnl_risk * 0.2)  # 20% weight  
        else:
            risk_components.append(0.0)
        
        # Component 4: Market condition risk (if metadata available)
        market_risk = 0.0
        if market_metadata:
            spread = market_metadata.get('spread', 0.0)
            if spread > 5.0:  # High spread = higher risk
                market_risk = min(spread / 20.0, 1.0)  # Normalize by 20 pips
        
        risk_components.append(market_risk * 0.1)  # 10% weight
        
        # Total risk score
        total_risk = sum(risk_components)
        
        # Ensure risk is in [0, 1] range
        return np.clip(total_risk, 0.0, 1.0)
    
    def _validate_features(self, features: np.ndarray) -> None:
        """Validate extracted features match training expectations"""
        # Check dimensions
        if features.shape != (9,):
            raise FeatureProcessingError(
                f"Position features must be 9D, got shape {features.shape}"
            )
        
        # Check data type
        if features.dtype != np.float32:
            raise FeatureProcessingError(
                f"Position features must be float32, got {features.dtype}"
            )
        
        # Check for invalid values
        if np.isnan(features).any():
            nan_indices = np.where(np.isnan(features))[0]
            raise FeatureProcessingError(
                f"Position features contain NaN at indices: {nan_indices.tolist()}"
            )
        
        if np.isinf(features).any():
            inf_indices = np.where(np.isinf(features))[0]
            raise FeatureProcessingError(
                f"Position features contain Inf at indices: {inf_indices.tolist()}"
            )
        
        # Check feature ranges (based on training environment)
        feature_ranges = {
            0: (-1.0, 1.0),      # position_side
            1: (0.0, 5.0),       # duration_bars (up to 5x session length)
            2: (-50.0, 50.0),    # unrealized_pnl_pips
            3: (-0.1, 0.1),      # entry_price_relative (10% max)
            4: (-20.0, 20.0),    # recent_price_change
            5: (0.0, 10.0),      # max_drawdown_pips
            6: (0.0, 10.0),      # accumulated_drawdown  
            7: (0.0, 5.0),       # bars_since_max_drawdown
            8: (0.0, 1.0)        # risk_flags
        }
        
        for i, (min_val, max_val) in feature_ranges.items():
            if not (min_val <= features[i] <= max_val):
                logger.warning(
                    f"⚠️ Feature {i} out of expected range [{min_val}, {max_val}]: "
                    f"got {features[i]:.4f}"
                )
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for logging and debugging"""
        return [
            "position_side",           # 0: -1/0/+1 encoding
            "duration_bars",           # 1: normalized duration  
            "unrealized_pnl_pips",     # 2: normalized PnL
            "entry_price_relative",    # 3: price relative to entry
            "recent_price_change",     # 4: price momentum
            "max_drawdown_pips",       # 5: maximum adverse excursion
            "accumulated_drawdown",    # 6: total drawdown  
            "bars_since_max_drawdown", # 7: time since max drawdown
            "risk_flags"               # 8: composite risk score
        ]
    
    def get_feature_description(self, feature_index: int) -> str:
        """Get detailed description of specific feature"""
        descriptions = {
            0: "Position direction: SHORT=-1, FLAT=0, LONG=+1",
            1: f"Bars since entry / {self.norm_config['duration_scale']:.0f}",
            2: f"Unrealized PnL pips / {self.norm_config['pnl_scale']:.0f}",
            3: "(entry_price - current_price) / current_price",
            4: f"Recent price momentum / {self.norm_config['price_change_scale']:.0f}",
            5: f"Maximum adverse excursion / {self.norm_config['drawdown_scale']:.0f}",
            6: f"Total accumulated drawdown / {self.norm_config['accumulated_dd_scale']:.0f}",
            7: f"Bars since max drawdown / {self.norm_config['bars_since_dd_scale']:.0f}",
            8: "Composite risk score [0=low risk, 1=high risk]"
        }
        
        return descriptions.get(feature_index, f"Feature {feature_index}")
    
    def reset_price_history(self) -> None:
        """Reset price history (useful for new trading sessions)"""
        self._price_history.clear()
        logger.info("🔄 Price history reset")
    
    def get_diagnostics(self, position_state: PositionState, 
                       current_price: float) -> Dict[str, Any]:
        """Get diagnostic information for debugging"""
        features = self.extract_features(position_state, current_price)
        
        return {
            "features": features.tolist(),
            "feature_names": self.get_feature_names(),
            "position_state": {
                "type": position_state.position_type.name,
                "is_flat": position_state.is_flat(),
                "bars_since_entry": position_state.bars_since_entry,
                "unrealized_pnl": position_state.unrealized_pnl_pips,
                "max_adverse": position_state.max_adverse_pips
            },
            "normalization_config": self.norm_config,
            "price_history_length": len(self._price_history),
            "validation_passed": True  # If we got here, validation passed
        }