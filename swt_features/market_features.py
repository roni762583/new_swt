"""
Market Feature Extraction  
Processes raw market data into WST-enhanced features for the trading system

Handles price series processing, normalization, and gap detection with
consistent output for both training and live trading systems.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from collections import deque
from dataclasses import dataclass

from swt_core.types import MarketState
from swt_core.exceptions import FeatureProcessingError
from .wst_transform import WSTProcessor, WSTConfig

logger = logging.getLogger(__name__)


@dataclass 
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: Optional[float] = None


class MarketFeatureExtractor:
    """
    Market feature extractor with WST processing
    
    Processes raw OHLC data into consistent market features using
    Wavelet Scattering Transform for both training and live systems.
    """
    
    def __init__(self, wst_config: WSTConfig, 
                 window_size: int = 256,
                 normalization_method: str = "zscore"):
        """
        Initialize market feature extractor
        
        Args:
            wst_config: WST processor configuration
            window_size: Size of price window for WST processing
            normalization_method: Price normalization method
        """
        self.window_size = window_size
        self.normalization_method = normalization_method
        
        # Initialize WST processor
        self.wst_processor = WSTProcessor(wst_config)
        
        # Price history buffer (efficient circular buffer)
        self._price_buffer = deque(maxlen=window_size)
        self._volume_buffer = deque(maxlen=window_size)
        self._timestamp_buffer = deque(maxlen=window_size)
        
        # Normalization statistics (for zscore)
        self._price_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0
        }
        
        # Gap detection
        self._last_close = None
        self._gap_threshold_pips = 5.0
        
        logger.info(f"📈 MarketFeatureExtractor initialized: "
                   f"window_size={window_size}, normalization={normalization_method}")
    
    def add_market_data(self, data_point: MarketDataPoint) -> None:
        """
        Add new market data point to the buffer
        
        Args:
            data_point: New market data to add
        """
        try:
            # Validate market data
            self._validate_market_data(data_point)
            
            # Detect gaps
            if self._last_close is not None:
                gap_pips = abs(data_point.open - self._last_close) * 100  # Convert to pips
                if gap_pips > self._gap_threshold_pips:
                    logger.warning(f"⚠️ Market gap detected: {gap_pips:.1f} pips")
            
            # Add to buffers
            self._price_buffer.append(data_point.close)
            self._volume_buffer.append(data_point.volume)
            self._timestamp_buffer.append(data_point.timestamp)
            
            # Update last close
            self._last_close = data_point.close
            
            # Update normalization statistics
            self._update_normalization_stats()
            
        except Exception as e:
            raise FeatureProcessingError(
                f"Failed to add market data: {str(e)}",
                context={
                    "timestamp": data_point.timestamp,
                    "close_price": data_point.close
                },
                original_error=e
            )
    
    def extract_features(self, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Extract WST market features from current price buffer
        
        Args:
            cache_key: Optional cache key for WST transform
            
        Returns:
            128D numpy array of WST market features
        """
        try:
            # Check if we have enough data
            if len(self._price_buffer) < self.window_size:
                raise FeatureProcessingError(
                    f"Insufficient market data: need {self.window_size}, have {len(self._price_buffer)}"
                )
            
            # Get price series
            price_series = np.array(self._price_buffer, dtype=np.float32)
            
            # Normalize price series
            normalized_prices = self._normalize_prices(price_series)
            
            # Apply WST transform
            wst_features = self.wst_processor.transform(normalized_prices, cache_key=cache_key)
            
            # Validate output
            self._validate_features(wst_features)
            
            return wst_features
            
        except Exception as e:
            raise FeatureProcessingError(
                f"Market feature extraction failed: {str(e)}",
                context={
                    "buffer_size": len(self._price_buffer),
                    "normalization_method": self.normalization_method
                },
                original_error=e
            )
    
    def get_market_state(self, cache_key: Optional[str] = None) -> MarketState:
        """
        Get complete market state with WST features
        
        Args:
            cache_key: Optional cache key for WST transform
            
        Returns:
            MarketState object with WST features and metadata
        """
        if len(self._price_buffer) < 4:  # Need at least OHLC
            raise FeatureProcessingError("Insufficient data for market state construction")
        
        # Get WST features
        wst_features = self.extract_features(cache_key=cache_key)
        
        # Get latest OHLC (approximate from close prices)
        recent_prices = list(self._price_buffer)[-4:]
        ohlc = (
            recent_prices[0],   # Open (4 bars ago)
            max(recent_prices), # High
            min(recent_prices), # Low  
            recent_prices[-1]   # Close (current)
        )
        
        # Get latest volume and timestamp
        volume = self._volume_buffer[-1] if self._volume_buffer else 0.0
        timestamp = self._timestamp_buffer[-1] if self._timestamp_buffer else datetime.now()
        
        # Calculate spread (if available) or estimate
        spread = self._estimate_spread(recent_prices)
        
        # Calculate volatility
        volatility = self._calculate_volatility(recent_prices) if len(recent_prices) >= 20 else None
        
        return MarketState(
            wst_features=wst_features,
            ohlc=ohlc,
            volume=volume,
            spread=spread,
            timestamp=timestamp,
            volatility=volatility
        )
    
    def _validate_market_data(self, data_point: MarketDataPoint) -> None:
        """Validate incoming market data"""
        # Check for None/invalid values
        if data_point.close is None or data_point.close <= 0:
            raise FeatureProcessingError(f"Invalid close price: {data_point.close}")
        
        if data_point.volume is None or data_point.volume < 0:
            raise FeatureProcessingError(f"Invalid volume: {data_point.volume}")
        
        # Check OHLC consistency
        prices = [data_point.open, data_point.high, data_point.low, data_point.close]
        if any(p is None or p <= 0 for p in prices):
            raise FeatureProcessingError("Invalid OHLC prices")
        
        if data_point.high < max(data_point.open, data_point.close):
            raise FeatureProcessingError("High price less than open/close")
        
        if data_point.low > min(data_point.open, data_point.close):
            raise FeatureProcessingError("Low price greater than open/close")
        
        # Check for extreme price movements (circuit breaker)
        if self._last_close is not None:
            price_change = abs(data_point.close - self._last_close) / self._last_close
            if price_change > 0.1:  # 10% price change
                logger.warning(f"⚠️ Extreme price movement: {price_change:.2%}")
    
    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize price series using configured method"""
        if self.normalization_method == "zscore":
            mean = self._price_stats["mean"]
            std = self._price_stats["std"]
            if std > 0:
                return (prices - mean) / std
            else:
                return prices - mean
                
        elif self.normalization_method == "minmax":
            min_price = self._price_stats["min"]
            max_price = self._price_stats["max"]
            price_range = max_price - min_price
            if price_range > 0:
                return (prices - min_price) / price_range
            else:
                return np.zeros_like(prices)
                
        elif self.normalization_method == "robust":
            # Use median and IQR for robust normalization
            median = np.median(prices)
            q75, q25 = np.percentile(prices, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                return (prices - median) / iqr
            else:
                return prices - median
                
        else:
            # No normalization
            return prices
    
    def _update_normalization_stats(self) -> None:
        """Update running normalization statistics"""
        if len(self._price_buffer) == 0:
            return
        
        prices = np.array(self._price_buffer)
        
        # Update statistics
        self._price_stats["mean"] = np.mean(prices)
        self._price_stats["std"] = np.std(prices) + 1e-8  # Add epsilon for stability
        self._price_stats["min"] = np.min(prices)
        self._price_stats["max"] = np.max(prices)
    
    def _estimate_spread(self, recent_prices: List[float]) -> float:
        """Estimate bid-ask spread from price volatility"""
        if len(recent_prices) < 2:
            return 2.5  # Default spread for GBP/JPY
        
        # Estimate spread from recent price volatility
        price_changes = np.diff(recent_prices)
        volatility = np.std(price_changes) if len(price_changes) > 1 else 0.001
        
        # Convert to pips and apply reasonable bounds
        estimated_spread = volatility * 100 * 2  # Convert to pips, factor for bid-ask
        return np.clip(estimated_spread, 1.5, 10.0)  # Reasonable bounds for GBP/JPY
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility over recent period"""
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Return standard deviation of returns (annualized approximation)
        return np.std(returns) * np.sqrt(252 * 24 * 60)  # Approximate annualization for M1 data
    
    def _validate_features(self, features: np.ndarray) -> None:
        """Validate extracted market features"""
        expected_shape = (128,)  # WST output dimension
        
        if features.shape != expected_shape:
            raise FeatureProcessingError(
                f"Market features shape mismatch: expected {expected_shape}, got {features.shape}"
            )
        
        if features.dtype != np.float32:
            raise FeatureProcessingError(
                f"Market features must be float32, got {features.dtype}"
            )
        
        if np.isnan(features).any():
            nan_count = np.isnan(features).sum()
            raise FeatureProcessingError(f"Market features contain {nan_count} NaN values")
        
        if np.isinf(features).any():
            inf_count = np.isinf(features).sum()
            raise FeatureProcessingError(f"Market features contain {inf_count} infinite values")
    
    def is_ready(self) -> bool:
        """Check if extractor has sufficient data for feature extraction"""
        return len(self._price_buffer) >= self.window_size
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        return {
            "buffer_size": len(self._price_buffer),
            "required_size": self.window_size,
            "is_ready": self.is_ready(),
            "buffer_utilization": len(self._price_buffer) / self.window_size,
            "latest_timestamp": self._timestamp_buffer[-1] if self._timestamp_buffer else None,
            "latest_price": self._price_buffer[-1] if self._price_buffer else None
        }
    
    def clear_buffers(self) -> None:
        """Clear all data buffers (useful for new trading sessions)"""
        self._price_buffer.clear()
        self._volume_buffer.clear()
        self._timestamp_buffer.clear()
        self._last_close = None
        
        # Reset normalization stats
        self._price_stats = {
            "mean": 0.0,
            "std": 1.0,
            "min": 0.0,
            "max": 1.0
        }
        
        logger.info("🗑️ Market data buffers cleared")
    
    def get_wst_cache_stats(self) -> Dict[str, Any]:
        """Get WST cache statistics"""
        return self.wst_processor.get_cache_stats()
    
    def set_gap_threshold(self, threshold_pips: float) -> None:
        """Set gap detection threshold in pips"""
        self._gap_threshold_pips = threshold_pips
        logger.info(f"📏 Gap detection threshold set to {threshold_pips} pips")
    
    def get_recent_prices(self, num_bars: int = 10) -> List[float]:
        """Get recent prices for analysis"""
        if num_bars <= 0 or len(self._price_buffer) == 0:
            return []
        
        start_idx = max(0, len(self._price_buffer) - num_bars)
        return list(self._price_buffer)[start_idx:]
    
    def get_price_statistics(self) -> Dict[str, float]:
        """Get current price statistics"""
        if len(self._price_buffer) == 0:
            return {"count": 0}
        
        prices = np.array(self._price_buffer)
        return {
            "count": len(prices),
            "mean": float(np.mean(prices)),
            "std": float(np.std(prices)),
            "min": float(np.min(prices)),
            "max": float(np.max(prices)),
            "latest": float(prices[-1]),
            "change_from_first": float(prices[-1] - prices[0]) if len(prices) > 1 else 0.0
        }