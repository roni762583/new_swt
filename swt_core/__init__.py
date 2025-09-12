"""
SWT Core Module
Core business logic and shared components for the SWT trading system

This module provides the foundational infrastructure for both standard 
Stochastic MuZero and experimental enhanced algorithms.
"""

from .types import *
from .config_manager import ConfigManager, SWTConfig
from .exceptions import *

__version__ = "1.0.0"
__all__ = [
    # Core types
    "PositionState", "MarketState", "TradingDecision", "TradeResult",
    "AgentType", "NetworkArchitecture", "MCTSConfig",
    
    # Configuration
    "ConfigManager", "SWTConfig",
    
    # Exceptions  
    "SWTException", "ConfigurationError", "InferenceError", 
    "FeatureProcessingError", "TradingError"
]