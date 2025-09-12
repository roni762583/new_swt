"""
SWT Features Module
Shared feature processing for both standard and experimental agents

This module provides the critical "single source of truth" for feature 
processing, eliminating the training/live system feature mismatch that
plagued the original implementation.
"""

from .feature_processor import FeatureProcessor
from .position_features import PositionFeatureExtractor  
from .market_features import MarketFeatureExtractor
from .wst_transform import WSTProcessor

__version__ = "1.0.0"
__all__ = [
    "FeatureProcessor",
    "PositionFeatureExtractor", 
    "MarketFeatureExtractor",
    "WSTProcessor"
]