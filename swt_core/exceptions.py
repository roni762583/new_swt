"""
SWT Core Exceptions
Custom exception hierarchy for the SWT trading system

Provides specific exceptions for different system components with
clear error messages and context for debugging.
"""

from typing import Optional, Any, Dict


class SWTException(Exception):
    """
    Base exception for all SWT system errors
    Provides context and debugging information
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None) -> None:
        """
        Initialize SWT exception with context
        
        Args:
            message: Human-readable error description
            context: Additional context for debugging
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """String representation with context"""
        error_str = self.message
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            error_str += f" (Context: {context_str})"
        
        if self.original_error:
            error_str += f" (Caused by: {type(self.original_error).__name__}: {self.original_error})"
        
        return error_str


class ConfigurationError(SWTException):
    """
    Configuration-related errors
    Raised when configuration validation fails or required parameters are missing
    """
    pass


class FeatureProcessingError(SWTException):
    """
    Feature processing errors
    Raised when WST transforms, position feature extraction, or validation fails
    """
    pass


class InferenceError(SWTException):
    """
    Neural network inference errors
    Raised when model loading, forward passes, or MCTS fails
    """
    pass


class TradingError(SWTException):
    """
    Trading execution errors
    Raised when order placement, position management, or API communication fails
    """
    pass


class DataValidationError(SWTException):
    """
    Data validation errors
    Raised when market data, observations, or actions are invalid
    """
    pass


class NetworkArchitectureError(SWTException):
    """
    Neural network architecture errors
    Raised when network construction or parameter loading fails
    """
    pass


class MCTSError(SWTException):
    """
    MCTS-related errors
    Raised when tree search, node expansion, or value computation fails
    """
    pass


class CheckpointError(SWTException):
    """
    Model checkpoint errors
    Raised when checkpoint loading, saving, or validation fails
    """
    pass


class EnvironmentError(SWTException):
    """
    Environment-related errors
    Raised when environment step, reset, or state management fails
    """
    pass


class LiveTradingError(TradingError):
    """
    Live trading specific errors
    Raised when real-money trading operations fail
    """
    pass