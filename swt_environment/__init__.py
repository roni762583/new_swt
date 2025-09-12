"""
SWT Training Environment
Unified training environment for all SWT agent types

Provides consistent interface for:
- Standard Stochastic MuZero training
- Experimental enhanced training
- Shared environment logic
- AMDDP1 reward system
"""

from .swt_trading_env import SWTTradingEnv, SWTTradingEnvConfig
from .reward_system import AMDDPRewardSystem, RewardResult
from .training_wrapper import SWTTrainingWrapper
from .data_loader import SWTDataLoader, MarketDataBatch

__version__ = "1.0.0"
__all__ = [
    "SWTTradingEnv",
    "SWTTradingEnvConfig", 
    "AMDDPRewardSystem",
    "RewardResult",
    "SWTTrainingWrapper",
    "SWTDataLoader",
    "MarketDataBatch"
]