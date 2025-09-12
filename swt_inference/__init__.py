"""
SWT Inference Module
Unified inference engine supporting seamless algorithm switching

Provides consistent inference interface for:
- Standard Stochastic MuZero
- Experimental Enhanced Agents (EfficientZero + Advanced MuZero)
- Future algorithm implementations

Key Features:
- Factory pattern for agent instantiation
- Shared MCTS implementations
- Algorithm-specific optimizations
- Consistent interface across all agents
"""

from .inference_engine import InferenceEngine, InferenceResult
from .agent_factory import AgentFactory, BaseAgent
from .checkpoint_loader import CheckpointLoader
from .mcts_engine import MCTSEngine, MCTSResult, MCTSVariant

__version__ = "1.0.0"
__all__ = [
    "InferenceEngine",
    "InferenceResult", 
    "AgentFactory",
    "BaseAgent",
    "CheckpointLoader",
    "MCTSEngine",
    "MCTSResult",
    "MCTSVariant"
]