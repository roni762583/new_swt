"""
Agent Factory
Creates appropriate agent instances based on configuration

Supports seamless switching between different agent algorithms:
- Standard Stochastic MuZero
- Experimental Enhanced Agent  
- Future algorithm implementations
"""

import logging
from typing import Optional, Dict, Any, Type, List
from abc import ABC, abstractmethod

from swt_core.types import AgentType, TradingDecision
from swt_core.config_manager import SWTConfig
from swt_core.exceptions import ConfigurationError, InferenceError
from swt_features.feature_processor import ProcessedObservation

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    Ensures consistent interface across different algorithms
    """
    
    def __init__(self, config: SWTConfig):
        """Initialize base agent"""
        self.config = config
        self.agent_type = config.agent_system
        self._inference_count = 0
        self._total_inference_time = 0.0
        
    @abstractmethod
    def get_trading_decision(self, observation: ProcessedObservation,
                           deterministic: bool = False) -> TradingDecision:
        """
        Get trading decision from observation
        
        Args:
            observation: Processed observation from feature processor
            deterministic: Whether to use deterministic inference
            
        Returns:
            Trading decision with action, confidence, and metadata
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint"""
        pass
    
    @abstractmethod 
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        pass
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_time = self._total_inference_time / max(1, self._inference_count)
        return {
            "inference_count": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_time,
            "agent_type": self.agent_type.value
        }
    
    def _record_inference_time(self, inference_time: float) -> None:
        """Record inference timing"""
        self._inference_count += 1
        self._total_inference_time += inference_time


class StochasticMuZeroAgent(BaseAgent):
    """
    Standard Stochastic MuZero agent implementation
    Uses classic 5-network architecture with standard MCTS
    """
    
    def __init__(self, config: SWTConfig):
        """Initialize Stochastic MuZero agent"""
        super().__init__(config)
        
        # Import experimental research networks (adapted for standard use)
        from experimental_research.swt_models.swt_stochastic_networks import SWTStochasticMuZeroNetworks
        from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
        
        # Initialize networks
        self.networks = None  # Will be loaded from checkpoint
        self.mcts_runner = None
        
        logger.info("🧠 StochasticMuZeroAgent initialized")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load Stochastic MuZero checkpoint"""
        try:
            from swt_inference.checkpoint_loader import CheckpointLoader
            
            loader = CheckpointLoader(self.config)
            checkpoint_data = loader.load_checkpoint(checkpoint_path)
            
            # Initialize networks with loaded weights
            self.networks = checkpoint_data["networks"]
            
            # Initialize MCTS runner
            from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
            self.mcts_runner = SWTMCTSRunner(
                networks=self.networks,
                config=self.config.mcts_config
            )
            
            logger.info(f"✅ Loaded Stochastic MuZero checkpoint: {checkpoint_path}")
            
        except Exception as e:
            raise InferenceError(
                f"Failed to load checkpoint: {str(e)}",
                context={"checkpoint_path": checkpoint_path}
            )
    
    def get_trading_decision(self, observation: ProcessedObservation,
                           deterministic: bool = False) -> TradingDecision:
        """Get trading decision using standard MCTS"""
        if self.networks is None or self.mcts_runner is None:
            raise InferenceError("Agent not initialized - load checkpoint first")
        
        import time
        start_time = time.time()
        
        try:
            # Run MCTS
            mcts_result = self.mcts_runner.run_mcts(
                observation=observation,
                num_simulations=self.config.mcts_config.num_simulations,
                deterministic=deterministic
            )
            
            # Create trading decision
            decision = TradingDecision(
                action=mcts_result.selected_action,
                confidence=mcts_result.action_confidence,
                value_estimate=mcts_result.root_value,
                policy_distribution=mcts_result.action_probabilities,
                mcts_visits=mcts_result.visit_counts,
                mcts_simulations=mcts_result.num_simulations,
                search_time_ms=(time.time() - start_time) * 1000,
                agent_type=self.agent_type,
                model_confidence=mcts_result.root_value
            )
            
            # Record timing
            self._record_inference_time(time.time() - start_time)
            
            return decision
            
        except Exception as e:
            raise InferenceError(
                f"Standard MuZero inference failed: {str(e)}",
                original_error=e
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.networks is None:
            return {"status": "not_loaded"}
        
        return {
            "agent_type": "stochastic_muzero",
            "networks": {
                "representation": str(type(self.networks.representation_network)),
                "dynamics": str(type(self.networks.dynamics_network)),
                "policy": str(type(self.networks.policy_network)),
                "value": str(type(self.networks.value_network)),
                "chance": str(type(self.networks.chance_encoder))
            },
            "mcts_config": {
                "simulations": self.config.mcts_config.num_simulations,
                "c_puct": self.config.mcts_config.c_puct,
                "discount": self.config.mcts_config.discount_factor
            }
        }


class ExperimentalAgent(BaseAgent):
    """
    Experimental agent with EfficientZero + Advanced MuZero enhancements
    Includes all experimental optimizations and enhancements
    """
    
    def __init__(self, config: SWTConfig):
        """Initialize Experimental agent"""
        super().__init__(config)
        
        # Import experimental components
        from experimental_research.efficientzero_main import EfficientZeroSWTTrainer
        from experimental_research.rezero_mcts import ReZeroMCTS
        from experimental_research.gumbel_action_selection import GumbelActionSelector
        
        self.networks = None
        self.mcts_engine = None
        self.gumbel_selector = None
        
        # Enhanced components
        self._consistency_tracker = {}
        self._value_prefix_cache = {}
        
        logger.info("🚀 ExperimentalAgent initialized with enhanced features")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load experimental checkpoint with enhanced features"""
        try:
            from swt_inference.checkpoint_loader import CheckpointLoader
            
            loader = CheckpointLoader(self.config)
            checkpoint_data = loader.load_checkpoint(checkpoint_path)
            
            # Initialize enhanced networks
            self.networks = checkpoint_data["networks"]
            
            # Initialize ReZero MCTS if enabled
            if self.config.mcts_config.enable_rezero_optimization:
                from experimental_research.rezero_mcts import ReZeroMCTS
                self.mcts_engine = ReZeroMCTS(
                    networks=self.networks,
                    config=self.config.mcts_config
                )
            else:
                # Fallback to standard MCTS
                from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
                self.mcts_engine = SWTMCTSRunner(
                    networks=self.networks,
                    config=self.config.mcts_config
                )
            
            # Initialize Gumbel selector if enabled
            if self.config.mcts_config.enable_gumbel_selection:
                from experimental_research.gumbel_action_selection import GumbelActionSelector
                self.gumbel_selector = GumbelActionSelector(
                    temperature=self.config.mcts_config.gumbel_temperature
                )
            
            logger.info(f"✅ Loaded Experimental checkpoint: {checkpoint_path}")
            
        except Exception as e:
            raise InferenceError(
                f"Failed to load experimental checkpoint: {str(e)}",
                context={"checkpoint_path": checkpoint_path}
            )
    
    def get_trading_decision(self, observation: ProcessedObservation,
                           deterministic: bool = False) -> TradingDecision:
        """Get trading decision using experimental enhancements"""
        if self.networks is None or self.mcts_engine is None:
            raise InferenceError("Experimental agent not initialized")
        
        import time
        start_time = time.time()
        
        try:
            # Run enhanced MCTS
            mcts_result = self.mcts_engine.run_mcts(
                observation=observation,
                num_simulations=self.config.mcts_config.num_simulations,
                deterministic=deterministic
            )
            
            # Apply Gumbel action selection if enabled
            if self.gumbel_selector is not None and not deterministic:
                enhanced_result = self.gumbel_selector.select_action(
                    policy_logits=mcts_result.policy_logits,
                    visit_counts=mcts_result.visit_counts
                )
                selected_action = enhanced_result.selected_action
                action_confidence = enhanced_result.confidence
            else:
                selected_action = mcts_result.selected_action
                action_confidence = mcts_result.action_confidence
            
            # Create enhanced trading decision
            decision = TradingDecision(
                action=selected_action,
                confidence=action_confidence,
                value_estimate=mcts_result.root_value,
                policy_distribution=mcts_result.action_probabilities,
                mcts_visits=mcts_result.visit_counts,
                mcts_simulations=mcts_result.num_simulations,
                search_time_ms=(time.time() - start_time) * 1000,
                agent_type=self.agent_type,
                model_confidence=mcts_result.root_value,
                features_used={
                    "rezero_enabled": self.config.mcts_config.enable_rezero_optimization,
                    "gumbel_enabled": self.config.mcts_config.enable_gumbel_selection,
                    "cache_hits": getattr(mcts_result, "cache_hits", 0),
                    "enhanced_features": True
                }
            )
            
            # Record timing
            self._record_inference_time(time.time() - start_time)
            
            return decision
            
        except Exception as e:
            raise InferenceError(
                f"Experimental inference failed: {str(e)}",
                original_error=e
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get experimental model information"""
        if self.networks is None:
            return {"status": "not_loaded"}
        
        base_info = {
            "agent_type": "experimental",
            "enhancements": {
                "consistency_loss": self.config.experimental_config.get("consistency_loss", {}).get("enabled", False),
                "value_prefix": self.config.experimental_config.get("value_prefix", {}).get("enabled", False),
                "off_policy_correction": self.config.experimental_config.get("off_policy_correction", {}).get("enabled", False),
                "gumbel_selection": self.config.mcts_config.enable_gumbel_selection,
                "rezero_mcts": self.config.mcts_config.enable_rezero_optimization
            },
            "networks": {
                "representation": str(type(self.networks.representation_network)),
                "dynamics": str(type(self.networks.dynamics_network)),
                "policy": str(type(self.networks.policy_network)),
                "value": str(type(self.networks.value_network)),
                "chance": str(type(self.networks.chance_encoder))
            }
        }
        
        # Add value-prefix network info if available
        if hasattr(self.networks, "value_prefix_network"):
            base_info["networks"]["value_prefix"] = str(type(self.networks.value_prefix_network))
        
        return base_info


class AgentFactory:
    """
    Factory class for creating appropriate agent instances
    Handles seamless switching between different agent algorithms
    """
    
    # Registry of available agents
    _agent_registry: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.STOCHASTIC_MUZERO: StochasticMuZeroAgent,
        AgentType.EXPERIMENTAL: ExperimentalAgent
    }
    
    @classmethod
    def create_agent(cls, config: SWTConfig) -> BaseAgent:
        """
        Create appropriate agent based on configuration
        
        Args:
            config: SWT configuration specifying agent type
            
        Returns:
            Initialized agent instance
            
        Raises:
            ConfigurationError: If agent type is not supported
        """
        agent_type = config.agent_system
        
        if agent_type not in cls._agent_registry:
            available_types = list(cls._agent_registry.keys())
            raise ConfigurationError(
                f"Unsupported agent type: {agent_type}",
                context={
                    "requested_type": agent_type.value,
                    "available_types": [t.value for t in available_types]
                }
            )
        
        agent_class = cls._agent_registry[agent_type]
        
        try:
            agent = agent_class(config)
            logger.info(f"🏭 Created agent: {agent_type.value}")
            return agent
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create agent of type {agent_type.value}: {str(e)}",
                original_error=e
            )
    
    @classmethod
    def register_agent(cls, agent_type: AgentType, agent_class: Type[BaseAgent]) -> None:
        """
        Register new agent type
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class to register
        """
        cls._agent_registry[agent_type] = agent_class
        logger.info(f"📝 Registered new agent type: {agent_type.value}")
    
    @classmethod
    def get_available_agents(cls) -> List[AgentType]:
        """Get list of available agent types"""
        return list(cls._agent_registry.keys())
    
    @classmethod
    def switch_agent_type(cls, current_agent: BaseAgent, 
                         new_config: SWTConfig) -> BaseAgent:
        """
        Switch to different agent type while preserving inference state
        
        Args:
            current_agent: Current agent instance
            new_config: Configuration for new agent type
            
        Returns:
            New agent instance
        """
        logger.info(f"🔄 Switching agent: {current_agent.agent_type.value} → {new_config.agent_system.value}")
        
        # Create new agent
        new_agent = cls.create_agent(new_config)
        
        # Transfer any transferable state (if possible)
        try:
            inference_stats = current_agent.get_inference_stats()
            logger.info(f"📊 Previous agent stats: {inference_stats}")
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve previous agent stats: {e}")
        
        logger.info(f"✅ Successfully switched to {new_config.agent_system.value}")
        
        return new_agent