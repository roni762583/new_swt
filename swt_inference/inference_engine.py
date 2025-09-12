"""
Inference Engine
Main orchestration component for SWT trading inference

Provides unified interface for trading decisions across all agent types.
Coordinates feature processing, agent selection, and result formatting.
"""

import time
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from swt_core.types import AgentType, TradingDecision, PositionState, MarketState
from swt_core.config_manager import SWTConfig, ConfigManager
from swt_core.exceptions import InferenceError, ConfigurationError, FeatureProcessingError

from swt_features.feature_processor import FeatureProcessor, ProcessedObservation, MarketDataPoint
from .agent_factory import AgentFactory, BaseAgent
from .checkpoint_loader import CheckpointLoader

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Complete result from inference engine"""
    trading_decision: TradingDecision
    processing_time_ms: float
    agent_info: Dict[str, Any]
    feature_metadata: Dict[str, Any]
    system_status: Dict[str, Any]
    inference_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "trading_decision": {
                "action": self.trading_decision.action,
                "confidence": self.trading_decision.confidence,
                "value_estimate": self.trading_decision.value_estimate,
                "policy_distribution": self.trading_decision.policy_distribution.tolist() if hasattr(self.trading_decision.policy_distribution, 'tolist') else self.trading_decision.policy_distribution,
                "mcts_visits": self.trading_decision.mcts_visits.tolist() if hasattr(self.trading_decision.mcts_visits, 'tolist') else self.trading_decision.mcts_visits,
                "agent_type": self.trading_decision.agent_type.value,
                "model_confidence": self.trading_decision.model_confidence
            },
            "processing_time_ms": self.processing_time_ms,
            "agent_info": self.agent_info,
            "feature_metadata": self.feature_metadata,
            "system_status": self.system_status,
            "inference_id": self.inference_id,
            "timestamp": self.timestamp.isoformat()
        }


class InferenceEngine:
    """
    Main inference engine for SWT trading system
    
    Coordinates all components to provide unified trading decisions:
    - Feature processing (market + position)
    - Agent selection and switching
    - MCTS search execution
    - Result formatting and validation
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None,
                 config: Optional[SWTConfig] = None):
        """
        Initialize inference engine
        
        Args:
            config_path: Path to configuration file
            config: Pre-loaded configuration (takes precedence)
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            config_manager = ConfigManager()
            self.config = config_manager.load_config(config_path)
        else:
            # Load default configuration
            config_manager = ConfigManager()
            self.config = config_manager.get_default_config()
        
        # Initialize components
        self.feature_processor = FeatureProcessor(self.config)
        self.checkpoint_loader = CheckpointLoader(self.config)
        
        # Agent management
        self.current_agent: Optional[BaseAgent] = None
        self.agent_checkpoint_path: Optional[str] = None
        
        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0
        self._last_inference_time = None
        
        # Status tracking
        self._is_ready = False
        self._initialization_time = datetime.now()
        
        # Handle missing agent_system attribute for Episode 13475 compatibility
        agent_system = getattr(self.config, 'agent_system', 'stochastic_muzero')
        if hasattr(agent_system, 'value'):
            agent_system = agent_system.value
        logger.info(f"🚀 InferenceEngine initialized: agent_system={agent_system}")
    
    def initialize_agent(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Initialize trading agent with checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        try:
            checkpoint_path = str(checkpoint_path)
            
            # Create agent using factory
            agent = AgentFactory.create_agent(self.config)
            
            # Load checkpoint
            agent.load_checkpoint(checkpoint_path)
            
            # Set as current agent
            self.current_agent = agent
            self.agent_checkpoint_path = checkpoint_path
            self._is_ready = True
            
            logger.info(f"✅ Agent initialized: {self.config.agent_system.value} from {checkpoint_path}")
            
        except Exception as e:
            self._is_ready = False
            raise ConfigurationError(
                f"Failed to initialize agent: {str(e)}",
                context={
                    "checkpoint_path": str(checkpoint_path),
                    "agent_type": self.config.agent_system.value
                },
                original_error=e
            )
    
    def switch_agent_system(self, new_agent_type: AgentType,
                           checkpoint_path: Optional[str] = None) -> None:
        """
        Switch to different agent system
        
        Args:
            new_agent_type: New agent type to switch to
            checkpoint_path: Optional new checkpoint path
        """
        try:
            old_agent_type = self.config.agent_system
            
            # Update configuration
            self.config.switch_agent_system(new_agent_type, validate=True)
            
            # Create new agent
            new_agent = AgentFactory.create_agent(self.config)
            
            # Load checkpoint (use existing path if not provided)
            checkpoint_to_load = checkpoint_path or self.agent_checkpoint_path
            if checkpoint_to_load:
                new_agent.load_checkpoint(checkpoint_to_load)
                self.agent_checkpoint_path = checkpoint_to_load
            else:
                logger.warning("⚠️ No checkpoint path available for new agent")
                self._is_ready = False
            
            # Switch agents
            old_agent = self.current_agent
            self.current_agent = new_agent
            
            # Log performance stats from old agent if available
            if old_agent:
                try:
                    old_stats = old_agent.get_inference_stats()
                    logger.info(f"📊 Previous agent stats: {old_stats}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get old agent stats: {e}")
            
            logger.info(f"🔄 Agent system switched: {old_agent_type.value} → {new_agent_type.value}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to switch agent system: {str(e)}",
                context={
                    "old_agent": self.config.agent_system.value,
                    "new_agent": new_agent_type.value,
                    "checkpoint_path": checkpoint_path
                },
                original_error=e
            )
    
    def add_market_data(self, data_point: MarketDataPoint) -> None:
        """
        Add market data to feature processor
        
        Args:
            data_point: New market data point
        """
        try:
            self.feature_processor.add_market_data(data_point)
            
        except Exception as e:
            raise FeatureProcessingError(
                f"Failed to add market data: {str(e)}",
                context={"timestamp": data_point.timestamp},
                original_error=e
            )
    
    def get_trading_decision(self, position_state: PositionState,
                           current_price: float,
                           deterministic: bool = False,
                           market_metadata: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """
        Get trading decision from current market state and position
        
        Args:
            position_state: Current position state
            current_price: Current market price
            deterministic: Whether to use deterministic inference
            market_metadata: Additional market information
            
        Returns:
            InferenceResult with trading decision and metadata
        """
        if not self._is_ready or self.current_agent is None:
            raise InferenceError(
                "Inference engine not ready - initialize agent first",
                context={"is_ready": self._is_ready, "has_agent": self.current_agent is not None}
            )
        
        start_time = time.time()
        inference_id = f"inf_{int(start_time * 1000000)}"
        
        try:
            # Process observation
            observation = self.feature_processor.process_observation(
                position_state=position_state,
                current_price=current_price,
                market_cache_key=f"price_{current_price}_{int(time.time())}",
                market_metadata=market_metadata
            )
            
            # Get trading decision from agent
            trading_decision = self.current_agent.get_trading_decision(
                observation=observation,
                deterministic=deterministic
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Gather system information
            agent_info = self._get_agent_info()
            feature_metadata = observation.metadata
            system_status = self._get_system_status()
            
            # Create inference result
            result = InferenceResult(
                trading_decision=trading_decision,
                processing_time_ms=processing_time,
                agent_info=agent_info,
                feature_metadata=feature_metadata,
                system_status=system_status,
                inference_id=inference_id,
                timestamp=datetime.now()
            )
            
            # Update performance statistics
            self._inference_count += 1
            self._total_inference_time += processing_time
            self._last_inference_time = processing_time
            
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            raise InferenceError(
                f"Trading decision inference failed: {str(e)}",
                context={
                    "inference_id": inference_id,
                    "processing_time_ms": processing_time,
                    "agent_type": self.config.agent_system.value,
                    "position_type": position_state.position_type.name,
                    "current_price": current_price
                },
                original_error=e
            )
    
    def _get_agent_info(self) -> Dict[str, Any]:
        """Get current agent information"""
        if self.current_agent is None:
            return {"status": "no_agent"}
        
        try:
            model_info = self.current_agent.get_model_info()
            inference_stats = self.current_agent.get_inference_stats()
            
            return {
                "agent_type": self.config.agent_system.value,
                "model_info": model_info,
                "inference_stats": inference_stats,
                "checkpoint_path": self.agent_checkpoint_path
            }
            
        except Exception as e:
            return {
                "agent_type": self.config.agent_system.value,
                "error": f"Could not get agent info: {str(e)}"
            }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            feature_status = self.feature_processor.get_system_status()
            
            return {
                "is_ready": self._is_ready,
                "inference_count": self._inference_count,
                "average_inference_time_ms": self._total_inference_time / max(1, self._inference_count),
                "last_inference_time_ms": self._last_inference_time,
                "feature_processor": feature_status,
                "agent_system": self.config.agent_system.value,
                "initialization_time": self._initialization_time.isoformat()
            }
            
        except Exception as e:
            return {
                "is_ready": self._is_ready,
                "error": f"Could not get full system status: {str(e)}"
            }
    
    def is_ready(self) -> bool:
        """Check if inference engine is ready for trading decisions"""
        return (self._is_ready and 
                self.current_agent is not None and 
                self.feature_processor.is_ready())
    
    def get_diagnostics(self, position_state: Optional[PositionState] = None,
                       current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information
        
        Args:
            position_state: Optional position state for detailed diagnostics
            current_price: Optional current price for feature diagnostics
            
        Returns:
            Comprehensive diagnostic information
        """
        diagnostics = {
            "inference_engine": {
                "is_ready": self.is_ready(),
                "agent_initialized": self.current_agent is not None,
                "feature_processor_ready": self.feature_processor.is_ready(),
                "configuration": {
                    "agent_system": self.config.agent_system.value,
                    "checkpoint_path": self.agent_checkpoint_path
                }
            },
            "performance": {
                "total_inferences": self._inference_count,
                "total_inference_time_ms": self._total_inference_time,
                "average_inference_time_ms": self._total_inference_time / max(1, self._inference_count),
                "last_inference_time_ms": self._last_inference_time
            }
        }
        
        # Add agent diagnostics
        if self.current_agent:
            try:
                agent_info = self._get_agent_info()
                diagnostics["agent"] = agent_info
            except Exception as e:
                diagnostics["agent"] = {"error": str(e)}
        
        # Add feature processor diagnostics
        if position_state and current_price:
            try:
                feature_diagnostics = self.feature_processor.get_diagnostics(position_state, current_price)
                diagnostics["features"] = feature_diagnostics
            except Exception as e:
                diagnostics["features"] = {"error": str(e)}
        else:
            try:
                system_status = self.feature_processor.get_system_status()
                diagnostics["features"] = system_status
            except Exception as e:
                diagnostics["features"] = {"error": str(e)}
        
        return diagnostics
    
    def reset(self, clear_market_data: bool = False) -> None:
        """
        Reset inference engine state
        
        Args:
            clear_market_data: Whether to clear market data buffers
        """
        try:
            # Reset feature processor
            self.feature_processor.reset(clear_market_data=clear_market_data)
            
            # Reset performance statistics
            self._inference_count = 0
            self._total_inference_time = 0.0
            self._last_inference_time = None
            
            logger.info(f"🔄 InferenceEngine reset (clear_market_data={clear_market_data})")
            
        except Exception as e:
            logger.error(f"❌ InferenceEngine reset failed: {e}")
            raise
    
    def save_state(self, output_dir: Union[str, Path]) -> None:
        """
        Save inference engine state
        
        Args:
            output_dir: Directory to save state files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save feature processor cache
            self.feature_processor.save_cache(output_path / "feature_cache")
            
            # Save engine statistics
            stats_file = output_path / "inference_stats.json"
            import json
            
            stats = {
                "inference_count": self._inference_count,
                "total_inference_time_ms": self._total_inference_time,
                "last_inference_time_ms": self._last_inference_time,
                "initialization_time": self._initialization_time.isoformat(),
                "agent_type": self.config.agent_system.value,
                "checkpoint_path": self.agent_checkpoint_path,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"💾 InferenceEngine state saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save InferenceEngine state: {e}")
            raise
    
    def load_state(self, input_dir: Union[str, Path]) -> None:
        """
        Load inference engine state
        
        Args:
            input_dir: Directory containing state files
        """
        try:
            input_path = Path(input_dir)
            
            # Load feature processor cache
            self.feature_processor.load_cache(input_path / "feature_cache")
            
            # Load engine statistics if available
            stats_file = input_path / "inference_stats.json"
            if stats_file.exists():
                import json
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                # Restore statistics (but not agent state)
                self._inference_count = stats.get("inference_count", 0)
                self._total_inference_time = stats.get("total_inference_time_ms", 0.0)
                self._last_inference_time = stats.get("last_inference_time_ms", None)
                
                logger.info(f"📊 Restored inference statistics: {self._inference_count} inferences")
            
            logger.info(f"📁 InferenceEngine state loaded from {input_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not fully load InferenceEngine state: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        if self._inference_count == 0:
            return {"status": "no_inferences_yet"}
        
        avg_time = self._total_inference_time / self._inference_count
        
        return {
            "total_inferences": self._inference_count,
            "average_inference_time_ms": avg_time,
            "last_inference_time_ms": self._last_inference_time,
            "total_inference_time_ms": self._total_inference_time,
            "agent_type": self.config.agent_system.value,
            "is_ready": self.is_ready(),
            "uptime_seconds": (datetime.now() - self._initialization_time).total_seconds()
        }
    
    def create_mock_inference_result(self, action: int = 0) -> InferenceResult:
        """Create mock inference result for testing"""
        from swt_core.types import TradingDecision, PositionType
        import numpy as np
        
        # Create mock trading decision
        trading_decision = TradingDecision(
            action=action,
            confidence=0.75,
            value_estimate=0.1,
            policy_distribution=np.array([0.1, 0.2, 0.4, 0.3]),
            mcts_visits=np.array([10, 20, 40, 30]),
            mcts_simulations=100,
            search_time_ms=50.0,
            agent_type=self.config.agent_system,
            model_confidence=0.8
        )
        
        return InferenceResult(
            trading_decision=trading_decision,
            processing_time_ms=75.0,
            agent_info={"mock": True},
            feature_metadata={"mock": True},
            system_status={"mock": True},
            inference_id="mock_inference",
            timestamp=datetime.now()
        )