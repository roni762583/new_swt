"""
MCTS Engine
Unified Monte Carlo Tree Search implementation supporting multiple algorithms

Provides consistent MCTS interface for:
- Standard Stochastic MuZero MCTS
- ReZero MCTS optimizations 
- Gumbel action selection enhancements
- Cache management and performance optimization
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from swt_core.types import AgentType, MCTSConfig
from swt_core.exceptions import InferenceError, ConfigurationError
from swt_features.feature_processor import ProcessedObservation

logger = logging.getLogger(__name__)


class MCTSVariant(Enum):
    """MCTS algorithm variants"""
    STANDARD = "standard"
    REZERO = "rezero"
    GUMBEL = "gumbel"
    UNIZERO = "unizero"


@dataclass
class MCTSResult:
    """Result from MCTS search"""
    selected_action: int
    action_probabilities: np.ndarray
    visit_counts: np.ndarray
    action_confidence: float
    root_value: float
    policy_logits: np.ndarray
    num_simulations: int
    search_time_ms: float
    cache_hits: int = 0
    tree_depth: int = 0
    nodes_expanded: int = 0
    algorithm_used: str = "standard"
    
    def validate(self) -> bool:
        """Validate MCTS result integrity"""
        try:
            # Check action bounds
            if not (0 <= self.selected_action <= 3):
                return False
            
            # Check probability distributions
            if self.action_probabilities.shape != (4,):
                return False
            
            if self.visit_counts.shape != (4,):
                return False
            
            # Check probability sum (approximately 1.0)
            if not (0.95 <= np.sum(self.action_probabilities) <= 1.05):
                return False
            
            # Check for invalid values
            if np.isnan(self.action_probabilities).any() or np.isinf(self.action_probabilities).any():
                return False
            
            if np.isnan(self.root_value) or np.isinf(self.root_value):
                return False
            
            return True
            
        except Exception:
            return False


class MCTSEngine:
    """
    Unified MCTS engine supporting multiple algorithms and optimizations
    
    Handles different MCTS variants while providing consistent interface
    for the inference system.
    """
    
    def __init__(self, networks: Any, config: MCTSConfig, agent_type: AgentType):
        """
        Initialize MCTS engine
        
        Args:
            networks: Neural network collection
            config: MCTS configuration
            agent_type: Agent type for algorithm selection
        """
        self.networks = networks
        self.config = config
        self.agent_type = agent_type
        
        # Determine MCTS variant based on configuration
        self.variant = self._determine_mcts_variant()
        
        # Initialize variant-specific components
        self._initialize_variant_components()
        
        # Performance tracking
        self._total_searches = 0
        self._total_search_time = 0.0
        self._cache_hits = 0
        
        logger.info(f"🌳 MCTSEngine initialized: variant={self.variant.value}, agent={agent_type.value}")
    
    def run_mcts(self, observation: ProcessedObservation, 
                 num_simulations: int, 
                 deterministic: bool = False) -> MCTSResult:
        """
        Run MCTS search on observation
        
        Args:
            observation: Processed observation from feature processor
            num_simulations: Number of MCTS simulations to run
            deterministic: Whether to use deterministic action selection
            
        Returns:
            MCTSResult with action selection and search statistics
        """
        start_time = time.time()
        
        try:
            # Validate observation
            if not self._validate_observation(observation):
                raise InferenceError("Invalid observation for MCTS")
            
            # Run variant-specific MCTS
            if self.variant == MCTSVariant.STANDARD:
                result = self._run_standard_mcts(observation, num_simulations, deterministic)
            elif self.variant == MCTSVariant.REZERO:
                result = self._run_rezero_mcts(observation, num_simulations, deterministic)
            elif self.variant == MCTSVariant.GUMBEL:
                result = self._run_gumbel_mcts(observation, num_simulations, deterministic)
            else:
                # Fallback to standard
                logger.warning(f"⚠️ Unknown MCTS variant {self.variant}, using standard")
                result = self._run_standard_mcts(observation, num_simulations, deterministic)
            
            # Record timing and validate result
            search_time = (time.time() - start_time) * 1000
            result.search_time_ms = search_time
            result.algorithm_used = self.variant.value
            
            if not result.validate():
                raise InferenceError(f"MCTS result validation failed: {self.variant.value}")
            
            # Update performance statistics
            self._total_searches += 1
            self._total_search_time += search_time
            
            return result
            
        except Exception as e:
            search_time = (time.time() - start_time) * 1000
            raise InferenceError(
                f"MCTS search failed ({self.variant.value}): {str(e)}",
                context={
                    "variant": self.variant.value,
                    "simulations": num_simulations,
                    "search_time_ms": search_time
                },
                original_error=e
            )
    
    def _determine_mcts_variant(self) -> MCTSVariant:
        """Determine MCTS variant from configuration"""
        if self.config.enable_rezero_optimization:
            return MCTSVariant.REZERO
        elif self.config.enable_gumbel_selection:
            return MCTSVariant.GUMBEL
        elif getattr(self.config, 'enable_unizero', False):
            return MCTSVariant.UNIZERO
        else:
            return MCTSVariant.STANDARD
    
    def _initialize_variant_components(self) -> None:
        """Initialize components specific to MCTS variant"""
        try:
            if self.variant == MCTSVariant.REZERO:
                # Import and initialize ReZero MCTS
                from experimental_research.rezero_mcts import ReZeroMCTS
                self._rezero_mcts = ReZeroMCTS(
                    networks=self.networks,
                    config=self.config
                )
                
            elif self.variant == MCTSVariant.GUMBEL:
                # Import and initialize Gumbel components
                from experimental_research.gumbel_action_selection import GumbelActionSelector
                self._gumbel_selector = GumbelActionSelector(
                    temperature=getattr(self.config, 'gumbel_temperature', 1.0)
                )
                
                # Also initialize standard MCTS for Gumbel variant
                from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
                self._standard_mcts = SWTMCTSRunner(
                    networks=self.networks,
                    config=self.config
                )
            
            else:
                # Standard MCTS
                from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
                self._standard_mcts = SWTMCTSRunner(
                    networks=self.networks,
                    config=self.config
                )
                
        except ImportError as e:
            logger.warning(f"⚠️ Could not import {self.variant.value} components: {e}")
            # Fallback to standard
            self.variant = MCTSVariant.STANDARD
            from experimental_research.swt_core.swt_mcts import SWTMCTSRunner
            self._standard_mcts = SWTMCTSRunner(
                networks=self.networks,
                config=self.config
            )
    
    def _run_standard_mcts(self, observation: ProcessedObservation,
                          num_simulations: int, deterministic: bool) -> MCTSResult:
        """Run standard Stochastic MuZero MCTS"""
        try:
            # Run MCTS through standard runner
            mcts_result = self._standard_mcts.run_mcts(
                observation=observation,
                num_simulations=num_simulations,
                deterministic=deterministic
            )
            
            # Convert to unified MCTSResult format
            return MCTSResult(
                selected_action=mcts_result.selected_action,
                action_probabilities=mcts_result.action_probabilities,
                visit_counts=mcts_result.visit_counts,
                action_confidence=mcts_result.action_confidence,
                root_value=mcts_result.root_value,
                policy_logits=getattr(mcts_result, 'policy_logits', mcts_result.action_probabilities),
                num_simulations=num_simulations,
                search_time_ms=0.0,  # Will be set by caller
                nodes_expanded=getattr(mcts_result, 'nodes_expanded', num_simulations)
            )
            
        except Exception as e:
            raise InferenceError(f"Standard MCTS failed: {str(e)}", original_error=e)
    
    def _run_rezero_mcts(self, observation: ProcessedObservation,
                        num_simulations: int, deterministic: bool) -> MCTSResult:
        """Run ReZero optimized MCTS"""
        try:
            # Run enhanced MCTS with ReZero optimizations
            mcts_result = self._rezero_mcts.run_mcts(
                observation=observation,
                num_simulations=num_simulations,
                deterministic=deterministic
            )
            
            # Extract ReZero-specific metrics
            cache_hits = getattr(mcts_result, 'cache_hits', 0)
            self._cache_hits += cache_hits
            
            return MCTSResult(
                selected_action=mcts_result.selected_action,
                action_probabilities=mcts_result.action_probabilities,
                visit_counts=mcts_result.visit_counts,
                action_confidence=mcts_result.action_confidence,
                root_value=mcts_result.root_value,
                policy_logits=getattr(mcts_result, 'policy_logits', mcts_result.action_probabilities),
                num_simulations=num_simulations,
                search_time_ms=0.0,  # Will be set by caller
                cache_hits=cache_hits,
                tree_depth=getattr(mcts_result, 'max_depth', 0),
                nodes_expanded=getattr(mcts_result, 'nodes_expanded', num_simulations)
            )
            
        except Exception as e:
            # Fallback to standard MCTS
            logger.warning(f"⚠️ ReZero MCTS failed, falling back to standard: {e}")
            return self._run_standard_mcts(observation, num_simulations, deterministic)
    
    def _run_gumbel_mcts(self, observation: ProcessedObservation,
                        num_simulations: int, deterministic: bool) -> MCTSResult:
        """Run MCTS with Gumbel action selection"""
        try:
            # First run standard MCTS to get base result
            base_result = self._run_standard_mcts(observation, num_simulations, deterministic)
            
            # Apply Gumbel selection if not deterministic
            if not deterministic and hasattr(self, '_gumbel_selector'):
                gumbel_result = self._gumbel_selector.select_action(
                    policy_logits=base_result.policy_logits,
                    visit_counts=base_result.visit_counts
                )
                
                # Update result with Gumbel selection
                base_result.selected_action = gumbel_result.selected_action
                base_result.action_confidence = gumbel_result.confidence
            
            return base_result
            
        except Exception as e:
            # Fallback to standard MCTS
            logger.warning(f"⚠️ Gumbel MCTS failed, falling back to standard: {e}")
            return self._run_standard_mcts(observation, num_simulations, deterministic)
    
    def _validate_observation(self, observation: ProcessedObservation) -> bool:
        """Validate observation for MCTS processing"""
        try:
            # Check market features
            if observation.market_features.shape != (128,):
                return False
            
            # Check position features
            if observation.position_features.shape != (9,):
                return False
            
            # Check for invalid values
            if np.isnan(observation.combined_features).any():
                return False
            
            if np.isinf(observation.combined_features).any():
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get MCTS engine performance statistics"""
        avg_search_time = self._total_search_time / max(1, self._total_searches)
        
        return {
            "variant": self.variant.value,
            "total_searches": self._total_searches,
            "total_search_time_ms": self._total_search_time,
            "average_search_time_ms": avg_search_time,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_searches),
            "simulations_per_search": self.config.num_simulations,
            "c_puct": self.config.c_puct,
            "discount_factor": self.config.discount_factor
        }
    
    def clear_cache(self) -> None:
        """Clear MCTS caches and reset statistics"""
        # Clear variant-specific caches
        if hasattr(self, '_rezero_mcts'):
            if hasattr(self._rezero_mcts, 'clear_cache'):
                self._rezero_mcts.clear_cache()
        
        if hasattr(self, '_standard_mcts'):
            if hasattr(self._standard_mcts, 'clear_cache'):
                self._standard_mcts.clear_cache()
        
        # Reset statistics
        self._cache_hits = 0
        
        logger.info(f"🧹 MCTS engine cache cleared ({self.variant.value})")
    
    def update_config(self, new_config: MCTSConfig) -> None:
        """Update MCTS configuration"""
        old_variant = self.variant
        self.config = new_config
        
        # Check if variant changed
        new_variant = self._determine_mcts_variant()
        if new_variant != old_variant:
            logger.info(f"🔄 MCTS variant changed: {old_variant.value} → {new_variant.value}")
            self.variant = new_variant
            self._initialize_variant_components()
        
        # Update component configurations
        if hasattr(self, '_standard_mcts'):
            self._standard_mcts.config = new_config
        
        if hasattr(self, '_rezero_mcts'):
            self._rezero_mcts.config = new_config
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get current tree structure statistics"""
        stats = {
            "variant": self.variant.value,
            "config": {
                "simulations": self.config.num_simulations,
                "c_puct": self.config.c_puct,
                "discount": self.config.discount_factor
            }
        }
        
        # Get variant-specific tree stats
        if hasattr(self, '_rezero_mcts') and hasattr(self._rezero_mcts, 'get_tree_stats'):
            stats["rezero_stats"] = self._rezero_mcts.get_tree_stats()
        
        if hasattr(self, '_standard_mcts') and hasattr(self._standard_mcts, 'get_tree_stats'):
            stats["standard_stats"] = self._standard_mcts.get_tree_stats()
        
        return stats
    
    def warmup(self, observation: ProcessedObservation, iterations: int = 5) -> None:
        """Warm up MCTS engine with dummy runs"""
        logger.info(f"🔥 Warming up MCTS engine ({self.variant.value}) with {iterations} iterations")
        
        try:
            for i in range(iterations):
                # Run with minimal simulations for warmup
                self.run_mcts(observation, num_simulations=10, deterministic=True)
            
            logger.info("✅ MCTS engine warmup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ MCTS warmup failed: {e}")