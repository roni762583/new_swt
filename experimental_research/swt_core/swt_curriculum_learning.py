#!/usr/bin/env python3
"""
SWT Curriculum Learning System
Progressive training with full reward reassignment for WST-Enhanced Stochastic MuZero

Based on V7/V8 curriculum learning but adapted for:
- WST multi-resolution market analysis
- AMDDP5 reward system progression
- Stochastic MuZero uncertainty handling
- Progressive complexity scaling
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages for SWT system"""
    BASIC_HOLD = "basic_hold"           # Learn to hold positions
    SIMPLE_TRADES = "simple_trades"     # Basic buy/sell actions  
    RISK_AWARENESS = "risk_awareness"   # AMDDP5 drawdown penalties
    FULL_COMPLEXITY = "full_complexity" # Complete market dynamics
    ADVANCED_WST = "advanced_wst"       # Full WST pattern recognition


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    enable_curriculum: bool = True
    initial_stage: CurriculumStage = CurriculumStage.BASIC_HOLD
    stage_transition_episodes: int = 100  # Episodes per stage
    performance_threshold: float = 0.6    # Win rate threshold for progression
    reassignment_frequency: int = 50      # Full reward reassignment frequency
    complexity_scaling: bool = True       # Progressive complexity increase
    adaptive_transitions: bool = True     # Performance-based transitions


class SWTCurriculumLearning:
    """
    Curriculum learning system for SWT-Enhanced Stochastic MuZero
    
    Implements progressive training complexity:
    1. Basic position holding
    2. Simple trading actions
    3. Risk-aware AMDDP5 rewards  
    4. Full market complexity
    5. Advanced WST pattern recognition
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_stage = config.initial_stage
        self.episodes_in_stage = 0
        self.stage_performance_history: List[float] = []
        self.stage_transitions: List[Tuple[int, CurriculumStage]] = []
        
        # Performance tracking
        self.performance_window = 50  # Episodes to evaluate performance
        self.min_episodes_per_stage = 25  # Minimum episodes before transition
        
        logger.info(f"📚 SWT Curriculum Learning initialized")
        logger.info(f"   Starting stage: {self.current_stage.value}")
        logger.info(f"   Adaptive transitions: {config.adaptive_transitions}")
        logger.info(f"   Reassignment frequency: {config.reassignment_frequency}")
        
    def get_current_stage_config(self) -> Dict[str, Any]:
        """Get configuration for current curriculum stage"""
        
        stage_configs = {
            CurriculumStage.BASIC_HOLD: {
                'allowed_actions': ['hold'],
                'reward_scaling': 0.5,
                'wst_complexity': 'basic',  # Simplified WST features
                'amddp5_enabled': False,
                'max_position_duration': 10,
                'description': 'Learn basic position holding'
            },
            
            CurriculumStage.SIMPLE_TRADES: {
                'allowed_actions': ['hold', 'buy', 'sell'],
                'reward_scaling': 0.7,
                'wst_complexity': 'intermediate',
                'amddp5_enabled': False,
                'max_position_duration': 30,
                'description': 'Learn simple trading actions'
            },
            
            CurriculumStage.RISK_AWARENESS: {
                'allowed_actions': ['hold', 'buy', 'sell', 'close'],
                'reward_scaling': 0.85,
                'wst_complexity': 'intermediate',
                'amddp5_enabled': True,
                'amddp5_threshold': 3.0,  # Easier threshold initially
                'max_position_duration': 60,
                'description': 'Learn risk management with AMDDP5'
            },
            
            CurriculumStage.FULL_COMPLEXITY: {
                'allowed_actions': ['hold', 'buy', 'sell', 'close'],
                'reward_scaling': 1.0,
                'wst_complexity': 'full',
                'amddp5_enabled': True,
                'amddp5_threshold': 5.0,  # Full 5% drawdown threshold
                'max_position_duration': 120,
                'description': 'Full market complexity'
            },
            
            CurriculumStage.ADVANCED_WST: {
                'allowed_actions': ['hold', 'buy', 'sell', 'close'],
                'reward_scaling': 1.0,
                'wst_complexity': 'advanced',  # Full WST pattern recognition
                'amddp5_enabled': True,
                'amddp5_threshold': 5.0,
                'max_position_duration': 240,
                'use_advanced_features': True,
                'description': 'Advanced WST pattern mastery'
            }
        }
        
        return stage_configs.get(self.current_stage, stage_configs[CurriculumStage.FULL_COMPLEXITY])
        
    def update_performance(self, episode_reward: float, win_rate: float, 
                          episode: int) -> bool:
        """
        Update performance tracking and check for stage transitions
        
        Args:
            episode_reward: Reward for the episode
            win_rate: Current win rate
            episode: Episode number
            
        Returns:
            Whether stage transition occurred
        """
        
        self.episodes_in_stage += 1
        self.stage_performance_history.append(win_rate)
        
        # Keep performance history within window
        if len(self.stage_performance_history) > self.performance_window:
            self.stage_performance_history.pop(0)
            
        # Check for stage transition
        should_transition = False
        
        if self.config.adaptive_transitions:
            # Performance-based transition
            if (len(self.stage_performance_history) >= self.min_episodes_per_stage and
                self.episodes_in_stage >= self.min_episodes_per_stage):
                
                recent_performance = np.mean(self.stage_performance_history[-self.min_episodes_per_stage:])
                
                if recent_performance >= self.config.performance_threshold:
                    should_transition = True
                    logger.info(f"🎯 Performance threshold met: {recent_performance:.3f} >= {self.config.performance_threshold}")
                    
        else:
            # Episode-based transition
            if self.episodes_in_stage >= self.config.stage_transition_episodes:
                should_transition = True
                
        # Execute transition if needed
        if should_transition and self._can_advance_stage():
            return self._advance_to_next_stage(episode)
            
        return False
        
    def _can_advance_stage(self) -> bool:
        """Check if we can advance to the next stage"""
        current_stages = list(CurriculumStage)
        current_index = current_stages.index(self.current_stage)
        return current_index < len(current_stages) - 1
        
    def _advance_to_next_stage(self, episode: int) -> bool:
        """Advance to the next curriculum stage"""
        
        current_stages = list(CurriculumStage)
        current_index = current_stages.index(self.current_stage)
        
        if current_index >= len(current_stages) - 1:
            logger.info("📚 Already at final curriculum stage")
            return False
            
        old_stage = self.current_stage
        self.current_stage = current_stages[current_index + 1]
        
        # Reset stage tracking
        self.episodes_in_stage = 0
        self.stage_performance_history = []
        
        # Record transition
        self.stage_transitions.append((episode, self.current_stage))
        
        # Get stage info
        old_config = self.get_stage_config_by_stage(old_stage)
        new_config = self.get_current_stage_config()
        
        logger.info(f"🎓 Curriculum stage transition at episode {episode}:")
        logger.info(f"   {old_stage.value} → {self.current_stage.value}")
        logger.info(f"   {old_config['description']} → {new_config['description']}")
        
        return True
        
    def get_stage_config_by_stage(self, stage: CurriculumStage) -> Dict[str, Any]:
        """Get config for specific stage"""
        temp_stage = self.current_stage
        self.current_stage = stage
        config = self.get_current_stage_config()
        self.current_stage = temp_stage
        return config
        
    def should_reassign_rewards(self, episode: int) -> bool:
        """Check if full reward reassignment should occur"""
        return (self.config.reassignment_frequency > 0 and 
                episode % self.config.reassignment_frequency == 0)
                
    def get_reward_multiplier(self) -> float:
        """Get current reward scaling multiplier"""
        stage_config = self.get_current_stage_config()
        return stage_config.get('reward_scaling', 1.0)
        
    def get_action_mask(self, available_actions: List[str]) -> List[bool]:
        """Get action mask based on current curriculum stage"""
        
        stage_config = self.get_current_stage_config()
        allowed_actions = stage_config.get('allowed_actions', available_actions)
        
        # Create mask for allowed actions
        mask = []
        for action in available_actions:
            mask.append(action in allowed_actions)
            
        return mask
        
    def get_wst_complexity_level(self) -> str:
        """Get WST complexity level for current stage"""
        stage_config = self.get_current_stage_config()
        return stage_config.get('wst_complexity', 'full')
        
    def get_amddp5_config(self) -> Dict[str, Any]:
        """Get AMDDP5 configuration for current stage"""
        stage_config = self.get_current_stage_config()
        
        return {
            'enabled': stage_config.get('amddp5_enabled', True),
            'threshold_pct': stage_config.get('amddp5_threshold', 5.0),
            'progressive': True  # Use progressive AMDDP5 implementation
        }
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get curriculum learning statistics"""
        
        if not self.stage_performance_history:
            avg_performance = 0.0
        else:
            avg_performance = np.mean(self.stage_performance_history)
            
        return {
            'current_stage': self.current_stage.value,
            'episodes_in_stage': self.episodes_in_stage,
            'avg_stage_performance': avg_performance,
            'total_transitions': len(self.stage_transitions),
            'stage_progress': self._get_stage_progress(),
            'transitions_history': [(ep, stage.value) for ep, stage in self.stage_transitions]
        }
        
    def _get_stage_progress(self) -> float:
        """Get progress through current stage (0-1)"""
        if self.config.adaptive_transitions:
            if not self.stage_performance_history:
                return 0.0
                
            recent_performance = np.mean(self.stage_performance_history[-self.min_episodes_per_stage:]) if len(self.stage_performance_history) >= self.min_episodes_per_stage else 0.0
            return min(recent_performance / self.config.performance_threshold, 1.0)
        else:
            return min(self.episodes_in_stage / self.config.stage_transition_episodes, 1.0)
            
    def force_stage_transition(self, target_stage: CurriculumStage, episode: int) -> bool:
        """Force transition to specific stage (for debugging/testing)"""
        
        if target_stage == self.current_stage:
            return False
            
        old_stage = self.current_stage
        self.current_stage = target_stage
        self.episodes_in_stage = 0
        self.stage_performance_history = []
        
        self.stage_transitions.append((episode, target_stage))
        
        logger.info(f"🔧 Forced curriculum transition: {old_stage.value} → {target_stage.value}")
        return True


def create_swt_curriculum_learning(config_dict: dict = None) -> SWTCurriculumLearning:
    """Factory function to create SWT curriculum learning system"""
    
    if config_dict is None:
        config = CurriculumConfig()
    else:
        config = CurriculumConfig(**config_dict)
        
    return SWTCurriculumLearning(config)