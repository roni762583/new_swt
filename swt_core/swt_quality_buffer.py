#!/usr/bin/env python3
"""
SWT Quality Experience Buffer
Enhanced experience buffer with smart eviction based on V7/V8 QualityExperienceBuffer
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import heapq
import time
import logging
from collections import deque

# JIT compilation for performance-critical functions
from numba import njit

logger = logging.getLogger(__name__)


@njit(fastmath=True, cache=True)
def calculate_quality_score_jit(pip_pnl: float, trade_complete: bool, position_change: bool,
                               done: bool, td_error: float, reward: float, 
                               session_expectancy: float, history_length: int) -> float:
    """
    JIT-compiled quality score calculation for experience prioritization
    
    Called for every experience in buffer (~100K+ times) - critical hot path
    ~25% speedup over Python implementation with bounds checking
    """
    score = 0.0
    
    # Primary: Pip P&L (vectorizable operations)
    if pip_pnl > 0:
        score += pip_pnl * 0.5  # Heavy weight on profitable trades
    else:
        score += pip_pnl * 0.1  # Light penalty for losses
        
    # Trade completion bonus
    if trade_complete:
        if pip_pnl > 0:
            score += 5.0  # Major bonus for profitable completed trades
        else:
            score += 1.0  # Minor bonus for completed losing trades
    
    # Position change and terminal bonuses
    if position_change:
        score += 3.0
    if done:
        score += 1.5
        
    # TD error contribution (bounded)
    score += min(abs(td_error), 5.0) * 0.15
    
    # History length bonus
    if history_length >= 4:
        score += 0.3
        
    # Reward contribution (bounded)
    if reward > 0:
        score += min(reward * 0.1, 2.0)
        
    # Session expectancy bonus (bounded)
    if session_expectancy > 0:
        score += min(session_expectancy * 0.3, 3.0)
        
    return max(score, 0.05)  # Minimum quality score


@dataclass
class SWTExperience:
    """Enhanced experience with quality scoring for SWT system"""
    # Core experience data
    observation_history: torch.Tensor  # (T, 128) WST-processed history
    market_prices: torch.Tensor        # (256,) raw price series for WST
    position_features: torch.Tensor    # (9,) V7/V8-compatible position features
    action: int
    reward: float
    value_target: float
    policy_target: np.ndarray
    next_observation: Dict[str, torch.Tensor]
    done: bool
    
    # Quality metrics (automatically calculated)
    timestamp: float = field(default_factory=time.time)
    quality_score: float = field(default=0.0)
    td_error: float = field(default=0.0)
    pip_pnl: float = field(default=0.0)  # Direct pip P&L for trading quality
    position_change: bool = field(default=False)
    trade_complete: bool = field(default=False)  # True when trade is fully closed
    session_expectancy: float = field(default=0.0)  # Session expectancy for context
    
    def __post_init__(self):
        """Calculate quality score based on experience characteristics"""
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate trading-focused quality score using JIT-compiled hot path
        Prioritizes profitable trading experiences over losses
        """
        # Use JIT-compiled quality calculation (25% faster)
        return calculate_quality_score_jit(
            self.pip_pnl, self.trade_complete, self.position_change,
            self.done, self.td_error, self.reward, 
            self.session_expectancy, self.observation_history.shape[0]
        )


class SWTQualityExperienceBuffer:
    """
    Quality-based experience buffer with smart eviction
    Based on V7/V8 QualityExperienceBuffer but enhanced for WST system
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 eviction_batch: int = 2000,
                 min_quality_threshold: float = 0.1,
                 diversity_sampling: bool = True):
        """
        Initialize quality experience buffer
        
        Args:
            capacity: Maximum buffer size
            eviction_batch: Number of experiences to evict when full
            min_quality_threshold: Minimum quality score to keep
            diversity_sampling: Whether to maintain experience diversity
        """
        self.capacity = capacity
        self.eviction_batch = eviction_batch
        self.min_quality_threshold = min_quality_threshold
        self.diversity_sampling = diversity_sampling
        
        # Core buffer storage
        self.buffer: List[SWTExperience] = []
        self.quality_heap: List[Tuple[float, int]] = []  # (quality, index) min-heap
        
        # Statistics tracking
        self.total_added = 0
        self.total_evicted = 0
        self.eviction_count = 0
        
        # Diversity tracking (for diverse sampling)
        self.action_counts = [0, 0, 0, 0]  # Hold, Buy, Sell, Close
        self.reward_buckets = {'positive': 0, 'negative': 0, 'zero': 0}
        
        logger.info(f"🗃️ SWT Quality Experience Buffer initialized")
        logger.info(f"   Capacity: {capacity:,}")
        logger.info(f"   Eviction batch: {eviction_batch}")
        logger.info(f"   Quality threshold: {min_quality_threshold}")
        
    def add(self, experience: SWTExperience) -> None:
        """
        Add experience to buffer with quality-based management
        
        Args:
            experience: Experience to add
        """
        # Update experience metadata
        experience.timestamp = time.time()
        if experience.quality_score <= 0:
            experience.quality_score = experience._calculate_quality_score()
            
        # Add to buffer
        self.buffer.append(experience)
        self.total_added += 1
        
        # Update diversity tracking
        self._update_diversity_stats(experience)
        
        # Add to quality heap
        heapq.heappush(self.quality_heap, (experience.quality_score, len(self.buffer) - 1))
        
        # Check if eviction needed
        if len(self.buffer) > self.capacity:
            self._evict_low_quality()
            
    def sample(self, batch_size: int, prioritized: bool = True) -> List[SWTExperience]:
        """
        Sample experiences from buffer
        
        Args:
            batch_size: Number of experiences to sample
            prioritized: Whether to use prioritized sampling
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []
            
        if len(self.buffer) <= batch_size:
            return self.buffer.copy()
            
        if prioritized:
            return self._prioritized_sample(batch_size)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
            
    def _prioritized_sample(self, batch_size: int) -> List[SWTExperience]:
        """Sample experiences based on quality scores"""
        
        # Calculate sampling probabilities based on quality scores
        qualities = np.array([exp.quality_score for exp in self.buffer])
        
        # Add small epsilon to avoid zero probabilities
        qualities = qualities + 1e-6
        
        # Convert to probabilities (higher quality = higher probability)
        probabilities = qualities / qualities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probabilities
        )
        
        return [self.buffer[i] for i in indices]
        
    def _evict_low_quality(self) -> None:
        """Evict lowest quality experiences to make room"""
        
        if len(self.buffer) <= self.capacity:
            return
            
        # Calculate experiences to remove
        target_size = self.capacity - self.eviction_batch
        experiences_to_remove = len(self.buffer) - target_size
        
        # Get quality scores for sorting
        quality_indexed = [(exp.quality_score, i) for i, exp in enumerate(self.buffer)]
        quality_indexed.sort(key=lambda x: x[0])  # Sort by quality (lowest first)
        
        # Identify indices to remove (lowest quality)
        indices_to_remove = set()
        
        # Always remove experiences below quality threshold
        for quality, idx in quality_indexed:
            if quality < self.min_quality_threshold:
                indices_to_remove.add(idx)
                
        # Remove additional low-quality experiences if needed
        remaining_to_remove = experiences_to_remove - len(indices_to_remove)
        if remaining_to_remove > 0:
            for quality, idx in quality_indexed:
                if len(indices_to_remove) >= experiences_to_remove:
                    break
                if idx not in indices_to_remove:
                    indices_to_remove.add(idx)
                    
        # Perform eviction (keep high-quality experiences)
        new_buffer = []
        evicted_count = 0
        
        for i, exp in enumerate(self.buffer):
            if i not in indices_to_remove:
                new_buffer.append(exp)
            else:
                evicted_count += 1
                
        self.buffer = new_buffer
        self.total_evicted += evicted_count
        self.eviction_count += 1
        
        # Rebuild quality heap
        self.quality_heap = []
        for i, exp in enumerate(self.buffer):
            heapq.heappush(self.quality_heap, (exp.quality_score, i))
            
        logger.info(f"🧹 Buffer eviction #{self.eviction_count}:")
        logger.info(f"   Removed {evicted_count} low-quality experiences")
        logger.info(f"   Buffer size: {len(self.buffer):,}")
        logger.info(f"   Avg quality: {np.mean([e.quality_score for e in self.buffer]):.3f}")
        
    def _update_diversity_stats(self, experience: SWTExperience) -> None:
        """Update diversity tracking statistics"""
        
        # Update action counts
        if 0 <= experience.action < len(self.action_counts):
            self.action_counts[experience.action] += 1
            
        # Update reward buckets
        if experience.reward > 0.1:
            self.reward_buckets['positive'] += 1
        elif experience.reward < -0.1:
            self.reward_buckets['negative'] += 1
        else:
            self.reward_buckets['zero'] += 1
            
    def update_td_errors(self, experiences: List[SWTExperience], td_errors: List[float]) -> None:
        """Update TD errors for experiences (for quality recalculation)"""
        
        for exp, td_error in zip(experiences, td_errors):
            exp.td_error = abs(td_error)
            exp.quality_score = exp._calculate_quality_score()
            
    def get_statistics(self) -> Dict[str, any]:
        """Get buffer statistics"""
        
        if not self.buffer:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_added': self.total_added,
                'total_evicted': self.total_evicted
            }
            
        qualities = [exp.quality_score for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'total_evicted': self.total_evicted,
            'eviction_count': self.eviction_count,
            'avg_quality': np.mean(qualities),
            'min_quality': np.min(qualities),
            'max_quality': np.max(qualities),
            'action_distribution': self.action_counts.copy(),
            'reward_distribution': self.reward_buckets.copy(),
            'quality_percentiles': {
                'p25': np.percentile(qualities, 25),
                'p50': np.percentile(qualities, 50),
                'p75': np.percentile(qualities, 75),
                'p90': np.percentile(qualities, 90)
            }
        }
        
    def clear(self) -> None:
        """Clear all experiences from buffer"""
        self.buffer.clear()
        self.quality_heap.clear()
        self.action_counts = [0, 0, 0, 0]
        self.reward_buckets = {'positive': 0, 'negative': 0, 'zero': 0}
        
        logger.info("🧹 Quality experience buffer cleared")
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def is_ready(self, min_experiences: int = 1000) -> bool:
        """Check if buffer has enough experiences for training"""
        return len(self.buffer) >= min_experiences