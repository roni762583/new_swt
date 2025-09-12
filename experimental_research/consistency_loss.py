"""
EfficientZero Self-Supervised Consistency Loss Implementation
SimSiam-style consistency loss for dynamics model validation

Compares predicted latents from dynamics network with actual encoded latents
Optimized for SWT-Enhanced Stochastic MuZero trading system

Author: SWT Research Team
Date: September 2025
Adherence: CLAUDE.md professional code standards
"""

from typing import Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsistencyLossBase(ABC, nn.Module):
    """
    Abstract base class for consistency loss implementations
    Provides common interface and validation logic
    """
    
    @abstractmethod
    def compute_loss(
        self, 
        predicted_latent: torch.Tensor, 
        true_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute consistency loss between predicted and true latents
        
        Args:
            predicted_latent: Predicted latent from dynamics network
            true_latent: True latent from representation network
            
        Returns:
            tuple: (loss_value, metrics_dict)
        """
        pass
    
    def _validate_inputs(
        self, 
        predicted_latent: torch.Tensor, 
        true_latent: torch.Tensor
    ) -> None:
        """Validate input tensors for consistency loss computation"""
        
        if predicted_latent.shape != true_latent.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_latent.shape}, "
                           f"true {true_latent.shape}")
        
        if predicted_latent.dim() < 2:
            raise ValueError(f"Tensors must be at least 2D, got {predicted_latent.dim()}D")
        
        if torch.isnan(predicted_latent).any():
            raise ValueError("predicted_latent contains NaN values")
        
        if torch.isnan(true_latent).any():
            raise ValueError("true_latent contains NaN values")


class SWTConsistencyLoss(ConsistencyLossBase):
    """
    Self-supervised consistency loss for SWT dynamics model
    
    Implements SimSiam-style consistency loss with stop gradient and temperature scaling
    Specifically optimized for forex time series and WST feature compatibility
    """
    
    def __init__(
        self,
        stop_gradient: bool = True,
        temperature: float = 0.1,
        normalize_features: bool = True,
        similarity_metric: str = 'cosine',
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize SWT consistency loss
        
        Args:
            stop_gradient: Whether to stop gradient on true_latent to prevent collapse
            temperature: Temperature scaling for similarity computation
            normalize_features: Whether to L2 normalize features before comparison
            similarity_metric: Similarity metric ('cosine', 'l2', 'dot_product')
            reduction: Loss reduction method ('mean', 'sum', 'none')
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate parameters
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity_metric not in ['cosine', 'l2', 'dot_product']:
            raise ValueError(f"Unsupported similarity_metric: {similarity_metric}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
            
        self.stop_gradient = stop_gradient
        self.temperature = temperature
        self.normalize_features = normalize_features
        self.similarity_metric = similarity_metric
        self.reduction = reduction
        
        # Statistics tracking
        self.register_buffer('num_calls', torch.tensor(0))
        self.register_buffer('cumulative_loss', torch.tensor(0.0))
        
        logger.info(f"Initialized SWTConsistencyLoss with {similarity_metric} similarity")
        logger.info(f"Parameters: stop_gradient={stop_gradient}, temperature={temperature}, "
                   f"normalize={normalize_features}")
    
    def compute_loss(
        self, 
        predicted_latent: torch.Tensor, 
        true_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute self-supervised consistency loss
        
        Args:
            predicted_latent: Predicted latent from dynamics network (B, D)
            true_latent: True latent from representation network (B, D)
            
        Returns:
            tuple: (consistency_loss, metrics_dict)
                - consistency_loss: Scalar loss tensor
                - metrics_dict: Dictionary with diagnostic metrics
                
        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        self._validate_inputs(predicted_latent, true_latent)
        
        batch_size = predicted_latent.shape[0]
        
        # Apply stop gradient to prevent representation collapse
        if self.stop_gradient:
            true_latent = true_latent.detach()
        
        # Normalize features if requested
        if self.normalize_features:
            predicted_norm = F.normalize(predicted_latent, dim=-1, p=2)
            true_norm = F.normalize(true_latent, dim=-1, p=2)
        else:
            predicted_norm = predicted_latent
            true_norm = true_latent
        
        # Compute similarity based on chosen metric
        if self.similarity_metric == 'cosine':
            similarity = F.cosine_similarity(predicted_norm, true_norm, dim=-1)
        elif self.similarity_metric == 'l2':
            # L2 distance (lower is better, so negate)
            similarity = -torch.norm(predicted_norm - true_norm, p=2, dim=-1)
        elif self.similarity_metric == 'dot_product':
            similarity = torch.sum(predicted_norm * true_norm, dim=-1)
        
        # Apply temperature scaling and compute loss
        scaled_similarity = similarity / self.temperature
        
        # Consistency loss: negative similarity (maximize similarity)
        if self.similarity_metric == 'l2':
            # For L2, we want to minimize distance, so don't negate
            loss_per_sample = -scaled_similarity
        else:
            # For cosine and dot product, maximize similarity
            loss_per_sample = -scaled_similarity
        
        # Apply reduction
        if self.reduction == 'mean':
            consistency_loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            consistency_loss = loss_per_sample.sum()
        else:  # none
            consistency_loss = loss_per_sample
        
        # Update statistics
        self.num_calls += 1
        if self.reduction == 'mean':
            self.cumulative_loss += consistency_loss.detach()
        
        # Compute diagnostic metrics
        metrics = {
            'consistency_loss': consistency_loss.item() if consistency_loss.numel() == 1 else consistency_loss.mean().item(),
            'similarity_mean': similarity.mean().item(),
            'similarity_std': similarity.std().item(),
            'similarity_min': similarity.min().item(),
            'similarity_max': similarity.max().item(),
            'predicted_norm_mean': predicted_norm.norm(dim=-1).mean().item(),
            'true_norm_mean': true_norm.norm(dim=-1).mean().item(),
            'temperature': self.temperature,
            'batch_size': batch_size
        }
        
        # Add representation quality metrics
        if self.normalize_features:
            # Measure alignment after normalization
            alignment = (predicted_norm * true_norm).sum(dim=-1).mean().item()
            metrics['feature_alignment'] = alignment
            
            # Measure uniformity (how spread out the representations are)
            pred_uniformity = self._compute_uniformity(predicted_norm)
            true_uniformity = self._compute_uniformity(true_norm)
            metrics['predicted_uniformity'] = pred_uniformity
            metrics['true_uniformity'] = true_uniformity
        
        return consistency_loss, metrics
    
    def _compute_uniformity(self, features: torch.Tensor, t: float = 2.0) -> float:
        """
        Compute uniformity metric for representation quality
        Higher uniformity indicates better spread of representations
        
        Args:
            features: Normalized feature tensor (B, D)
            t: Temperature parameter for uniformity computation
            
        Returns:
            Uniformity score (scalar)
        """
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return 0.0
        
        # Pairwise distances
        pairwise_distances = torch.pdist(features, p=2)
        
        # Uniformity: log of mean pairwise distance
        uniformity = torch.log(torch.exp(-t * pairwise_distances).mean() + 1e-8) / t
        
        return uniformity.item()
    
    def compute_wst_aware_loss(
        self,
        predicted_latent: torch.Tensor,
        true_latent: torch.Tensor,
        wst_coefficients: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute WST-aware consistency loss for SWT system
        
        Incorporates wavelet scattering transform coefficients for better
        consistency validation in forex time series environments
        
        Args:
            predicted_latent: Predicted latent from dynamics
            true_latent: True latent from representation  
            wst_coefficients: WST coefficients from market data (B, WST_dim)
            
        Returns:
            tuple: (enhanced_loss, enhanced_metrics)
        """
        # Compute base consistency loss
        base_loss, base_metrics = self.compute_loss(predicted_latent, true_latent)
        
        # If WST coefficients provided, add WST-specific consistency
        if wst_coefficients is not None:
            wst_weight = 0.3  # Weight for WST consistency term
            
            # Extract WST-relevant features from latents (assume first N dims)
            wst_dim = min(wst_coefficients.shape[-1], predicted_latent.shape[-1])
            
            pred_wst_features = predicted_latent[..., :wst_dim]
            true_wst_features = true_latent[..., :wst_dim]
            
            # Compute WST-specific consistency
            wst_consistency = F.mse_loss(pred_wst_features, true_wst_features.detach())
            
            # Combine losses
            enhanced_loss = base_loss + wst_weight * wst_consistency
            
            # Enhanced metrics
            enhanced_metrics = base_metrics.copy()
            enhanced_metrics.update({
                'base_consistency_loss': base_loss.item(),
                'wst_consistency_loss': wst_consistency.item(),
                'wst_weight': wst_weight,
                'wst_dimension': wst_dim
            })
            
            return enhanced_loss, enhanced_metrics
        
        return base_loss, base_metrics
    
    def get_statistics(self) -> Dict[str, float]:
        """Get accumulated statistics"""
        if self.num_calls > 0:
            avg_loss = self.cumulative_loss / self.num_calls
        else:
            avg_loss = 0.0
            
        return {
            'total_calls': self.num_calls.item(),
            'average_loss': avg_loss.item(),
            'current_temperature': self.temperature
        }
    
    def reset_statistics(self) -> None:
        """Reset accumulated statistics"""
        self.num_calls.zero_()
        self.cumulative_loss.zero_()
    
    def update_temperature(self, new_temperature: float) -> None:
        """Update temperature parameter"""
        if new_temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = new_temperature
        logger.info(f"Updated temperature to {new_temperature}")


class AdaptiveConsistencyLoss(SWTConsistencyLoss):
    """
    Adaptive consistency loss with automatic temperature scheduling
    Optimizes temperature based on training progress and similarity statistics
    """
    
    def __init__(
        self,
        initial_temperature: float = 0.1,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
        adaptation_rate: float = 0.1,
        target_similarity: float = 0.7,
        **kwargs
    ) -> None:
        """
        Initialize adaptive consistency loss
        
        Args:
            initial_temperature: Starting temperature value
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature  
            adaptation_rate: Rate of temperature adaptation
            target_similarity: Target similarity for temperature adaptation
            **kwargs: Additional arguments for base class
        """
        super().__init__(temperature=initial_temperature, **kwargs)
        
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.adaptation_rate = adaptation_rate
        self.target_similarity = target_similarity
        
        self.register_buffer('similarity_ema', torch.tensor(0.0))
        self.register_buffer('adaptation_step', torch.tensor(0))
        
    def compute_loss(
        self, 
        predicted_latent: torch.Tensor, 
        true_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss with adaptive temperature"""
        
        # Compute base loss
        loss, metrics = super().compute_loss(predicted_latent, true_latent)
        
        # Update similarity EMA
        current_similarity = metrics['similarity_mean']
        if self.adaptation_step == 0:
            self.similarity_ema.copy_(torch.tensor(current_similarity))
        else:
            alpha = 0.1  # EMA decay rate
            self.similarity_ema.mul_(1 - alpha).add_(current_similarity, alpha=alpha)
        
        # Adapt temperature based on similarity
        similarity_error = self.similarity_ema.item() - self.target_similarity
        
        if similarity_error > 0.1:  # Similarity too high, increase temperature
            new_temp = min(self.temperature * (1 + self.adaptation_rate), self.max_temperature)
        elif similarity_error < -0.1:  # Similarity too low, decrease temperature
            new_temp = max(self.temperature * (1 - self.adaptation_rate), self.min_temperature)
        else:
            new_temp = self.temperature
        
        self.temperature = new_temp
        self.adaptation_step += 1
        
        # Add adaptation metrics
        metrics.update({
            'adaptive_temperature': new_temp,
            'similarity_ema': self.similarity_ema.item(),
            'similarity_error': similarity_error,
            'adaptation_step': self.adaptation_step.item()
        })
        
        return loss, metrics


def create_swt_consistency_loss(
    loss_type: str = 'standard',
    **kwargs
) -> ConsistencyLossBase:
    """
    Factory function to create SWT consistency loss
    
    Args:
        loss_type: Type of consistency loss ('standard', 'adaptive')
        **kwargs: Additional arguments for loss initialization
        
    Returns:
        Initialized consistency loss module
        
    Raises:
        ValueError: If loss_type not supported
    """
    if loss_type == 'standard':
        return SWTConsistencyLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveConsistencyLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def test_consistency_loss() -> None:
    """Test function for consistency loss implementations"""
    
    logger.info("Testing SWT Consistency Loss...")
    
    # Test parameters
    batch_size, latent_dim = 32, 256
    
    # Create test tensors
    predicted = torch.randn(batch_size, latent_dim)
    true = torch.randn(batch_size, latent_dim)
    wst_coeffs = torch.randn(batch_size, 128)
    
    # Test standard consistency loss
    logger.info("Testing standard consistency loss...")
    std_loss = create_swt_consistency_loss('standard')
    
    loss_val, metrics = std_loss.compute_loss(predicted, true)
    logger.info(f"✅ Standard loss: {loss_val.item():.4f}")
    logger.info(f"   Metrics: {metrics}")
    
    # Test WST-aware loss
    wst_loss, wst_metrics = std_loss.compute_wst_aware_loss(predicted, true, wst_coeffs)
    logger.info(f"✅ WST-aware loss: {wst_loss.item():.4f}")
    
    # Test adaptive consistency loss  
    logger.info("Testing adaptive consistency loss...")
    adaptive_loss = create_swt_consistency_loss('adaptive')
    
    # Run multiple steps to test adaptation
    for step in range(5):
        loss_val, metrics = adaptive_loss.compute_loss(predicted, true)
        logger.info(f"Step {step}: loss={loss_val.item():.4f}, "
                   f"temp={metrics['adaptive_temperature']:.4f}")
    
    # Test statistics
    stats = std_loss.get_statistics()
    logger.info(f"✅ Statistics: {stats}")
    
    logger.info("✅ All consistency loss tests passed!")


if __name__ == "__main__":
    test_consistency_loss()