"""
WST Transform Implementation
Wavelet Scattering Transform for market feature extraction

Provides consistent WST processing for both training and live trading,
with caching and optimization for production use.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass

from swt_core.types import MarketState
from swt_core.exceptions import FeatureProcessingError

logger = logging.getLogger(__name__)


@dataclass
class WSTConfig:
    """WST processor configuration"""
    J: int = 2                          # Number of scales
    Q: int = 6                          # Number of orientations per octave
    backend: str = "kymatio"           # "kymatio" or "pytorch-wavelets"
    output_dim: int = 128              # Expected output dimension
    max_order: int = 2                 # Maximum scattering order
    normalize: bool = True             # Apply normalization
    cache_enabled: bool = True         # Enable transform caching


class WSTProcessor:
    """
    Wavelet Scattering Transform processor for market data
    
    Extracts multi-scale, translation-invariant features from price series
    using wavelet scattering transforms. Provides consistent output for
    both training and live trading systems.
    """
    
    def __init__(self, config: WSTConfig, device: Optional[torch.device] = None):
        """
        Initialize WST processor
        
        Args:
            config: WST configuration parameters
            device: Torch device for computations
        """
        self.config = config
        self.device = device or torch.device("cpu")
        
        # Initialize WST backend
        self._scattering_transform = None
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize transform
        self._initialize_transform()
        
        logger.info(f"🌊 WST Processor initialized: J={config.J}, Q={config.Q}, "
                   f"backend={config.backend}, device={self.device}")
    
    def _initialize_transform(self) -> None:
        """Initialize the wavelet scattering transform"""
        try:
            if self.config.backend == "kymatio":
                self._initialize_kymatio()
            elif self.config.backend == "pytorch-wavelets":
                self._initialize_pytorch_wavelets()
            elif self.config.backend == "manual":
                self._initialize_manual()
            else:
                raise FeatureProcessingError(
                    f"Unsupported WST backend: {self.config.backend}",
                    context={"supported_backends": ["kymatio", "pytorch-wavelets", "manual"]}
                )
                
        except ImportError as e:
            # Fallback to manual implementation
            logger.warning(f"⚠️ WST backend {self.config.backend} not available: {e}")
            logger.info("🔄 Falling back to manual WST implementation")
            self._initialize_manual()
    
    def _initialize_kymatio(self) -> None:
        """Initialize Kymatio scattering transform"""
        try:
            from kymatio.torch import Scattering1D
            
            self._scattering_transform = Scattering1D(
                J=self.config.J,
                shape=(256,),  # Input length
                Q=self.config.Q,
                max_order=self.config.max_order,
                frontend='torch'
            ).to(self.device)
            
            logger.info("✅ Kymatio WST backend initialized successfully")
            
        except ImportError:
            # Don't raise error - let the fallback handle it
            raise ImportError("Kymatio not installed")
    
    def _initialize_pytorch_wavelets(self) -> None:
        """Initialize PyTorch Wavelets implementation"""
        try:
            import pywt
            
            # Manual implementation using PyWavelets
            self._wavelet_filters = self._create_wavelet_filters()
            logger.info("✅ PyTorch Wavelets backend initialized successfully")
            
        except ImportError:
            # Don't raise error - let the fallback handle it
            raise ImportError("PyWavelets not installed")
    
    def _initialize_manual(self) -> None:
        """Initialize manual WST implementation"""
        # Create manual wavelet filters
        self._wavelet_filters = self._create_wavelet_filters()
        logger.info("✅ Manual WST backend initialized successfully")
    
    def _create_wavelet_filters(self) -> Dict[str, torch.Tensor]:
        """Create wavelet filter bank"""
        filters = {}
        
        # Create Morlet wavelets at different scales and orientations
        for j in range(self.config.J):
            for q in range(self.config.Q):
                # Scale and orientation parameters
                scale = 2 ** j
                xi = (q + 0.5) * np.pi / self.config.Q
                
                # Create Morlet wavelet
                filter_size = min(256, 4 * scale)
                t = torch.linspace(-filter_size//2, filter_size//2, filter_size, dtype=torch.float32)
                
                # Morlet wavelet formula
                sigma = scale / 4.0
                morlet_real = torch.exp(-(t ** 2) / (2 * sigma ** 2)) * torch.cos(xi * t)
                morlet_imag = torch.exp(-(t ** 2) / (2 * sigma ** 2)) * torch.sin(xi * t)
                
                filters[f"real_{j}_{q}"] = morlet_real.to(self.device)
                filters[f"imag_{j}_{q}"] = morlet_imag.to(self.device)
        
        return filters
    
    def _create_manual_filters(self) -> Dict[str, torch.Tensor]:
        """Create manual wavelet filters for fallback implementation"""
        return self._create_wavelet_filters()
    
    def transform(self, price_series: Union[np.ndarray, torch.Tensor],
                 cache_key: Optional[str] = None) -> np.ndarray:
        """
        Apply WST transform to price series
        
        Args:
            price_series: Price series of shape (256,)
            cache_key: Optional cache key for repeated transforms
            
        Returns:
            WST features of shape (128,)
        """
        try:
            # Input validation
            if isinstance(price_series, np.ndarray):
                price_tensor = torch.from_numpy(price_series).float()
            else:
                price_tensor = price_series.float()
                
            if price_tensor.shape[-1] != 256:
                raise FeatureProcessingError(
                    f"Expected price series length 256, got {price_tensor.shape[-1]}"
                )
            
            # Check cache first
            if cache_key and self.config.cache_enabled:
                cached_result = self._get_cached_transform(cache_key)
                if cached_result is not None:
                    self._cache_hits += 1
                    return cached_result
                else:
                    self._cache_misses += 1
            
            # Ensure tensor is on correct device and has batch dimension
            price_tensor = price_tensor.to(self.device)
            if price_tensor.dim() == 1:
                price_tensor = price_tensor.unsqueeze(0)  # Add batch dimension
            
            # Apply WST transform
            if self._scattering_transform is not None:
                # Kymatio backend
                wst_output = self._scattering_transform(price_tensor)
                wst_features = wst_output.squeeze(0)  # Remove batch dimension
            else:
                # Manual backend
                wst_features = self._manual_transform(price_tensor.squeeze(0))
            
            # Post-processing
            wst_features = self._postprocess_features(wst_features)
            
            # Convert to numpy
            wst_numpy = wst_features.detach().cpu().numpy().astype(np.float32)
            
            # Cache result
            if cache_key and self.config.cache_enabled:
                self._cache_transform(cache_key, wst_numpy)
            
            # Validate output
            self._validate_output(wst_numpy)
            
            return wst_numpy
            
        except Exception as e:
            raise FeatureProcessingError(
                f"WST transform failed: {str(e)}",
                context={
                    "input_shape": price_series.shape if hasattr(price_series, 'shape') else None,
                    "backend": self.config.backend
                },
                original_error=e
            )
    
    def _manual_transform(self, price_tensor: torch.Tensor) -> torch.Tensor:
        """Manual WST implementation using convolutions"""
        # Normalize input
        price_normalized = (price_tensor - price_tensor.mean()) / (price_tensor.std() + 1e-8)
        
        scattering_coefficients = []
        
        # Zero-th order (low-pass filter)
        lowpass = F.avg_pool1d(
            price_normalized.unsqueeze(0).unsqueeze(0), 
            kernel_size=8, stride=1, padding=4
        ).squeeze()
        scattering_coefficients.append(lowpass[::8].mean())  # Subsampling and averaging
        
        # First and second order scattering
        for j in range(self.config.J):
            for q in range(self.config.Q):
                # Get wavelet filters
                real_filter = self._wavelet_filters.get(f"real_{j}_{q}")
                imag_filter = self._wavelet_filters.get(f"imag_{j}_{q}")
                
                if real_filter is None or imag_filter is None:
                    continue
                
                # Pad filters to match input size
                filter_size = real_filter.shape[0]
                pad_size = (256 - filter_size) // 2
                
                if pad_size > 0:
                    real_filter = F.pad(real_filter, (pad_size, 256 - filter_size - pad_size))
                    imag_filter = F.pad(imag_filter, (pad_size, 256 - filter_size - pad_size))
                
                # Convolution with wavelets
                real_conv = F.conv1d(
                    price_normalized.unsqueeze(0).unsqueeze(0),
                    real_filter.unsqueeze(0).unsqueeze(0),
                    padding='same'
                ).squeeze()
                
                imag_conv = F.conv1d(
                    price_normalized.unsqueeze(0).unsqueeze(0), 
                    imag_filter.unsqueeze(0).unsqueeze(0),
                    padding='same'
                ).squeeze()
                
                # Complex modulus (first-order scattering)
                first_order = torch.sqrt(real_conv ** 2 + imag_conv ** 2)
                
                # Subsampling and averaging
                scale_factor = 2 ** (j + 1)
                if len(first_order) > scale_factor:
                    first_order_coeff = first_order[::scale_factor].mean()
                    scattering_coefficients.append(first_order_coeff)
                
                # Second-order scattering (optional, for richer features)
                if self.config.max_order >= 2 and j < self.config.J - 1:
                    # Apply another wavelet transform to first-order coefficients
                    for q2 in range(min(self.config.Q, 3)):  # Limit second-order complexity
                        real_filter_2 = self._wavelet_filters.get(f"real_{j+1}_{q2}")
                        imag_filter_2 = self._wavelet_filters.get(f"imag_{j+1}_{q2}")
                        
                        if real_filter_2 is not None and imag_filter_2 is not None:
                            # Simplified second-order computation
                            second_order = F.conv1d(
                                first_order.unsqueeze(0).unsqueeze(0),
                                real_filter_2[:min(len(first_order), len(real_filter_2))].unsqueeze(0).unsqueeze(0),
                                padding='same'
                            ).squeeze()
                            
                            if len(second_order) > 0:
                                second_order_coeff = torch.abs(second_order).mean()
                                scattering_coefficients.append(second_order_coeff)
        
        # Stack coefficients and pad/truncate to target dimension
        if scattering_coefficients:
            wst_features = torch.stack(scattering_coefficients)
            
            # Pad or truncate to target dimension
            current_dim = wst_features.shape[0]
            if current_dim < self.config.output_dim:
                # Pad with zeros
                padding = torch.zeros(self.config.output_dim - current_dim, device=self.device)
                wst_features = torch.cat([wst_features, padding])
            elif current_dim > self.config.output_dim:
                # Truncate
                wst_features = wst_features[:self.config.output_dim]
        else:
            # Fallback: return zero features
            wst_features = torch.zeros(self.config.output_dim, device=self.device)
        
        return wst_features
    
    def _postprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        """Post-process WST features"""
        # Ensure correct output dimension
        if features.numel() != self.config.output_dim:
            if features.numel() > self.config.output_dim:
                # Truncate
                features = features.flatten()[:self.config.output_dim]
            else:
                # Pad with zeros
                padding_size = self.config.output_dim - features.numel()
                features = F.pad(features.flatten(), (0, padding_size))
        else:
            features = features.flatten()
        
        # Normalization
        if self.config.normalize:
            # L2 normalization
            features = F.normalize(features, p=2, dim=0)
        
        # Handle NaN/Inf values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def _get_cached_transform(self, cache_key: str) -> Optional[np.ndarray]:
        """Get cached transform result"""
        return self._cache.get(cache_key)
    
    def _cache_transform(self, cache_key: str, result: np.ndarray) -> None:
        """Cache transform result"""
        if len(self._cache) >= 10000:  # Limit cache size
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result
    
    def _validate_output(self, output: np.ndarray) -> None:
        """Validate WST output"""
        if output.shape != (self.config.output_dim,):
            raise FeatureProcessingError(
                f"WST output shape mismatch: expected ({self.config.output_dim},), got {output.shape}"
            )
        
        if np.isnan(output).any():
            raise FeatureProcessingError("WST output contains NaN values")
        
        if np.isinf(output).any():
            raise FeatureProcessingError("WST output contains infinite values")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self) -> None:
        """Clear transform cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("🗑️ WST transform cache cleared")
    
    def save_cache(self, filepath: Union[str, Path]) -> None:
        """Save cache to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"💾 WST cache saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save WST cache: {e}")
    
    def load_cache(self, filepath: Union[str, Path]) -> None:
        """Load cache from disk"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"📁 WST cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load WST cache: {e}")
            self._cache = {}