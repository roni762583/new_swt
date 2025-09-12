"""
EfficientZero Value-Prefix Network Implementation
Multi-architecture support for forex time series prediction

Implements Transformer, TCN, LSTM, and 1D-CNN architectures for value-prefix prediction
Optimized for SWT-Enhanced Stochastic MuZero trading system

Author: SWT Research Team
Date: September 2025
Adherence: CLAUDE.md professional code standards
"""

from typing import Dict, Any, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network implementation
    Optimized for causal time series processing in forex environments
    """
    
    def __init__(
        self, 
        num_inputs: int, 
        num_channels: list[int], 
        kernel_size: int = 3,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize Temporal Convolutional Network
        
        Args:
            num_inputs: Number of input features
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        if not num_channels:
            raise ValueError("num_channels must be non-empty list")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive integer")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
            
        self.num_levels = len(num_channels)
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal convolution with proper padding
            conv = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size,
                stride=1, 
                padding=(kernel_size-1) * dilation_size, 
                dilation=dilation_size
            )
            
            # Initialize weights
            nn.init.normal_(conv.weight, 0, 0.01)
            
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_channels[-1], sequence_length)
        """
        output = self.network(x)
        
        # Remove future information (causal masking)
        return output[:, :, :-self.network[0].padding[0]]


class SWTValuePrefixNetwork(nn.Module):
    """
    Multi-architecture value-prefix network for EfficientZero integration
    Supports Transformer, TCN, LSTM, and 1D-CNN architectures
    
    Predicts cumulative returns from sequences of predicted rewards and values
    Optimized for forex time series and SWT system integration
    """
    
    SUPPORTED_ARCHITECTURES = ['transformer', 'tcn', 'lstm', 'conv1d']
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        architecture: str = 'transformer',
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        """
        Initialize value-prefix network with specified architecture
        
        Args:
            input_dim: Dimension of input features (reward + value + position)
            hidden_dim: Hidden dimension for neural networks
            architecture: Architecture type ('transformer', 'tcn', 'lstm', 'conv1d')
            num_layers: Number of layers in the architecture
            dropout: Dropout probability for regularization
            **kwargs: Additional architecture-specific parameters
            
        Raises:
            ValueError: If architecture not supported or parameters invalid
        """
        super().__init__()
        
        # Validate parameters
        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Supported: {self.SUPPORTED_ARCHITECTURES}")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize architecture-specific components
        self._build_architecture(**kwargs)
        
        # Common components
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        logger.info(f"Initialized SWTValuePrefixNetwork with {architecture} architecture")
        logger.info(f"Parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, dropout={dropout}")
    
    def _build_architecture(self, **kwargs) -> None:
        """Build architecture-specific components"""
        
        if self.architecture == 'transformer':
            self._build_transformer(**kwargs)
        elif self.architecture == 'tcn':
            self._build_tcn(**kwargs)
        elif self.architecture == 'lstm':
            self._build_lstm(**kwargs)
        elif self.architecture == 'conv1d':
            self._build_conv1d(**kwargs)
    
    def _build_transformer(self, nhead: int = 8, **kwargs) -> None:
        """Build Transformer encoder architecture"""
        
        # Validate transformer-specific parameters
        if self.input_dim % nhead != 0:
            logger.warning(f"input_dim ({self.input_dim}) not divisible by nhead ({nhead}), "
                          f"adjusting nhead to {min(nhead, self.input_dim)}")
            nhead = min(nhead, self.input_dim)
            
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=self.num_layers,
            enable_nested_tensor=False  # For consistent behavior
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def _build_tcn(self, kernel_size: int = 3, **kwargs) -> None:
        """Build Temporal Convolutional Network architecture"""
        
        # TCN channel progression
        num_channels = [self.hidden_dim, self.hidden_dim // 2]
        
        self.tcn = TemporalConvNet(
            num_inputs=self.input_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=self.dropout
        )
        
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _build_lstm(self, **kwargs) -> None:
        """Build LSTM architecture (original EfficientZero choice)"""
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False  # Causal for time series
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _build_conv1d(self, kernel_sizes: Optional[list[int]] = None, **kwargs) -> None:
        """Build 1D CNN architecture for local pattern detection"""
        
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        
        layers = []
        in_channels = self.input_dim
        
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = self.hidden_dim // (2 ** i) if i > 0 else self.hidden_dim
            out_channels = max(out_channels, 16)  # Minimum 16 channels
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
    
    def forward(
        self, 
        reward_sequence: torch.Tensor, 
        value_sequence: Optional[torch.Tensor] = None,
        position_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through value-prefix network
        
        Args:
            reward_sequence: Predicted rewards (batch_size, seq_len, 1)
            value_sequence: Predicted values (batch_size, seq_len, 1), optional
            position_sequence: Position features (batch_size, seq_len, n_pos), optional
            
        Returns:
            tuple: (predicted_return, auxiliary_outputs)
                - predicted_return: (batch_size, 1) predicted cumulative return
                - auxiliary_outputs: Dictionary with intermediate outputs
                
        Raises:
            ValueError: If input shapes are invalid
        """
        batch_size, seq_len = reward_sequence.shape[:2]
        
        # Validate inputs
        if reward_sequence.dim() != 3 or reward_sequence.shape[2] != 1:
            raise ValueError(f"reward_sequence must have shape (B, T, 1), got {reward_sequence.shape}")
        
        # Concatenate input sequences
        input_sequences = [reward_sequence]
        
        if value_sequence is not None:
            if value_sequence.shape != reward_sequence.shape:
                raise ValueError(f"value_sequence shape {value_sequence.shape} "
                               f"doesn't match reward_sequence shape {reward_sequence.shape}")
            input_sequences.append(value_sequence)
            
        if position_sequence is not None:
            if position_sequence.shape[:2] != (batch_size, seq_len):
                raise ValueError(f"position_sequence shape {position_sequence.shape[:2]} "
                               f"doesn't match batch and sequence dimensions ({batch_size}, {seq_len})")
            input_sequences.append(position_sequence)
        
        # Combine input sequences
        input_seq = torch.cat(input_sequences, dim=-1)
        
        # Pad or truncate to expected input dimension
        if input_seq.shape[-1] != self.input_dim:
            if input_seq.shape[-1] < self.input_dim:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, seq_len, 
                    self.input_dim - input_seq.shape[-1],
                    device=input_seq.device, dtype=input_seq.dtype
                )
                input_seq = torch.cat([input_seq, padding], dim=-1)
            else:
                # Truncate
                input_seq = input_seq[..., :self.input_dim]
        
        # Apply layer normalization
        input_seq = self.layer_norm(input_seq)
        input_seq = self.dropout_layer(input_seq)
        
        # Architecture-specific forward pass
        auxiliary_outputs = {}
        
        if self.architecture == 'transformer':
            # Transformer: Global attention across sequence
            transformer_out = self.transformer(input_seq)
            auxiliary_outputs['attention_weights'] = transformer_out  # Full sequence for analysis
            
            # Use final position for prediction
            final_repr = transformer_out[:, -1, :]
            predicted_return = self.output_head(final_repr)
            
        elif self.architecture == 'tcn':
            # TCN: Causal convolutions for time series
            input_transposed = input_seq.transpose(1, 2)  # (B, C, T)
            tcn_out = self.tcn(input_transposed)
            auxiliary_outputs['tcn_features'] = tcn_out
            
            predicted_return = self.output_head(tcn_out)
            
        elif self.architecture == 'lstm':
            # LSTM: Sequential processing
            lstm_out, (hidden, cell) = self.lstm(input_seq)
            auxiliary_outputs['hidden_states'] = lstm_out
            auxiliary_outputs['final_hidden'] = hidden[-1]  # Last layer final hidden
            
            # Use final timestep output
            final_output = lstm_out[:, -1, :]
            predicted_return = self.output_head(final_output)
            
        elif self.architecture == 'conv1d':
            # 1D CNN: Local pattern detection
            input_transposed = input_seq.transpose(1, 2)  # (B, C, T)
            conv_out = self.conv_layers(input_transposed)
            auxiliary_outputs['conv_features'] = conv_out
            
            predicted_return = self.output_head(conv_out)
        
        return predicted_return, auxiliary_outputs
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about current architecture configuration"""
        return {
            'architecture': self.architecture,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def compute_loss(
        self, 
        predicted_return: torch.Tensor, 
        target_return: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute value-prefix loss
        
        Args:
            predicted_return: Model predictions (batch_size, 1)
            target_return: True cumulative returns (batch_size,) or (batch_size, 1)
            reduction: Loss reduction ('mean', 'sum', 'none')
            
        Returns:
            Loss tensor
        """
        # Ensure consistent shapes
        if target_return.dim() == 1:
            target_return = target_return.unsqueeze(1)
        
        if predicted_return.shape != target_return.shape:
            raise ValueError(f"Shape mismatch: predicted {predicted_return.shape}, "
                           f"target {target_return.shape}")
        
        # Mean squared error loss
        loss = F.mse_loss(predicted_return, target_return, reduction=reduction)
        
        return loss


def create_swt_value_prefix_network(
    input_dim: int = 32,
    hidden_dim: int = 64,
    architecture: str = 'transformer',
    **kwargs
) -> SWTValuePrefixNetwork:
    """
    Factory function to create SWT value-prefix network
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        architecture: Network architecture
        **kwargs: Additional architecture-specific parameters
        
    Returns:
        Initialized SWTValuePrefixNetwork
    """
    return SWTValuePrefixNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
        **kwargs
    )


def test_value_prefix_network() -> None:
    """Test function for value-prefix network"""
    
    logger.info("Testing SWTValuePrefixNetwork...")
    
    # Test parameters
    batch_size, seq_len = 16, 10
    input_dim = 32
    
    # Test data
    rewards = torch.randn(batch_size, seq_len, 1)
    values = torch.randn(batch_size, seq_len, 1)
    positions = torch.randn(batch_size, seq_len, 5)
    targets = torch.randn(batch_size, 1)
    
    # Test each architecture
    for arch in SWTValuePrefixNetwork.SUPPORTED_ARCHITECTURES:
        logger.info(f"Testing {arch} architecture...")
        
        network = create_swt_value_prefix_network(
            input_dim=input_dim,
            architecture=arch
        )
        
        # Forward pass
        predicted_return, aux_outputs = network(rewards, values, positions)
        
        # Compute loss
        loss = network.compute_loss(predicted_return, targets)
        
        # Backward pass
        loss.backward()
        
        logger.info(f"✅ {arch}: output_shape={predicted_return.shape}, loss={loss.item():.4f}")
        logger.info(f"   Architecture info: {network.get_architecture_info()}")
    
    logger.info("✅ All architectures tested successfully!")


if __name__ == "__main__":
    test_value_prefix_network()