#!/usr/bin/env python3
"""
TCN (Temporal Convolutional Network) Implementation for Micro MuZero.

Implements dilated causal convolutions for temporal feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding."""
        # Remove future timesteps (causal)
        return self.conv(x)[:, :, :-self.padding] if self.padding > 0 else self.conv(x)


class TemporalBlock(nn.Module):
    """Single temporal block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # First convolution
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual connection
        residual = self.residual(x) if self.residual is not None else x

        return self.relu(out + residual)


class TCNBlock(nn.Module):
    """
    TCN Block for temporal feature extraction.

    Implements stacked temporal blocks with exponentially increasing dilations
    to capture patterns at multiple timescales.
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 48,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4],
        dropout: float = 0.1,
        causal: bool = True
    ):
        """
        Initialize TCN block.

        Args:
            in_channels: Number of input features (15 for micro variant)
            out_channels: Number of output channels (48 for compression)
            kernel_size: Kernel size for convolutions (3)
            dilations: List of dilation factors [1, 2, 4]
            dropout: Dropout rate (0.1)
            causal: Whether to use causal convolutions (True)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations

        # Build temporal blocks
        layers = []
        for i, dilation in enumerate(dilations):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.temporal_blocks = nn.ModuleList(layers)

        # Final projection if needed
        self.final_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.final_bn = nn.BatchNorm1d(out_channels)

        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, dilations)

    def _calculate_receptive_field(
        self,
        kernel_size: int,
        dilations: List[int]
    ) -> int:
        """Calculate the receptive field of the TCN."""
        rf = 1
        for dilation in dilations:
            rf += 2 * (kernel_size - 1) * dilation
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.

        Args:
            x: Input tensor (batch, timesteps=32, features=15)

        Returns:
            Output tensor (batch, channels=48, timesteps=32)
        """
        # Reshape for Conv1d: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.transpose(1, 2)

        # Apply temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # Final projection
        x = self.final_conv(x)
        x = self.final_bn(x)

        return x  # (batch, channels=48, timesteps=32)

    def get_receptive_field(self) -> int:
        """Get the receptive field size."""
        return self.receptive_field