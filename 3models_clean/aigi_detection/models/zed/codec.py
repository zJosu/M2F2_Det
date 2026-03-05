"""
PixelCNN-based lossless image codec.

Implements a lightweight autoregressive model with masked convolutions
that estimates the conditional probability P(pixel_i | context).
Used by ZED to compute per-pixel NLL and entropy maps.

This implementation uses a simple PixelCNN (not PixelCNN++) for
tractability.  The model can be pre-trained on natural images or
loaded from a checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MaskedConv2d(nn.Conv2d):
    """Masked convolution for autoregressive image modelling.

    Ensures that each pixel only depends on previously scanned pixels
    (raster-scan order: top-to-bottom, left-to-right).

    Args:
        mask_type: 'A' for the first layer (excludes center pixel),
                   'B' for subsequent layers (includes center pixel).
        *args, **kwargs: Forwarded to nn.Conv2d.
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))

        _, _, kH, kW = self.weight.shape
        # Zero out the bottom half
        self.mask[:, :, kH // 2 + 1:, :] = 0
        # Zero out the right half of the center row
        self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
        # For type A: also zero out the center pixel itself
        if mask_type == "A":
            self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlock(nn.Module):
    """Residual block with masked convolutions.

    Args:
        n_filters: Number of filters for the hidden layers.
    """

    def __init__(self, n_filters: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            MaskedConv2d("B", n_filters, n_filters, kernel_size=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", n_filters, n_filters, kernel_size=1),
            nn.BatchNorm2d(n_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PixelCNNEncoder(nn.Module):
    """PixelCNN autoregressive model for image probability estimation.

    Models P(image) = ∏_i P(pixel_i | pixels_<i) using masked
    convolutions.  Outputs a 256-way softmax per channel per pixel
    (discrete pixel values 0-255).

    Args:
        n_channels: Number of input channels (3 for RGB).
        n_filters: Number of hidden filters.
        n_layers: Number of residual blocks.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_filters: int = 64,
        n_layers: int = 7,
    ):
        super().__init__()
        self.n_channels = n_channels

        # Initial masked convolution (type A — excludes center pixel)
        self.input_conv = MaskedConv2d(
            "A", n_channels, n_filters, kernel_size=7, padding=3,
        )
        self.input_bn = nn.BatchNorm2d(n_filters)

        # Residual blocks (type B — includes center pixel)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_filters) for _ in range(n_layers)]
        )

        # Output layers: predict 256 bins per channel per pixel
        self.output_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv2d("B", n_filters, n_filters, kernel_size=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            MaskedConv2d("B", n_filters, n_channels * 256, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Logits of shape (B, C * 256, H, W).
        """
        h = self.input_bn(self.input_conv(x))
        h = self.residual_blocks(h)
        logits = self.output_conv(h)
        return logits

    def compute_nll(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel negative log-likelihood.

        Args:
            x: Input tensor (B, C, H, W) with values in [0, 1].

        Returns:
            NLL map of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        logits = self.forward(x)  # (B, C*256, H, W)

        # Reshape logits to (B, C, 256, H, W)
        logits = logits.view(B, C, 256, H, W)

        # Quantise input to [0, 255] integer targets
        targets = (x * 255).long().clamp(0, 255)  # (B, C, H, W)

        # Compute per-pixel cross-entropy (NLL)
        log_probs = F.log_softmax(logits, dim=2)  # (B, C, 256, H, W)
        nll = -log_probs.gather(
            2, targets.unsqueeze(2)
        ).squeeze(2)  # (B, C, H, W)

        return nll

    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-pixel predictive entropy.

        Args:
            x: Input tensor (B, C, H, W) with values in [0, 1].

        Returns:
            Entropy map of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        logits = self.forward(x)
        logits = logits.view(B, C, 256, H, W)

        # Entropy = -∑ p log p
        probs = F.softmax(logits, dim=2)
        log_probs = F.log_softmax(logits, dim=2)
        entropy = -(probs * log_probs).sum(dim=2)  # (B, C, H, W)

        return entropy

    def compute_nll_and_entropy(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both NLL and entropy in a single forward pass.

        Args:
            x: Input tensor (B, C, H, W) with values in [0, 1].

        Returns:
            Tuple of (nll_map, entropy_map), each (B, C, H, W).
        """
        B, C, H, W = x.shape
        logits = self.forward(x)
        logits = logits.view(B, C, 256, H, W)

        targets = (x * 255).long().clamp(0, 255)

        log_probs = F.log_softmax(logits, dim=2)
        probs = F.softmax(logits, dim=2)

        # NLL
        nll = -log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

        # Entropy
        entropy = -(probs * log_probs).sum(dim=2)

        return nll, entropy
