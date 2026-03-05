"""
CIFAKENet — Lightweight CNN for real-vs-fake image classification.

Architecture:
    4 convolutional blocks (Conv → BN → ReLU → MaxPool) with channel
    progression 32 → 64 → 128 → 256, followed by adaptive average
    pooling and a 2-layer classifier head.

Designed for 32×32 input resolution (CIFAKE dataset).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        pool: Whether to apply 2×2 max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAKENet(nn.Module):
    """Lightweight 8-layer CNN for CIFAKE binary classification.

    Outputs a single logit (use sigmoid for probability).
    The ``features`` attribute exposes the convolutional backbone for
    Grad-CAM hooks (target the last ConvBlock).

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output classes (1 for binary).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # Convolutional backbone — 4 blocks doubling channels
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16, 3 → 32
            ConvBlock(in_channels, 32),
            # Block 2: 16×16 → 8×8, 32 → 64
            ConvBlock(32, 64),
            # Block 3: 8×8 → 4×4, 64 → 128
            ConvBlock(64, 128),
            # Block 4: 4×4 → 2×2, 128 → 256
            ConvBlock(128, 256),
        )

        # Additional conv layers without pooling for richer features
        self.extra_features = nn.Sequential(
            ConvBlock(256, 256, pool=False),
            ConvBlock(256, 256, pool=False),
        )

        # Classifier head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits of shape (B, 1).
        """
        x = self.features(x)
        x = self.extra_features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization.

        Returns the last convolutional block in the extra_features
        sequential, which has the richest spatial feature maps.
        """
        return self.extra_features[-1].block[0]  # last Conv2d
