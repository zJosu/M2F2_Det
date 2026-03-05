"""
Patch Artifact Expert — Low-level artifact detection via DCT analysis.

Extracts patches from an image, applies Discrete Cosine Transform (DCT)
to capture frequency-domain features, and aggregates them through
learnable convolution layers.  Detects noise patterns, anti-aliasing
artefacts, and other pixel-level signatures of AI generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTLayer(nn.Module):
    """Fixed (non-learnable) 2-D DCT basis for an N×N patch.

    Computes the Type-II DCT by storing the precomputed basis matrix and
    applying it via matrix multiplication, which is differentiable and
    efficient on GPU.

    Args:
        patch_size: Spatial dimension of the (square) patch.
    """

    def __init__(self, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        # Build 1-D DCT-II basis matrix (N×N)
        basis = self._dct_matrix(patch_size)
        # Register as buffer (not a parameter — no gradient)
        self.register_buffer("basis", basis)  # (N, N)

    @staticmethod
    def _dct_matrix(n: int) -> torch.Tensor:
        """Compute the N×N DCT-II basis matrix."""
        mat = torch.zeros(n, n)
        for k in range(n):
            for i in range(n):
                if k == 0:
                    mat[k, i] = 1.0 / (n ** 0.5)
                else:
                    mat[k, i] = (2.0 / n) ** 0.5 * torch.cos(
                        torch.tensor((2 * i + 1) * k * 3.141592653589793 / (2 * n))
                    )
        return mat

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply 2-D DCT to a batch of patches.

        Args:
            patches: Tensor of shape (B, C, N, N).

        Returns:
            DCT coefficients of shape (B, C, N, N).
        """
        # 2-D DCT = basis @ patch @ basis^T
        return self.basis @ patches @ self.basis.t()


class PatchArtifactExpert(nn.Module):
    """Low-level patch-based artifact detector.

    Pipeline:
        1.  Unfold image into a grid of patches.
        2.  Apply DCT to each patch.
        3.  Feed DCT coefficients through learnable 1×1 conv layers.
        4.  Aggregate via adaptive average pooling → feature vector.

    Args:
        patch_size: Grid patch size (default 8).
        in_channels: Number of input image channels.
        feature_dim: Output feature vector dimension.
    """

    def __init__(
        self,
        patch_size: int = 8,
        in_channels: int = 3,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dct = DCTLayer(patch_size)

        # Frequency analysis layers operating on DCT coefficients
        self.freq_encoder = nn.Sequential(
            # Treat each patch's DCT as a spatial feature map
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Spatial aggregation over re-assembled patch map
        self.aggregator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Unfold image into non-overlapping patches.

        Args:
            x: Image tensor (B, C, H, W).

        Returns:
            Patches tensor (B * n_patches, C, patch_size, patch_size).
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # Pad if needed
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape

        # Unfold: (B, C, H2/p, p, W2/p, p)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        n_h, n_w = patches.shape[2], patches.shape[3]
        # Reshape: (B * n_h * n_w, C, p, p)
        patches = patches.contiguous().view(B * n_h * n_w, C, p, p)
        return patches, n_h, n_w, B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level frequency features.

        Args:
            x: Input image tensor (B, C, H, W).

        Returns:
            Feature vector (B, feature_dim).
        """
        patches, n_h, n_w, B = self._extract_patches(x)

        # Apply DCT
        dct_patches = self.dct(patches)  # (B*N, C, p, p)

        # Encode frequencies
        freq_features = self.freq_encoder(dct_patches)  # (B*N, 256, p, p)

        # Reassemble into spatial grid
        # (B, n_h*n_w, 256, p, p) → (B, 256, n_h*p, n_w*p)
        freq_features = freq_features.view(B, n_h, n_w, 256, self.patch_size, self.patch_size)
        freq_features = freq_features.permute(0, 3, 1, 4, 2, 5).contiguous()
        freq_features = freq_features.view(B, 256, n_h * self.patch_size, n_w * self.patch_size)

        # Aggregate to fixed-size vector
        pooled = self.aggregator(freq_features)  # (B, 256)
        return self.projection(pooled)  # (B, feature_dim)
