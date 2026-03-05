"""
Semantic CLIP Expert — High-level semantic feature extraction.

Uses a frozen pre-trained CLIP vision encoder (ViT-B/32) to compute
image embeddings, then projects them through a learnable head to
produce a feature vector for the fusion module.

Captures high-level semantic inconsistencies that AI-generated images
may exhibit (e.g., unnatural compositions, object relationships).
"""

import torch
import torch.nn as nn

try:
    import open_clip
except ImportError:
    open_clip = None


class SemanticCLIPExpert(nn.Module):
    """CLIP-based semantic feature extractor.

    The CLIP backbone is **frozen** during training — only the
    projection head is learnable.

    Args:
        clip_model_name: OpenCLIP model identifier (e.g. "ViT-B-32").
        clip_pretrained: Pretrained weights tag (e.g. "openai").
        feature_dim: Output feature vector dimension.
        device: Compute device for CLIP model loading.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        feature_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.feature_dim = feature_dim

        if open_clip is None:
            raise ImportError(
                "open-clip-torch is required for AIDE. "
                "Install with: pip install open-clip-torch"
            )

        # Load pre-trained CLIP
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=device,
        )
        self.clip_visual = clip_model.visual

        # Freeze the entire CLIP backbone
        for param in self.clip_visual.parameters():
            param.requires_grad = False

        # Determine CLIP output dimension dynamically
        clip_dim = self._get_clip_dim(device)

        # Learnable projection head
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

    def _get_clip_dim(self, device: str) -> int:
        """Probe CLIP output dimension with a dummy forward pass."""
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            out = self.clip_visual(dummy)
            return out.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract semantic features via CLIP + projection.

        Args:
            x: Input image tensor (B, 3, 224, 224), normalised.

        Returns:
            Feature vector (B, feature_dim).
        """
        # CLIP forward (no gradient for backbone)
        with torch.no_grad():
            clip_features = self.clip_visual(x)  # (B, clip_dim)

        # Learnable projection
        return self.projection(clip_features.float())  # (B, feature_dim)
