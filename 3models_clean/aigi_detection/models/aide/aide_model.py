"""
AIDE Model — AI-generated Image DEtector with Hybrid Features.

Composes the PatchArtifactExpert and SemanticCLIPExpert through a
gated fusion module for binary real-vs-fake classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .patch_expert import PatchArtifactExpert
from .clip_expert import SemanticCLIPExpert
from .fusion import ExpertFusionModule


class AIDEModel(nn.Module):
    """AIDE: Multi-expert AI-generated image detector.

    Combines low-level patch artifact features (DCT / frequency domain)
    with high-level semantic features (CLIP embeddings) through a
    learned gating mechanism.

    Args:
        patch_size: Patch grid size for the artifact expert.
        expert_dim: Feature vector dimension for each expert.
        clip_model_name: OpenCLIP model identifier.
        clip_pretrained: Pretrained weights tag.
        device: Device for CLIP model loading.
    """

    def __init__(
        self,
        patch_size: int = 8,
        expert_dim: int = 256,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: str = "cpu",
    ):
        super().__init__()

        self.patch_expert = PatchArtifactExpert(
            patch_size=patch_size,
            in_channels=3,
            feature_dim=expert_dim,
        )

        self.clip_expert = SemanticCLIPExpert(
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            feature_dim=expert_dim,
            device=device,
        )

        self.fusion = ExpertFusionModule(
            expert_dim=expert_dim,
            n_experts=2,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both experts and fusion.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Tuple of (logits, gate_weights):
                - logits: (B, 1) binary classification logits
                - gate_weights: (B, 2) per-expert contribution weights
        """
        # Extract features from both experts
        patch_features = self.patch_expert(x)      # (B, expert_dim)
        clip_features = self.clip_expert(x)        # (B, expert_dim)

        # Fuse and classify
        logits, gate_weights = self.fusion([patch_features, clip_features])
        return logits, gate_weights

    def get_expert_contributions(self, x: torch.Tensor) -> Dict[str, float]:
        """Get per-expert contribution weights for a batch.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Dict with average gate weights per expert.
        """
        with torch.no_grad():
            self.forward(x)
        return self.fusion.get_expert_weights()

    def get_trainable_params(self) -> list:
        """Return only the trainable parameters (excludes frozen CLIP).

        Returns:
            List of parameter groups for the optimizer:
                - patch_expert parameters
                - clip_expert projection head parameters
                - fusion module parameters
        """
        return [
            {"params": self.patch_expert.parameters(), "lr_scale": 1.0},
            {"params": self.clip_expert.projection.parameters(), "lr_scale": 1.0},
            {"params": self.fusion.parameters(), "lr_scale": 1.0},
        ]
