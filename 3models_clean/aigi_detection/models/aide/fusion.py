"""
Expert Fusion Module — Gated mixture of experts.

Combines the outputs of multiple experts (PatchArtifactExpert and
SemanticCLIPExpert) using a learned gating network.  The gate
produces per-expert weights via softmax, enabling interpretation
of which expert contributed most to each decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ExpertFusionModule(nn.Module):
    """Gated fusion of expert feature vectors.

    Architecture:
        1.  Concatenate expert features → (B, n_experts × expert_dim)
        2.  Gating network → softmax weights per expert
        3.  Weighted sum of expert features → (B, expert_dim)
        4.  Classifier head → binary logit

    Args:
        expert_dim: Dimension of each expert's output.
        n_experts: Number of experts to fuse.
    """

    def __init__(self, expert_dim: int = 256, n_experts: int = 2):
        super().__init__()
        self.expert_dim = expert_dim
        self.n_experts = n_experts
        concat_dim = expert_dim * n_experts

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_experts),
        )

        # Binary classifier on fused features
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Cache for last gate weights (for visualization)
        self._last_gate_weights: torch.Tensor = None

    def forward(
        self,
        expert_features: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse expert features and classify.

        Args:
            expert_features: List of tensors, each (B, expert_dim).

        Returns:
            Tuple of (logits, gate_weights) where logits is (B, 1)
            and gate_weights is (B, n_experts).
        """
        assert len(expert_features) == self.n_experts

        # Concatenate all expert outputs
        concat = torch.cat(expert_features, dim=1)  # (B, n_experts * expert_dim)

        # Compute gating weights
        gate_weights = F.softmax(self.gate(concat), dim=1)  # (B, n_experts)
        self._last_gate_weights = gate_weights.detach()

        # Weighted sum of expert features
        # Stack: (B, n_experts, expert_dim) * (B, n_experts, 1) → (B, expert_dim)
        stacked = torch.stack(expert_features, dim=1)  # (B, n_experts, expert_dim)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, expert_dim)

        # Classify
        logits = self.classifier(fused)  # (B, 1)
        return logits, gate_weights

    def get_expert_weights(self) -> Dict[str, float]:
        """Return average gate weights from the last forward pass.

        Returns:
            Dict mapping expert index to average weight.
        """
        if self._last_gate_weights is None:
            return {}
        avg = self._last_gate_weights.mean(dim=0).cpu().tolist()
        return {
            "patch_expert_weight": avg[0],
            "clip_expert_weight": avg[1],
        }
