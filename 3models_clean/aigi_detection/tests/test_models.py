"""
Unit tests for model architectures.

Verifies forward pass output shapes and basic functionality
for CIFAKENet, AIDEModel, and PixelCNN codec.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCIFAKENet:
    """Tests for CIFAKENet CNN."""

    def test_forward_shape(self):
        from models.cifake.cnn import CIFAKENet

        model = CIFAKENet()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"

    def test_gradcam_layer(self):
        from models.cifake.cnn import CIFAKENet

        model = CIFAKENet()
        layer = model.get_gradcam_target_layer()
        assert isinstance(layer, torch.nn.Conv2d)

    def test_sigmoid_range(self):
        from models.cifake.cnn import CIFAKENet

        model = CIFAKENet()
        x = torch.randn(2, 3, 32, 32)
        out = torch.sigmoid(model(x))
        assert (out >= 0).all() and (out <= 1).all()


class TestPatchArtifactExpert:
    """Tests for PatchArtifactExpert."""

    def test_forward_shape(self):
        from models.aide.patch_expert import PatchArtifactExpert

        expert = PatchArtifactExpert(patch_size=8, feature_dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = expert(x)
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_small_image(self):
        from models.aide.patch_expert import PatchArtifactExpert

        expert = PatchArtifactExpert(patch_size=8, feature_dim=128)
        x = torch.randn(1, 3, 32, 32)
        out = expert(x)
        assert out.shape == (1, 128)


class TestPixelCNN:
    """Tests for PixelCNN codec."""

    def test_forward_shape(self):
        from models.zed.codec import PixelCNNEncoder

        codec = PixelCNNEncoder(n_channels=3, n_filters=32, n_layers=2)
        x = torch.randn(2, 3, 16, 16).clamp(0, 1)
        logits = codec(x)
        assert logits.shape == (2, 3 * 256, 16, 16)

    def test_nll_shape(self):
        from models.zed.codec import PixelCNNEncoder

        codec = PixelCNNEncoder(n_channels=3, n_filters=32, n_layers=2)
        x = torch.randn(2, 3, 16, 16).clamp(0, 1)
        nll = codec.compute_nll(x)
        assert nll.shape == (2, 3, 16, 16)

    def test_entropy_shape(self):
        from models.zed.codec import PixelCNNEncoder

        codec = PixelCNNEncoder(n_channels=3, n_filters=32, n_layers=2)
        x = torch.randn(2, 3, 16, 16).clamp(0, 1)
        entropy = codec.compute_entropy(x)
        assert entropy.shape == (2, 3, 16, 16)

    def test_nll_and_entropy_joint(self):
        from models.zed.codec import PixelCNNEncoder

        codec = PixelCNNEncoder(n_channels=3, n_filters=32, n_layers=2)
        x = torch.randn(1, 3, 8, 8).clamp(0, 1)
        nll, entropy = codec.compute_nll_and_entropy(x)
        assert nll.shape == entropy.shape == (1, 3, 8, 8)
        assert (nll >= 0).all(), "NLL should be non-negative"
        assert (entropy >= 0).all(), "Entropy should be non-negative"


class TestMaskedConv:
    """Tests for masked convolution causality."""

    def test_mask_type_a(self):
        from models.zed.codec import MaskedConv2d

        conv = MaskedConv2d("A", 3, 16, kernel_size=3, padding=1)
        # Center pixel mask should be zero
        k = conv.weight.shape[2] // 2
        assert conv.mask[:, :, k, k].sum() == 0

    def test_mask_type_b(self):
        from models.zed.codec import MaskedConv2d

        conv = MaskedConv2d("B", 3, 16, kernel_size=3, padding=1)
        # Center pixel mask should be non-zero
        k = conv.weight.shape[2] // 2
        assert conv.mask[:, :, k, k].sum() > 0
