"""
Unit tests for robustness transforms.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.transforms import (
    JPEGCompress,
    GaussianNoise,
    RandomCropResize,
    get_robustness_transforms,
)


class TestJPEGCompress:
    """Tests for JPEG compression transform."""

    def test_returns_pil(self):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        transform = JPEGCompress(quality=50)
        result = transform(img)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8))
        transform = JPEGCompress(quality=75)
        result = transform(img)
        assert result.size == img.size


class TestGaussianNoise:
    """Tests for Gaussian noise transform."""

    def test_output_range(self):
        tensor = torch.rand(3, 32, 32)
        transform = GaussianNoise(std=0.1)
        result = transform(tensor)
        assert (result >= 0).all() and (result <= 1).all()

    def test_modifies_input(self):
        tensor = torch.rand(3, 32, 32)
        transform = GaussianNoise(std=0.1)
        result = transform(tensor)
        assert not torch.allclose(tensor, result)


class TestRandomCropResize:
    """Tests for random crop + resize transform."""

    def test_output_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        transform = RandomCropResize(scale=0.8, image_size=32)
        result = transform(img)
        assert result.size == (32, 32)


class TestRobustnessTransforms:
    """Tests for the robustness transform builder."""

    def test_default_transforms(self):
        transforms = get_robustness_transforms()
        assert len(transforms) > 0
        assert "jpeg_q50" in transforms
        assert "noise_std0.05" in transforms
        assert "crop_80pct" in transforms
