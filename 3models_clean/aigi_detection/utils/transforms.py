"""
Image transforms for data augmentation and robustness evaluation.

Includes standard training transforms and robustness perturbations
(JPEG compression, Gaussian noise, random crop + resize).
"""

import io
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ─────────────────────────── Standard Transforms ─────────────────────────── #

def get_cifake_train_transforms(image_size: int = 32) -> transforms.Compose:
    """Training transforms for CIFAKE (32×32 images).

    Applies horizontal flip, slight rotation, colour jitter,
    and normalises to ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_cifake_val_transforms(image_size: int = 32) -> transforms.Compose:
    """Validation/test transforms for CIFAKE."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_aide_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms for AIDE (224×224 for CLIP)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_aide_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transforms for AIDE."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_zed_transforms(image_size: int = 32) -> transforms.Compose:
    """Transforms for ZED (no normalisation — needs raw pixel values)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1] range
    ])


# ──────────────────────── Robustness Perturbations ───────────────────────── #

class JPEGCompress:
    """Simulate JPEG compression artefacts.

    Args:
        quality: JPEG quality factor (1–100).
    """

    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __repr__(self) -> str:
        return f"JPEGCompress(quality={self.quality})"


class GaussianNoise:
    """Add Gaussian noise to a tensor image.

    Args:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussianNoise(std={self.std})"


class RandomCropResize:
    """Random crop followed by resize back to original dimensions.

    Simulates cropping attacks that remove image borders.

    Args:
        scale: Fraction of area to keep (e.g. 0.8 = 80%).
        image_size: Target resize dimensions.
    """

    def __init__(self, scale: float = 0.8, image_size: int = 32):
        self.scale = scale
        self.image_size = image_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        left = np.random.randint(0, w - new_w + 1)
        top = np.random.randint(0, h - new_h + 1)
        img = img.crop((left, top, left + new_w, top + new_h))
        return img.resize((self.image_size, self.image_size), Image.BILINEAR)

    def __repr__(self) -> str:
        return f"RandomCropResize(scale={self.scale}, image_size={self.image_size})"


def get_robustness_transforms(
    jpeg_qualities: list = None,
    noise_stds: list = None,
    crop_scales: list = None,
    image_size: int = 32,
) -> dict:
    """Build a dictionary of named robustness transforms.

    Returns:
        Dict[str, callable] mapping transform name to transform instance.
    """
    jpeg_qualities = jpeg_qualities or [50, 70, 90]
    noise_stds = noise_stds or [0.01, 0.05, 0.1]
    crop_scales = crop_scales or [0.8, 0.6]

    robustness = {}
    for q in jpeg_qualities:
        robustness[f"jpeg_q{q}"] = JPEGCompress(quality=q)
    for s in noise_stds:
        robustness[f"noise_std{s}"] = GaussianNoise(std=s)
    for sc in crop_scales:
        robustness[f"crop_{int(sc*100)}pct"] = RandomCropResize(
            scale=sc, image_size=image_size
        )
    return robustness
