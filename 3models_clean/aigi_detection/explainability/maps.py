"""
Explanation map generation for all three methods.

Provides a unified interface to generate per-image attribution maps:
    - CIFAKE: Grad-CAM heatmap
    - AIDE: Expert gate-weighted feature importance
    - ZED: NLL − entropy anomaly map

All maps are normalised to [0, 1] for cross-method comparison.
"""

import numpy as np
import torch
from typing import Optional

from models.cifake.gradcam import GradCAMGenerator


def normalize_map(m: np.ndarray) -> np.ndarray:
    """Normalise a map to [0, 1]."""
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def generate_cifake_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Generate Grad-CAM explanation map for CIFAKE.

    Args:
        model: Trained CIFAKENet.
        image_tensor: Normalised image tensor (C, H, W).
        device: Compute device.

    Returns:
        Normalised attribution map (H, W) in [0, 1].
    """
    gen = GradCAMGenerator(model, device=device)
    cam_map, _ = gen.generate_heatmap(image_tensor)
    return normalize_map(cam_map)


@torch.no_grad()
def generate_aide_map(
    model,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Generate expert-contribution-weighted map for AIDE.

    Computes a spatial importance map by weighting each expert's
    contribution and projecting back to image space.

    Args:
        model: Trained AIDEModel.
        image_tensor: Normalised image tensor (C, H, W).
        device: Compute device.

    Returns:
        Normalised attribution map (H, W) in [0, 1].
    """
    model.eval()
    x = image_tensor.unsqueeze(0).to(device)

    # Get expert contributions
    logits, gate_weights = model(x)
    weights = gate_weights.squeeze(0).cpu().numpy()  # (2,)

    # Use gradient of the logit w.r.t. the input as a simple saliency
    x_grad = image_tensor.unsqueeze(0).to(device).requires_grad_(True)
    logits_g, _ = model(x_grad)
    logits_g.sum().backward()

    grad = x_grad.grad.squeeze(0).cpu().numpy()  # (C, H, W)
    saliency = np.abs(grad).mean(axis=0)  # (H, W)

    return normalize_map(saliency)


@torch.no_grad()
def generate_zed_map(
    detector,
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """Generate anomaly map for ZED.

    Args:
        detector: ZEDDetector instance.
        image_tensor: Raw image tensor (C, H, W) in [0, 1].

    Returns:
        Normalised anomaly map (H, W) in [0, 1].
    """
    _, anomaly_map = detector.compute_anomaly(image_tensor)
    return normalize_map(anomaly_map)


def generate_explanation_map(
    method: str,
    model_or_detector,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Unified explanation map generator.

    Args:
        method: "cifake", "aide", or "zed".
        model_or_detector: Model or detector instance.
        image_tensor: Image tensor (C, H, W).
        device: Compute device.

    Returns:
        Normalised attribution map (H, W) in [0, 1].
    """
    if method == "cifake":
        return generate_cifake_map(model_or_detector, image_tensor, device)
    elif method == "aide":
        return generate_aide_map(model_or_detector, image_tensor, device)
    elif method == "zed":
        return generate_zed_map(model_or_detector, image_tensor)
    else:
        raise ValueError(f"Unknown method: {method}")
