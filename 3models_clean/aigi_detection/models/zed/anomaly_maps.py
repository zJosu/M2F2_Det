"""
Anomaly map generation and visualization for ZED.

Provides utilities for creating, processing, and visualising the
NLL-entropy difference maps that form the basis of zero-shot
AI-generated image detection.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def generate_anomaly_map(
    nll_map: np.ndarray,
    entropy_map: np.ndarray,
    channel_reduce: str = "mean",
) -> np.ndarray:
    """Generate anomaly map from NLL and entropy maps.

    Args:
        nll_map: Per-pixel NLL, shape (C, H, W) or (H, W).
        entropy_map: Per-pixel entropy, same shape as nll_map.
        channel_reduce: How to reduce channels ("mean" or "max").

    Returns:
        Anomaly map of shape (H, W) with values ≥ 0.
    """
    diff = np.abs(nll_map - entropy_map)

    if diff.ndim == 3:
        if channel_reduce == "mean":
            diff = diff.mean(axis=0)
        elif channel_reduce == "max":
            diff = diff.max(axis=0)
        else:
            raise ValueError(f"Unknown channel_reduce: {channel_reduce}")

    return diff


def visualize_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "hot",
    alpha: float = 0.6,
    figsize: tuple = (14, 4),
) -> None:
    """Visualize an anomaly map alongside the original image.

    Creates a 3-panel figure: original image, anomaly heatmap, overlay.

    Args:
        image: Original image, shape (H, W, 3) in [0, 1].
        anomaly_map: Anomaly map, shape (H, W).
        save_path: Optional file path to save the figure.
        title: Optional figure title.
        cmap: Matplotlib colormap for the anomaly map.
        alpha: Transparency for the overlay.
        figsize: Figure size.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Anomaly heatmap
    norm = Normalize(vmin=anomaly_map.min(), vmax=anomaly_map.max())
    im = axes[1].imshow(anomaly_map, cmap=cmap, norm=norm)
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap=cmap, alpha=alpha, norm=norm)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def save_anomaly_maps_batch(
    images: list,
    anomaly_maps: list,
    labels: list,
    output_dir: str,
    n_samples: int = 30,
) -> list:
    """Generate and save anomaly map visualizations for a batch.

    Args:
        images: List of ndarray images, each (H, W, 3) in [0, 1].
        anomaly_maps: List of ndarray anomaly maps, each (H, W).
        labels: List of integer labels (0=REAL, 1=FAKE).
        output_dir: Output directory.
        n_samples: Maximum number of maps to save.

    Returns:
        List of saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    for i in range(min(n_samples, len(images))):
        label_str = "FAKE" if labels[i] == 1 else "REAL"
        save_path = out / f"zed_anomaly_{i:04d}_{label_str}.png"
        visualize_anomaly_map(
            image=images[i],
            anomaly_map=anomaly_maps[i],
            save_path=str(save_path),
            title=f"ZED Anomaly Map - {label_str}",
        )
        saved.append(str(save_path))

    return saved
