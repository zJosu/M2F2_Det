"""
Explainability visualisation utilities.

Generates side-by-side comparison figures showing explanation maps
from all three detection methods for the same image.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_explanation_comparison(
    image: np.ndarray,
    maps: Dict[str, np.ndarray],
    label: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (18, 4),
) -> None:
    """Plot original image alongside explanation maps from each method.

    Args:
        image: Original image (H, W, 3) in [0, 1].
        maps: Dict[method_name, attribution_map (H, W)].
        label: Image label ("REAL" or "FAKE").
        save_path: Optional save path.
        figsize: Figure size.
    """
    n_methods = len(maps)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)

    # Original
    axes[0].imshow(image)
    axes[0].set_title(f"Original ({label})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Method maps
    cmaps = {"cifake": "jet", "aide": "inferno", "zed": "hot"}
    for i, (method, attr_map) in enumerate(maps.items()):
        cmap = cmaps.get(method, "viridis")
        axes[i + 1].imshow(attr_map, cmap=cmap, vmin=0, vmax=1)
        axes[i + 1].set_title(method.upper(), fontsize=12, fontweight="bold")
        axes[i + 1].axis("off")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_explanation_grid(
    images: List[np.ndarray],
    all_maps: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: Optional[str] = None,
    n_cols: int = 4,
    figsize_per_cell: tuple = (3, 3),
) -> None:
    """Grid of explanation comparisons for multiple images.

    Each row contains: original + one map per method.

    Args:
        images: List of (H, W, 3) images.
        all_maps: List of dicts mapping method → attribution map.
        labels: List of labels.
        save_path: Optional save path.
        n_cols: Number of columns (1 original + N methods).
        figsize_per_cell: Size per subplot cell.
    """
    if not images:
        return

    methods = list(all_maps[0].keys())
    n_cols = 1 + len(methods)  # original + each method
    n_rows = len(images)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )
    if n_rows == 1:
        axes = [axes]

    cmaps = {"cifake": "jet", "aide": "inferno", "zed": "hot"}

    for row, (img, maps, label) in enumerate(zip(images, all_maps, labels)):
        # Original
        axes[row][0].imshow(img)
        if row == 0:
            axes[row][0].set_title("Original", fontsize=10, fontweight="bold")
        axes[row][0].set_ylabel(label, fontsize=10, fontweight="bold")
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])

        # Methods
        for col, method in enumerate(methods):
            cmap = cmaps.get(method, "viridis")
            axes[row][col + 1].imshow(maps[method], cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                axes[row][col + 1].set_title(
                    method.upper(), fontsize=10, fontweight="bold"
                )
            axes[row][col + 1].set_xticks([])
            axes[row][col + 1].set_yticks([])

    fig.suptitle(
        "Explanation Map Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)


def plot_overlap_matrix(
    overlaps: Dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot a pairwise overlap (IoU) matrix as a heatmap.

    Args:
        overlaps: Dict[(method_a, method_b), IoU].
        save_path: Optional save path.
    """
    import seaborn as sns

    methods = sorted(set(m for pair in overlaps.keys() for m in pair))
    n = len(methods)
    matrix = np.eye(n)

    for (a, b), iou in overlaps.items():
        i, j = methods.index(a), methods.index(b)
        matrix[i, j] = iou
        matrix[j, i] = iou

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".3f",
        xticklabels=[m.upper() for m in methods],
        yticklabels=[m.upper() for m in methods],
        cmap="Blues", vmin=0, vmax=1,
        linewidths=1, ax=ax,
    )
    ax.set_title("Cross-Method Explanation Overlap (IoU)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
