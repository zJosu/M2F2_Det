"""
Explanation consistency and stability analysis.

Provides quantitative metrics for evaluating and comparing
explanation maps:

    - Map consistency: Spearman correlation between maps
    - Explanation stability: Variance under random augmentations
    - Cross-method overlap: IoU of high-attribution regions
"""

import numpy as np
from scipy.stats import spearmanr
from typing import List


def map_consistency(maps: List[np.ndarray]) -> float:
    """Compute average pairwise Spearman correlation between maps.

    Measures how consistent a set of explanation maps are with each
    other (e.g., maps from repeated runs or slight augmentations).

    Args:
        maps: List of 2-D attribution maps, all same shape.

    Returns:
        Average pairwise Spearman rank correlation ∈ [-1, 1].
    """
    if len(maps) < 2:
        return 1.0

    correlations = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            rho, _ = spearmanr(maps[i].ravel(), maps[j].ravel())
            if not np.isnan(rho):
                correlations.append(rho)

    return float(np.mean(correlations)) if correlations else 0.0


def explanation_stability(
    maps_under_perturbation: List[np.ndarray],
) -> float:
    """Measure how stable explanations are under augmentation.

    Computes the pixel-wise variance across maps generated from
    augmented versions of the same image, then averages.

    Lower values indicate more stable explanations.

    Args:
        maps_under_perturbation: List of maps from augmented versions
            of the same image.

    Returns:
        Average pixel-wise standard deviation (lower = more stable).
    """
    if len(maps_under_perturbation) < 2:
        return 0.0

    stacked = np.stack(maps_under_perturbation, axis=0)  # (N, H, W)
    pixel_std = np.std(stacked, axis=0)  # (H, W)
    return float(np.mean(pixel_std))


def cross_method_overlap(
    map_a: np.ndarray,
    map_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute IoU overlap between high-attribution regions.

    Binarises both maps at the given threshold and computes the
    Intersection over Union of the resulting masks.

    Args:
        map_a: First attribution map (H, W) in [0, 1].
        map_b: Second attribution map (H, W) in [0, 1].
        threshold: Binarisation threshold.

    Returns:
        IoU ∈ [0, 1].
    """
    mask_a = (map_a >= threshold).astype(bool)
    mask_b = (map_b >= threshold).astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_all_overlaps(
    maps_dict: dict,
    threshold: float = 0.5,
) -> dict:
    """Compute pairwise cross-method IoU for all method pairs.

    Args:
        maps_dict: Dict[method_name, ndarray (H, W)].
        threshold: Binarisation threshold.

    Returns:
        Dict[(method_a, method_b), IoU].
    """
    methods = list(maps_dict.keys())
    overlaps = {}

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            key = (methods[i], methods[j])
            # Resize maps if needed
            map_a = maps_dict[methods[i]]
            map_b = maps_dict[methods[j]]

            if map_a.shape != map_b.shape:
                from PIL import Image
                target_shape = max(map_a.shape, map_b.shape)
                map_a = np.array(Image.fromarray(
                    (map_a * 255).astype(np.uint8)
                ).resize(target_shape[::-1])) / 255.0
                map_b = np.array(Image.fromarray(
                    (map_b * 255).astype(np.uint8)
                ).resize(target_shape[::-1])) / 255.0

            overlaps[key] = cross_method_overlap(map_a, map_b, threshold)

    return overlaps
