"""
Robustness evaluation module.

Evaluates detection methods under various perturbations:
    - JPEG compression (Q = 50, 70, 90)
    - Gaussian noise (σ = 0.01, 0.05, 0.1)
    - Random crop + resize (80%, 60%)

Each perturbation is applied to the test images before inference
to measure degradation in detection accuracy.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import evaluate_predictions
from utils.transforms import get_robustness_transforms


class PerturbedDataset(Dataset):
    """Wraps an existing dataset with an additional perturbation.

    Perturbations that operate on PIL images are applied before the
    existing transform; perturbations on tensors are applied after.

    Args:
        base_dataset: Original dataset returning (image, label) or
            (image, label, path).
        perturbation: Callable perturbation to apply.
        is_tensor_perturbation: If True, perturbation operates on
            tensors; if False, on PIL images.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        perturbation: Callable,
        is_tensor_perturbation: bool = False,
    ):
        self.base = base_dataset
        self.perturbation = perturbation
        self.is_tensor = is_tensor_perturbation

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        result = self.base[idx]
        image = result[0]
        label = result[1]

        if self.is_tensor:
            # Apply perturbation in tensor domain
            image = self.perturbation(image)
        else:
            # PIL-based perturbation: convert tensor -> PIL -> perturb -> tensor
            if isinstance(image, torch.Tensor):
                orig_h, orig_w = image.shape[1], image.shape[2]
                # Undo normalisation: clamp to [0,1] and convert to PIL
                img_np = image.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np * 0.5 + 0.5, 0.0, 1.0)  # rough denorm
                pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
                pil_img = self.perturbation(pil_img)
                # Resize back to original spatial dims
                pil_img = pil_img.resize((orig_w, orig_h), Image.BILINEAR)
                # Convert back to tensor and re-normalise
                image = transforms.ToTensor()(pil_img)
                image = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )(image)
            else:
                image = self.perturbation(image)

        if len(result) == 3:
            return image, label, result[2]
        return image, label


def evaluate_robustness(
    method_name: str,
    model_or_detector,
    base_dataset: Dataset,
    config: dict,
    device: torch.device = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> List[Dict]:
    """Evaluate a method under multiple robustness perturbations.

    Args:
        method_name: "cifake", "aide", or "zed".
        model_or_detector: Model or detector instance.
        base_dataset: Clean test dataset (already transformed).
        config: Evaluation config section.
        device: Compute device.
        batch_size: Batch size for evaluation.
        num_workers: DataLoader workers.

    Returns:
        List of dicts, each with perturbation name + metrics.
    """
    from evaluation.evaluate import evaluate_method

    robustness_cfg = config.get("robustness", {})
    perturbations = get_robustness_transforms(
        jpeg_qualities=robustness_cfg.get("jpeg_qualities"),
        noise_stds=robustness_cfg.get("gaussian_noise_stds"),
        crop_scales=robustness_cfg.get("crop_scales"),
    )

    results = []

    for name, perturbation in perturbations.items():
        print(f"  Robustness: {name}...")

        # Determine if perturbation is tensor or PIL based
        is_tensor = name.startswith("noise_")

        perturbed = PerturbedDataset(
            base_dataset, perturbation, is_tensor_perturbation=is_tensor
        )
        loader = DataLoader(
            perturbed, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )

        metrics = evaluate_method(method_name, model_or_detector, loader, device)
        metrics["perturbation"] = name
        results.append(metrics)

    return results
