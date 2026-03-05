"""
Unified evaluation module.

Provides a consistent interface to evaluate all three detection
methods (CIFAKE, AIDE, ZED) and save results to CSV.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import CSVResultsWriter
from utils.metrics import evaluate_predictions


@torch.no_grad()
def evaluate_cifake(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate CIFAKENet on a test dataloader.

    Args:
        model: Trained CIFAKENet model.
        dataloader: Test data loader.
        device: Compute device.

    Returns:
        Metrics dictionary.
    """
    model.eval()
    all_labels, all_scores = [], []

    for batch in tqdm(dataloader, desc="Eval CIFAKE"):
        images, labels = batch[0].to(device), batch[1]
        logits = model(images)
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    return evaluate_predictions(np.array(all_labels), np.array(all_scores))


@torch.no_grad()
def evaluate_aide(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate AIDE model on a test dataloader.

    Returns metrics plus average expert gate weights.
    """
    model.eval()
    all_labels, all_scores = [], []
    gate_sum = np.zeros(2)
    n = 0

    for batch in tqdm(dataloader, desc="Eval AIDE"):
        images, labels = batch[0].to(device), batch[1]
        logits, gate_weights = model(images)
        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())
        gate_sum += gate_weights.mean(dim=0).cpu().numpy()
        n += 1

    metrics = evaluate_predictions(np.array(all_labels), np.array(all_scores))
    avg_gate = gate_sum / max(n, 1)
    metrics["patch_expert_weight"] = float(avg_gate[0])
    metrics["clip_expert_weight"] = float(avg_gate[1])
    return metrics


def evaluate_zed(
    detector,
    dataloader: DataLoader,
) -> dict:
    """Evaluate ZED detector on a test dataloader.

    The detector should already have a calibrated threshold.

    Returns:
        Metrics dictionary.
    """
    result = detector.detect_batch(dataloader)
    scores = np.array(result["scores"])
    labels = np.array(result["labels"])

    preds = detector.predict(result["scores"])
    metrics = evaluate_predictions(labels, scores)
    return metrics


def evaluate_method(
    method_name: str,
    model_or_detector,
    dataloader: DataLoader,
    device: torch.device = None,
) -> dict:
    """Unified evaluation dispatcher.

    Args:
        method_name: One of "cifake", "aide", "zed".
        model_or_detector: The model or detector instance.
        dataloader: Test data loader.
        device: Compute device (not needed for ZED).

    Returns:
        Metrics dictionary.
    """
    if method_name == "cifake":
        metrics = evaluate_cifake(model_or_detector, dataloader, device)
    elif method_name == "aide":
        metrics = evaluate_aide(model_or_detector, dataloader, device)
    elif method_name == "zed":
        metrics = evaluate_zed(model_or_detector, dataloader)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics["method"] = method_name
    return metrics


def save_evaluation_results(
    results: list,
    output_path: str,
) -> None:
    """Save list of result dicts to CSV.

    Args:
        results: List of metric dictionaries.
        output_path: CSV file path.
    """
    writer = CSVResultsWriter(output_path)
    for r in results:
        writer.add_row(r)
    writer.save()
    print(f"Results saved to {output_path}")
