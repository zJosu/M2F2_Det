"""
Evaluation metric utilities.

Provides accuracy, F1-score, and ROC-AUC computation with a unified
interface for comparing detection methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (thresholded).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1-score (binary).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        F1-score as a float in [0, 1].
    """
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC from continuous scores.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities or continuous scores.

    Returns:
        AUC as a float in [0, 1].
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def compute_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute full ROC curve.

    Returns:
        Dict with keys "fpr", "tpr", "thresholds".
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find threshold maximising Youden's J statistic on the ROC curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities.

    Returns:
        Optimal threshold value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_statistic = tpr - fpr
    return float(thresholds[np.argmax(j_statistic)])


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = 0.5,
) -> Dict[str, Any]:
    """Compute all standard metrics from predictions.

    Args:
        y_true: Ground-truth binary labels (0 = real, 1 = fake).
        y_score: Predicted probabilities / continuous scores.
        threshold: Decision threshold for binary metrics.

    Returns:
        Dictionary with accuracy, f1, auc, and optimal_threshold.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    optimal_thresh = find_optimal_threshold(y_true, y_score)
    y_pred_optimal = (y_score >= optimal_thresh).astype(int)

    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "auc": compute_auc(y_true, y_score),
        "accuracy_optimal": compute_accuracy(y_true, y_pred_optimal),
        "f1_optimal": compute_f1(y_true, y_pred_optimal),
        "optimal_threshold": optimal_thresh,
    }
