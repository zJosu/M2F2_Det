"""
Unit tests for evaluation metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    compute_accuracy,
    compute_f1,
    compute_auc,
    evaluate_predictions,
    find_optimal_threshold,
)


class TestMetrics:
    """Tests for metric computation."""

    def test_perfect_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_perfect_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert compute_f1(y_true, y_pred) == 1.0

    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_score) == 1.0

    def test_random_auc(self):
        np.random.seed(42)
        y_true = np.array([0, 1] * 500)
        y_score = np.random.rand(1000)
        auc = compute_auc(y_true, y_score)
        assert 0.4 < auc < 0.6, "Random predictions should give AUC ≈ 0.5"

    def test_evaluate_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.3, 0.7, 0.9])
        result = evaluate_predictions(y_true, y_score)
        assert "accuracy" in result
        assert "f1" in result
        assert "auc" in result
        assert "optimal_threshold" in result
        assert result["accuracy"] == 1.0

    def test_single_class_auc(self):
        y_true = np.array([0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        assert compute_auc(y_true, y_score) == 0.0

    def test_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_score)
        assert 0.3 <= threshold <= 0.7
