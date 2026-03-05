"""
ZED Detector — Zero-shot Entropy-based Detection.

Computes anomaly scores by comparing the actual coding cost (NLL) of
an image against the expected cost (entropy) from a learned model of
natural images.  Regions where NLL > entropy indicate AI-generated
content.  No fine-tuning on fake images is required.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .codec import PixelCNNEncoder


class ZEDDetector:
    """Zero-shot AI-generated image detector.

    Detection pipeline (no training on fake images):
        1.  A PixelCNN codec models the distribution of natural images.
        2.  For each test image, compute NLL and entropy maps.
        3.  Anomaly map = |NLL − entropy| averaged over colour channels.
        4.  Anomaly score = mean of the anomaly map.
        5.  Threshold is calibrated from a small set of known real images.

    Args:
        codec: Pre-trained PixelCNNEncoder instance.
        device: Compute device.
        threshold: Decision threshold for binary classification.
            If None, must be set via ``calibrate()``.
    """

    def __init__(
        self,
        codec: PixelCNNEncoder,
        device: str = "cpu",
        threshold: Optional[float] = None,
    ):
        self.codec = codec.eval().to(device)
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def compute_anomaly(
        self, image: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """Compute anomaly score and anomaly map for a single image.

        Args:
            image: Tensor of shape (C, H, W) with values in [0, 1].

        Returns:
            Tuple of (anomaly_score, anomaly_map):
                - anomaly_score: scalar float
                - anomaly_map: ndarray of shape (H, W)
        """
        x = image.unsqueeze(0).to(self.device)
        nll, entropy = self.codec.compute_nll_and_entropy(x)

        # Anomaly map: mean absolute difference over channels
        diff = (nll - entropy).abs()  # (1, C, H, W)
        anomaly_map = diff.mean(dim=1).squeeze(0)  # (H, W)

        score = anomaly_map.mean().item()
        return score, anomaly_map.cpu().numpy()

    @torch.no_grad()
    def detect_batch(
        self, dataloader, return_maps: bool = False
    ) -> Dict[str, list]:
        """Run detection on a full dataloader.

        Args:
            dataloader: Yields (images, labels) or (images, labels, paths).
            return_maps: If True, also return anomaly maps.

        Returns:
            Dict with keys: "scores", "labels", and optionally "maps".
        """
        all_scores = []
        all_labels = []
        all_maps = [] if return_maps else None

        for batch in dataloader:
            images, labels = batch[0], batch[1]
            images = images.to(self.device)
            B = images.shape[0]

            nll, entropy = self.codec.compute_nll_and_entropy(images)
            diff = (nll - entropy).abs()
            score_maps = diff.mean(dim=1)  # (B, H, W)

            scores = score_maps.view(B, -1).mean(dim=1)  # (B,)
            all_scores.extend(scores.cpu().tolist())
            all_labels.extend(labels.tolist())

            if return_maps:
                all_maps.extend(score_maps.cpu().numpy())

        result = {"scores": all_scores, "labels": all_labels}
        if return_maps:
            result["maps"] = all_maps
        return result

    def calibrate(
        self,
        real_dataloader,
        percentile: float = 95.0,
    ) -> float:
        """Calibrate detection threshold from real images only.

        Sets ``self.threshold`` to the given percentile of anomaly
        scores computed on a calibration set of real images.

        Args:
            real_dataloader: DataLoader of real images only.
            percentile: Percentile for threshold (default 95).

        Returns:
            Calibrated threshold value.
        """
        scores = []
        for batch in real_dataloader:
            images = batch[0].to(self.device)
            B = images.shape[0]

            with torch.no_grad():
                nll, entropy = self.codec.compute_nll_and_entropy(images)
            diff = (nll - entropy).abs()
            batch_scores = diff.mean(dim=1).view(B, -1).mean(dim=1)
            scores.extend(batch_scores.cpu().tolist())

        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold

    def predict(self, scores: list) -> list:
        """Classify scores as real (0) or fake (1) using threshold.

        Args:
            scores: List of anomaly scores.

        Returns:
            List of binary predictions.
        """
        if self.threshold is None:
            raise ValueError(
                "Threshold not set. Call calibrate() first."
            )
        return [1 if s > self.threshold else 0 for s in scores]
