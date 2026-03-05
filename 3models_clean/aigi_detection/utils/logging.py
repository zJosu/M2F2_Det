"""
Experiment logging utilities.

Provides TensorBoard integration and CSV metric logging.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Unified experiment logger with TensorBoard and CSV backends.

    Args:
        log_dir: Directory for TensorBoard logs.
        csv_path: Optional path for CSV metric file.
        experiment_name: Human-readable experiment identifier.
    """

    def __init__(
        self,
        log_dir: str,
        csv_path: Optional[str] = None,
        experiment_name: str = "experiment",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))

        # CSV logging
        self.csv_path = Path(csv_path) if csv_path else None
        self._csv_initialized = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalar values under one tag group."""
        self.writer.add_scalars(main_tag, values, step)

    def log_metrics_to_csv(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Append a row of metrics to the CSV file.

        Args:
            metrics: Dictionary of metric_name → value.
            step: Optional step/epoch number.
        """
        if self.csv_path is None:
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"step": step, **metrics} if step is not None else metrics

        write_header = not self._csv_initialized or not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def log_image(self, tag: str, image_tensor, step: int) -> None:
        """Log an image tensor to TensorBoard.

        Args:
            tag: Image tag.
            image_tensor: Tensor of shape (C, H, W) with values in [0, 1].
            step: Global step.
        """
        self.writer.add_image(tag, image_tensor, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()


class CSVResultsWriter:
    """Standalone CSV writer for final evaluation results.

    Args:
        csv_path: Path to the output CSV file.
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list = []

    def add_row(self, row: Dict[str, Any]) -> None:
        """Add a result row."""
        self.rows.append(row)

    def save(self) -> None:
        """Write all rows to CSV."""
        if not self.rows:
            return
        # Collect ALL unique keys across rows (methods have different columns)
        all_keys = []
        seen = set()
        for row in self.rows:
            for k in row.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
