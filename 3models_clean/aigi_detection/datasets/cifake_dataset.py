"""
CIFAKE dataset loader.

Expects the following directory layout:
    data/CIFAKE/
        train/
            REAL/   ← real images (from CIFAR-10)
            FAKE/   ← AI-generated images
        test/
            REAL/
            FAKE/

Each image is a 32×32 RGB PNG/JPEG.
Labels: 0 = REAL, 1 = FAKE.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class CIFAKEDataset(Dataset):
    """PyTorch Dataset for the CIFAKE benchmark.

    Args:
        root: Path to the CIFAKE split directory (e.g. ``data/CIFAKE/train``).
        transform: Optional torchvision transform pipeline.
        return_path: If True, __getitem__ returns (image, label, path).
    """

    CLASSES = {"REAL": 0, "FAKE": 1}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.return_path = return_path

        self.samples: list = []  # list of (path, label)
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory tree and build sample list."""
        for class_name, label in self.CLASSES.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected directory {class_dir} not found. "
                    f"Please download the CIFAKE dataset."
                )
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((str(img_file), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, path
        return image, label

    def __repr__(self) -> str:
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        return (
            f"CIFAKEDataset(root='{self.root}', "
            f"real={real_count}, fake={fake_count}, total={len(self)})"
        )
