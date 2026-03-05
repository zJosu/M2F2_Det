"""
CIFAKE dataset download helper.

The CIFAKE dataset is available on Kaggle:
  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Usage:
    python -m datasets.download_cifake --output data/CIFAKE

If kagglehub is installed, downloads automatically.
Otherwise prints manual download instructions.
"""

import argparse
import shutil
import sys
from pathlib import Path


def download_cifake(output_dir: str) -> None:
    """Download CIFAKE dataset to the specified directory.

    Attempts to use kagglehub for automatic download.
    Falls back to printing manual instructions.

    Args:
        output_dir: Target directory for the dataset.
    """
    output = Path(output_dir)

    # Check if already downloaded
    if (output / "train" / "REAL").is_dir() and (output / "test" / "REAL").is_dir():
        print(f"CIFAKE dataset already exists at {output}")
        return

    try:
        import kagglehub

        print("Downloading CIFAKE dataset via kagglehub...")
        path = kagglehub.dataset_download(
            "birdy654/cifake-real-and-ai-generated-synthetic-images"
        )
        print(f"Downloaded to: {path}")

        # Copy to target location
        src = Path(path)
        output.mkdir(parents=True, exist_ok=True)

        for split in ["train", "test"]:
            src_split = src / split
            dst_split = output / split
            if src_split.is_dir() and not dst_split.is_dir():
                shutil.copytree(str(src_split), str(dst_split))
                print(f"  Copied {split}/ → {dst_split}")

        print("Done!")

    except ImportError:
        print("=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print()
        print("1. Install kagglehub:  pip install kagglehub")
        print("   OR download manually from:")
        print("   https://www.kaggle.com/datasets/birdy654/"
              "cifake-real-and-ai-generated-synthetic-images")
        print()
        print("2. Extract to the following structure:")
        print(f"   {output}/")
        print(f"   ├── train/")
        print(f"   │   ├── REAL/   (60,000 images)")
        print(f"   │   └── FAKE/   (60,000 images)")
        print(f"   └── test/")
        print(f"       ├── REAL/   (10,000 images)")
        print(f"       └── FAKE/   (10,000 images)")
        print()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CIFAKE dataset")
    parser.add_argument(
        "--output", type=str, default="data/CIFAKE",
        help="Output directory for CIFAKE dataset",
    )
    args = parser.parse_args()
    download_cifake(args.output)
