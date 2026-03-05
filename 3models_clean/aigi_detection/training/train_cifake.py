"""
CIFAKE training script.

Trains the CIFAKENet CNN on the CIFAKE dataset with:
    - BCE loss with logits
    - Adam optimiser
    - Cosine annealing LR scheduler
    - TensorBoard logging
    - Model checkpointing (best val accuracy)

Usage:
    python -m training.train_cifake --config configs/default.yaml
    python -m training.train_cifake --config configs/default.yaml --smoke-test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml

# Allow running as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cifake_dataset import CIFAKEDataset
from models.cifake.cnn import CIFAKENet
from utils.logging import ExperimentLogger
from utils.metrics import evaluate_predictions
from utils.seed import set_seed, get_device
from utils.transforms import get_cifake_train_transforms, get_cifake_val_transforms


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch.

    Returns:
        Dict with "loss", "accuracy".
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate and compute full metrics.

    Returns:
        Dict with "loss", "accuracy", "f1", "auc", etc.
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []
    total = 0

    for images, labels in tqdm(dataloader, desc="  Val", leave=False):
        images = images.to(device)
        labels_gpu = labels.float().unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, labels_gpu)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    metrics = evaluate_predictions(
        np.array(all_labels), np.array(all_scores)
    )
    metrics["loss"] = total_loss / total
    return metrics


def train_cifake(config: dict, smoke_test: bool = False) -> str:
    """Full CIFAKE training pipeline.

    Args:
        config: Parsed YAML config dictionary.
        smoke_test: If True, use tiny subset and 1 epoch.

    Returns:
        Path to the best model checkpoint.
    """
    cfg = config["cifake"]
    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Device: {device}")

    # ── Data ──
    train_transform = get_cifake_train_transforms(cfg["image_size"])
    val_transform = get_cifake_val_transforms(cfg["image_size"])

    train_dataset = CIFAKEDataset(
        root=str(Path(cfg["data_dir"]) / "train"),
        transform=train_transform,
    )
    test_dataset = CIFAKEDataset(
        root=str(Path(cfg["data_dir"]) / "test"),
        transform=val_transform,
    )

    if smoke_test:
        # Balanced smoke test subset (both classes)
        n_per_class = 100
        real_idx = [i for i, (_, l) in enumerate(train_dataset.samples) if l == 0][:n_per_class]
        fake_idx = [i for i, (_, l) in enumerate(train_dataset.samples) if l == 1][:n_per_class]
        train_dataset = Subset(train_dataset, real_idx + fake_idx)
        real_idx_t = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 0][:50]
        fake_idx_t = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 1][:50]
        test_dataset = Subset(test_dataset, real_idx_t + fake_idx_t)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=config["num_workers"], pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} | Val: {len(test_dataset)}")

    # ── Model ──
    model = CIFAKENet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    epochs = 1 if smoke_test else cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Logging ──
    results_dir = Path(config["results_dir"]) / "cifake"
    logger = ExperimentLogger(
        log_dir=str(results_dir / "logs"),
        csv_path=str(results_dir / "training_metrics.csv"),
        experiment_name="cifake_cnn",
    )

    # ── Training loop ──
    best_auc = -1.0
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / "best_model.pt")

    print(f"\nTraining CIFAKENet for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        logger.log_scalar("train/loss", train_metrics["loss"], epoch)
        logger.log_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        logger.log_scalar("val/loss", val_metrics["loss"], epoch)
        logger.log_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        logger.log_scalar("val/f1", val_metrics["f1"], epoch)
        logger.log_scalar("val/auc", val_metrics["auc"], epoch)
        logger.log_metrics_to_csv(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_acc": f"{train_metrics['accuracy']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_acc": f"{val_metrics['accuracy']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint
        if val_metrics["auc"] >= best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_auc,
                    "config": cfg,
                },
                best_path,
            )
            print(f"  * New best AUC: {best_auc:.4f} - saved checkpoint")

    logger.close()
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAKENet")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a quick smoke test with tiny data",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cifake(config, smoke_test=args.smoke_test)
