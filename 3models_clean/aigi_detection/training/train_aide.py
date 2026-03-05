"""
AIDE training script.

Trains the AIDE multi-expert model on image data with:
    - BCE loss with logits
    - Adam optimiser (only trainable params — CLIP is frozen)
    - Cosine annealing LR scheduler
    - Expert gate weight monitoring
    - TensorBoard logging + CSV metrics
    - Model checkpointing

Usage:
    python -m training.train_aide --config configs/default.yaml
    python -m training.train_aide --config configs/default.yaml --smoke-test
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cifake_dataset import CIFAKEDataset
from models.aide.aide_model import AIDEModel
from utils.logging import ExperimentLogger
from utils.metrics import evaluate_predictions
from utils.seed import set_seed, get_device
from utils.transforms import get_aide_train_transforms, get_aide_val_transforms


def train_one_epoch(
    model: AIDEModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch.

    Returns:
        Dict with "loss", "accuracy", "patch_weight", "clip_weight".
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    gate_weights_sum = np.zeros(2)
    n_batches = 0

    for images, labels in tqdm(dataloader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits, gate_weights = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # Track gate weights
        gate_weights_sum += gate_weights.mean(dim=0).detach().cpu().numpy()
        n_batches += 1

    avg_gate = gate_weights_sum / max(n_batches, 1)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "patch_weight": float(avg_gate[0]),
        "clip_weight": float(avg_gate[1]),
    }


@torch.no_grad()
def validate(
    model: AIDEModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Validate and compute full metrics."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []
    total = 0
    gate_weights_sum = np.zeros(2)
    n_batches = 0

    for images, labels in tqdm(dataloader, desc="  Val", leave=False):
        images = images.to(device)
        labels_gpu = labels.float().unsqueeze(1).to(device)

        logits, gate_weights = model(images)
        loss = criterion(logits, labels_gpu)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        scores = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

        gate_weights_sum += gate_weights.mean(dim=0).cpu().numpy()
        n_batches += 1

    metrics = evaluate_predictions(
        np.array(all_labels), np.array(all_scores)
    )
    metrics["loss"] = total_loss / total
    avg_gate = gate_weights_sum / max(n_batches, 1)
    metrics["patch_weight"] = float(avg_gate[0])
    metrics["clip_weight"] = float(avg_gate[1])
    return metrics


def train_aide(config: dict, smoke_test: bool = False) -> str:
    """Full AIDE training pipeline.

    Args:
        config: Parsed YAML config dictionary.
        smoke_test: If True, use tiny subset and 1 epoch.

    Returns:
        Path to the best model checkpoint.
    """
    cfg = config["aide"]
    set_seed(config["seed"])
    device = get_device(config["device"])
    print(f"Device: {device}")

    # ── Data ──
    train_transform = get_aide_train_transforms(cfg["image_size"])
    val_transform = get_aide_val_transforms(cfg["image_size"])

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
        n_per_class = 50
        real_idx = [i for i, (_, l) in enumerate(train_dataset.samples) if l == 0][:n_per_class]
        fake_idx = [i for i, (_, l) in enumerate(train_dataset.samples) if l == 1][:n_per_class]
        train_dataset = Subset(train_dataset, real_idx + fake_idx)
        real_idx_t = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 0][:25]
        fake_idx_t = [i for i, (_, l) in enumerate(test_dataset.samples) if l == 1][:25]
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
    model = AIDEModel(
        patch_size=cfg["patch_size"],
        expert_dim=cfg["expert_dim"],
        clip_model_name=cfg["clip_model"],
        clip_pretrained=cfg["clip_pretrained"],
        device=str(device),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Only optimise trainable parameters (CLIP backbone is frozen)
    trainable_params = []
    for pg in model.get_trainable_params():
        trainable_params.extend(list(pg["params"]))

    optimizer = Adam(
        trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    epochs = 1 if smoke_test else cfg["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Logging ──
    results_dir = Path(config["results_dir"]) / "aide"
    logger = ExperimentLogger(
        log_dir=str(results_dir / "logs"),
        csv_path=str(results_dir / "training_metrics.csv"),
        experiment_name="aide_multiexpert",
    )

    # ── Training loop ──
    best_auc = -1.0
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / "best_model.pt")

    print(f"\nTraining AIDE for {epochs} epochs...")
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
        logger.log_scalar("gate/patch_weight", val_metrics["patch_weight"], epoch)
        logger.log_scalar("gate/clip_weight", val_metrics["clip_weight"], epoch)

        logger.log_metrics_to_csv(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.4f}",
                "train_acc": f"{train_metrics['accuracy']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_acc": f"{val_metrics['accuracy']:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
                "patch_w": f"{val_metrics['patch_weight']:.4f}",
                "clip_w": f"{val_metrics['clip_weight']:.4f}",
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
            f"Gate: P={val_metrics['patch_weight']:.2f} C={val_metrics['clip_weight']:.2f} | "
            f"{elapsed:.1f}s"
        )

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
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AIDE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_aide(config, smoke_test=args.smoke_test)
