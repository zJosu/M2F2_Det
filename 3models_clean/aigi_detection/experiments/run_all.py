"""
Master experiment runner.

Orchestrates the full pipeline:
    1. Train CIFAKE CNN
    2. Train AIDE multi-expert
    3. Pre-train PixelCNN codec for ZED
    4. Evaluate all methods on the same test set
    5. Run robustness evaluation
    6. Generate explainability analysis
    7. Produce comparison reports and figures

Usage:
    python -m experiments.run_all --config configs/default.yaml
    python -m experiments.run_all --config configs/default.yaml --smoke-test
    python -m experiments.run_all --config configs/default.yaml --eval-only
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cifake_dataset import CIFAKEDataset
from evaluation.compare import generate_all_comparison_plots
from evaluation.evaluate import evaluate_method, save_evaluation_results
from evaluation.robustness import evaluate_robustness
from explainability.consistency import (
    compute_all_overlaps,
    explanation_stability,
    map_consistency,
)
from explainability.maps import generate_explanation_map
from explainability.visualize import (
    plot_explanation_grid,
    plot_overlap_matrix,
)
from models.cifake.cnn import CIFAKENet
from models.cifake.gradcam import GradCAMGenerator
from models.aide.aide_model import AIDEModel
from models.zed.codec import PixelCNNEncoder
from models.zed.detector import ZEDDetector
from models.zed.anomaly_maps import save_anomaly_maps_batch
from training.train_cifake import train_cifake
from training.train_aide import train_aide
from utils.logging import CSVResultsWriter
from utils.metrics import compute_roc_curve
from utils.seed import set_seed, get_device
from utils.transforms import (
    get_cifake_val_transforms,
    get_aide_val_transforms,
    get_zed_transforms,
)


def pretrain_pixelcnn(config: dict, device: torch.device, smoke_test: bool = False) -> str:
    """Pre-train PixelCNN codec on real images.

    Args:
        config: Global config dict.
        device: Compute device.
        smoke_test: Use tiny data if True.

    Returns:
        Path to saved codec checkpoint.
    """
    from torch.utils.data import DataLoader, Subset
    from torch.optim import Adam
    from tqdm import tqdm

    cfg = config["zed"]
    transform = get_zed_transforms(cfg["image_size"])

    # Train on REAL images only (the codec learns natural image statistics)
    train_data = CIFAKEDataset(
        root=str(Path(config["cifake"]["data_dir"]) / "train"),
        transform=transform,
    )
    # Filter to real images only
    real_indices = [i for i, (_, label) in enumerate(train_data.samples) if label == 0]
    real_dataset = Subset(train_data, real_indices)

    if smoke_test:
        real_dataset = Subset(real_dataset, range(min(100, len(real_dataset))))

    loader = DataLoader(
        real_dataset, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=config["num_workers"],
    )

    # Model
    codec = PixelCNNEncoder(
        n_channels=cfg["n_channels"],
        n_filters=cfg["n_filters"],
        n_layers=cfg["n_layers"],
    ).to(device)

    optimizer = Adam(codec.parameters(), lr=1e-3)
    epochs = 2 if smoke_test else 10

    print(f"\nPre-training PixelCNN codec for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        codec.train()
        total_loss = 0.0
        n = 0

        for images, _ in tqdm(loader, desc=f"  Codec Epoch {epoch}", leave=False):
            images = images.to(device)

            optimizer.zero_grad()
            nll = codec.compute_nll(images)
            loss = nll.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n += images.size(0)

        avg_loss = total_loss / n
        print(f"  Epoch {epoch}/{epochs} - NLL loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt_dir = Path(config["results_dir"]) / "zed" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "pixelcnn_codec.pt")
    torch.save(codec.state_dict(), ckpt_path)
    print(f"Codec saved to {ckpt_path}")
    return ckpt_path


def run_all(config: dict, smoke_test: bool = False, eval_only: bool = False):
    """Execute the full experiment pipeline.

    Args:
        config: Parsed YAML config.
        smoke_test: Use tiny data subsets.
        eval_only: Skip training, load existing checkpoints.
    """
    set_seed(config["seed"])
    device = get_device(config["device"])
    results_dir = Path(config["results_dir"])
    print("=" * 60)
    print("AIGI Detection - Full Experiment Pipeline")
    print(f"Device: {device} | Smoke test: {smoke_test}")
    print("=" * 60)

    t_start = time.time()

    # ================================================================
    # PHASE 1: TRAINING
    # ================================================================
    if not eval_only:
        print("\n" + "-" * 40)
        print("PHASE 1: TRAINING")
        print("-" * 40)

        # 1a. Train CIFAKE
        print("\n> Training CIFAKE CNN...")
        cifake_ckpt = train_cifake(config, smoke_test=smoke_test)

        # 1b. Train AIDE
        print("\n> Training AIDE Multi-Expert...")
        aide_ckpt = train_aide(config, smoke_test=smoke_test)

        # 1c. Pre-train ZED codec
        print("\n> Pre-training ZED PixelCNN Codec...")
        codec_ckpt = pretrain_pixelcnn(config, device, smoke_test=smoke_test)
    else:
        cifake_ckpt = str(results_dir / "cifake" / "checkpoints" / "best_model.pt")
        aide_ckpt = str(results_dir / "aide" / "checkpoints" / "best_model.pt")
        codec_ckpt = str(results_dir / "zed" / "checkpoints" / "pixelcnn_codec.pt")

    # ================================================================
    # PHASE 2: LOAD MODELS
    # ================================================================
    print("\n" + "-" * 40)
    print("PHASE 2: LOADING MODELS")
    print("-" * 40)

    # CIFAKE
    cifake_model = CIFAKENet().to(device)
    ckpt = torch.load(cifake_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        cifake_model.load_state_dict(ckpt["model_state_dict"])
    else:
        cifake_model.load_state_dict(ckpt)
    cifake_model.eval()
    print("  [OK] CIFAKENet loaded")

    # AIDE
    aide_model = AIDEModel(
        patch_size=config["aide"]["patch_size"],
        expert_dim=config["aide"]["expert_dim"],
        clip_model_name=config["aide"]["clip_model"],
        clip_pretrained=config["aide"]["clip_pretrained"],
        device=str(device),
    ).to(device)
    ckpt = torch.load(aide_ckpt, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        aide_model.load_state_dict(ckpt["model_state_dict"])
    else:
        aide_model.load_state_dict(ckpt)
    aide_model.eval()
    print("  [OK] AIDE loaded")

    # ZED
    cfg_zed = config["zed"]
    codec = PixelCNNEncoder(
        n_channels=cfg_zed["n_channels"],
        n_filters=cfg_zed["n_filters"],
        n_layers=cfg_zed["n_layers"],
    ).to(device)
    codec.load_state_dict(torch.load(codec_ckpt, map_location=device, weights_only=True))
    zed_detector = ZEDDetector(codec, device=str(device))
    print("  [OK] ZED Detector loaded")

    # ================================================================
    # PHASE 3: EVALUATION
    # ================================================================
    print("\n" + "-" * 40)
    print("PHASE 3: EVALUATION")
    print("-" * 40)

    from torch.utils.data import DataLoader, Subset

    # Test datasets (different transforms for each method)
    cifake_test = CIFAKEDataset(
        root=str(Path(config["cifake"]["data_dir"]) / "test"),
        transform=get_cifake_val_transforms(config["cifake"]["image_size"]),
    )
    aide_test = CIFAKEDataset(
        root=str(Path(config["aide"]["data_dir"]) / "test"),
        transform=get_aide_val_transforms(config["aide"]["image_size"]),
    )
    zed_test = CIFAKEDataset(
        root=str(Path(config["cifake"]["data_dir"]) / "test"),
        transform=get_zed_transforms(cfg_zed["image_size"]),
    )

    if smoke_test:
        # Balanced subsets (both classes)
        for ds_name, ds_obj in [("cifake_test", cifake_test), ("aide_test", aide_test), ("zed_test", zed_test)]:
            real_idx = [i for i, (_, l) in enumerate(ds_obj.samples) if l == 0][:50]
            fake_idx = [i for i, (_, l) in enumerate(ds_obj.samples) if l == 1][:50]
            balanced = Subset(ds_obj, real_idx + fake_idx)
            if ds_name == "cifake_test":
                cifake_test = balanced
            elif ds_name == "aide_test":
                aide_test = balanced
            else:
                zed_test = balanced

    cifake_loader = DataLoader(cifake_test, batch_size=config["cifake"]["batch_size"], num_workers=config["num_workers"])
    aide_loader = DataLoader(aide_test, batch_size=config["aide"]["batch_size"], num_workers=config["num_workers"])
    zed_loader = DataLoader(zed_test, batch_size=cfg_zed["batch_size"], num_workers=config["num_workers"])

    # Calibrate ZED threshold
    print("\n> Calibrating ZED threshold...")
    zed_real = CIFAKEDataset(
        root=str(Path(config["cifake"]["data_dir"]) / "test"),
        transform=get_zed_transforms(cfg_zed["image_size"]),
    )
    real_indices = [i for i, (_, l) in enumerate(zed_real.samples) if l == 0]
    cal_n = min(cfg_zed.get("calibration_n", 200), len(real_indices))
    cal_subset = Subset(zed_real, real_indices[:cal_n])
    cal_loader = DataLoader(cal_subset, batch_size=cfg_zed["batch_size"], num_workers=config["num_workers"])
    threshold = zed_detector.calibrate(cal_loader)
    print(f"  ZED threshold: {threshold:.4f}")

    # Evaluate all methods
    all_results = []
    roc_data = {}

    for name, model, loader in [
        ("cifake", cifake_model, cifake_loader),
        ("aide", aide_model, aide_loader),
        ("zed", zed_detector, zed_loader),
    ]:
        print(f"\n> Evaluating {name.upper()}...")
        metrics = evaluate_method(name, model, loader, device)
        all_results.append(metrics)
        print(f"  Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")

    # Save results
    results_csv = str(results_dir / "comparison" / "results.csv")
    save_evaluation_results(all_results, results_csv)

    # ================================================================
    # PHASE 4: ROBUSTNESS
    # ================================================================
    print("\n" + "-" * 40)
    print("PHASE 4: ROBUSTNESS EVALUATION")
    print("-" * 40)

    all_robustness = []
    for name, model, test_ds in [
        ("cifake", cifake_model, cifake_test),
        ("aide", aide_model, aide_test),
        ("zed", zed_detector, zed_test),
    ]:
        print(f"\n> Robustness - {name.upper()}...")
        rob_results = evaluate_robustness(
            name, model, test_ds, config["evaluation"],
            device=device,
            batch_size=config["cifake"]["batch_size"],
            num_workers=config["num_workers"],
        )
        all_robustness.extend(rob_results)

    rob_csv = str(results_dir / "comparison" / "robustness.csv")
    save_evaluation_results(all_robustness, rob_csv)

    # ================================================================
    # PHASE 5: EXPLAINABILITY
    # ================================================================
    print("\n" + "-" * 40)
    print("PHASE 5: EXPLAINABILITY ANALYSIS")
    print("-" * 40)

    # Generate Grad-CAM heatmaps
    print("\n> Generating CIFAKE Grad-CAM heatmaps...")
    gradcam_test = CIFAKEDataset(
        root=str(Path(config["cifake"]["data_dir"]) / "test"),
        transform=get_cifake_val_transforms(config["cifake"]["image_size"]),
        return_path=True,
    )
    if smoke_test:
        gradcam_test = Subset(gradcam_test, range(min(20, len(gradcam_test))))
    gcam_loader = DataLoader(gradcam_test, batch_size=8, num_workers=0)
    gcam_gen = GradCAMGenerator(cifake_model, device=str(device))
    gcam_gen.save_heatmaps(
        gcam_loader,
        output_dir=str(results_dir / "cifake" / "heatmaps"),
        n_samples=config["explainability"]["n_samples"],
    )

    # ================================================================
    # PHASE 6: COMPARISON PLOTS
    # ================================================================
    print("\n" + "-" * 40)
    print("PHASE 6: GENERATING COMPARISON PLOTS")
    print("-" * 40)

    generate_all_comparison_plots(
        results_csv=results_csv,
        robustness_csv=rob_csv,
        output_dir=str(results_dir / "comparison"),
    )

    # ================================================================
    # DONE
    # ================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE - {elapsed:.1f}s total")
    print(f"Results: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIGI Detection - Full Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with tiny data")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, use existing checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_all(config, smoke_test=args.smoke_test, eval_only=args.eval_only)
