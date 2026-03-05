"""
Model comparison and visualization.

Generates publication-quality plots comparing the three detection
methods across metrics and robustness conditions.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metrics: list = None,
    output_path: str = "results/comparison/metric_comparison.png",
) -> None:
    """Bar chart comparing methods across metrics.

    Args:
        results_df: DataFrame with columns "method", "accuracy", "f1", "auc".
        metrics: List of metric columns to plot.
        output_path: Output file path.
    """
    if metrics is None:
        metrics = ["accuracy", "f1", "auc"]

    # Filter to clean (no perturbation) results
    if "perturbation" in results_df.columns:
        clean = results_df[results_df["perturbation"].isna() |
                           (results_df["perturbation"] == "none")]
    else:
        clean = results_df

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, metric in zip(axes, metrics):
        if metric not in clean.columns:
            continue
        methods = clean["method"].tolist()
        values = clean[metric].astype(float).tolist()

        bars = ax.bar(methods, values, color=colors[:len(methods)], edgecolor="white")
        ax.set_title(metric.upper(), fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
            )

    fig.suptitle("Detection Method Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric comparison to {output_path}")


def plot_roc_curves(
    roc_data: dict,
    output_path: str = "results/comparison/roc_curves.png",
) -> None:
    """Plot overlaid ROC curves for all methods.

    Args:
        roc_data: Dict[method_name, {"fpr": array, "tpr": array, "auc": float}].
        output_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"cifake": "#2196F3", "aide": "#FF9800", "zed": "#4CAF50"}

    for method, data in roc_data.items():
        ax.plot(
            data["fpr"], data["tpr"],
            color=colors.get(method, "gray"),
            linewidth=2,
            label=f'{method.upper()} (AUC = {data["auc"]:.3f})',
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves - Method Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves to {output_path}")


def plot_robustness_heatmap(
    robustness_df: pd.DataFrame,
    metric: str = "accuracy",
    output_path: str = "results/comparison/robustness_heatmap.png",
) -> None:
    """Heatmap of method performance under perturbations.

    Args:
        robustness_df: DataFrame with "method", "perturbation", and metric columns.
        metric: Which metric to display (default "accuracy").
        output_path: Output file path.
    """
    if metric not in robustness_df.columns:
        print(f"Warning: metric '{metric}' not found in robustness data")
        return

    # Pivot for heatmap
    pivot = robustness_df.pivot_table(
        index="perturbation", columns="method", values=metric, aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.6 + 2)))
    sns.heatmap(
        pivot.astype(float),
        annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": metric.upper()},
    )
    ax.set_title(
        f"Robustness - {metric.upper()} Under Perturbations",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Perturbation")
    ax.set_xlabel("Method")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robustness heatmap to {output_path}")


def generate_all_comparison_plots(
    results_csv: str,
    robustness_csv: str = None,
    roc_data: dict = None,
    output_dir: str = "results/comparison",
) -> None:
    """Generate all comparison plots from CSV data.

    Args:
        results_csv: Path to main results CSV.
        robustness_csv: Optional path to robustness results CSV.
        roc_data: Optional ROC curve data dict.
        output_dir: Output directory for plots.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Main metrics
    df = pd.read_csv(results_csv)
    plot_metric_comparison(df, output_path=str(out / "metric_comparison.png"))

    # ROC curves
    if roc_data:
        plot_roc_curves(roc_data, output_path=str(out / "roc_curves.png"))

    # Robustness
    if robustness_csv and Path(robustness_csv).exists():
        rob_df = pd.read_csv(robustness_csv)
        plot_robustness_heatmap(
            rob_df, metric="accuracy",
            output_path=str(out / "robustness_accuracy.png"),
        )
        plot_robustness_heatmap(
            rob_df, metric="auc",
            output_path=str(out / "robustness_auc.png"),
        )

    print(f"\nAll comparison plots saved to {out}")
