"""Plots for training curves, confusion matrices and per-class metrics."""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.config import MINORITY_LABELS  # noqa: E402

_TRAIN_COLOR = "#0072B2"
_VAL_COLOR = "#E69F00"


def plot_training_curves(history: list, save_path: Path, title: str) -> None:
    """Plot loss and F1 per epoch in two panels."""
    epochs = [h.epoch for h in history]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, [h.train_loss for h in history], color=_TRAIN_COLOR, marker="o", markersize=4, linewidth=2, label="train")
    ax_loss.plot(epochs, [h.val_loss for h in history], color=_VAL_COLOR, marker="o", markersize=4, linewidth=2, label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(frameon=False)

    ax_f1.plot(epochs, [h.train_macro_f1 for h in history], color=_TRAIN_COLOR, linestyle="-", marker="o", markersize=4, linewidth=2, label="train macro F1")
    ax_f1.plot(epochs, [h.val_macro_f1 for h in history], color=_VAL_COLOR, linestyle="-", marker="o", markersize=4, linewidth=2, label="val macro F1")
    ax_f1.plot(epochs, [h.train_micro_f1 for h in history], color=_TRAIN_COLOR, linestyle="--", marker="s", markersize=4, linewidth=1.5, label="train micro F1")
    ax_f1.plot(epochs, [h.val_micro_f1 for h in history], color=_VAL_COLOR, linestyle="--", marker="s", markersize=4, linewidth=1.5, label="val micro F1")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("F1")
    ax_f1.set_ylim(0, 1)
    ax_f1.set_xticks(epochs)
    ax_f1.set_title("Macro F1 / Micro F1")
    ax_f1.grid(True, alpha=0.25)
    ax_f1.legend(frameon=False, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def plot_per_class_f1_curves(history: list, labels: list[str], save_path: Path, title: str) -> None:
    """Plot per-class F1 per epoch, one subplot per class."""
    epochs = [h.epoch for h in history]

    ncols = min(5, len(labels))
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows + 0.6), sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, label in enumerate(labels):
        ax = axes[i]
        ax.plot(epochs, [h.train_f1_per_class[label] for h in history], color=_TRAIN_COLOR, marker="o", markersize=3, linewidth=1.5, label="train")
        ax.plot(epochs, [h.val_f1_per_class[label] for h in history], color=_VAL_COLOR, marker="o", markersize=3, linewidth=1.5, label="val")
        subtitle = f"{label}\n(minority)" if label in MINORITY_LABELS else label
        ax.set_title(subtitle, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    for ax in axes[len(labels):]:
        ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", frameon=False, ncols=2)
    fig.supxlabel("epoch")
    fig.supylabel("F1")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def plot_per_class_f1_bar(report: dict, labels: list[str], save_path: Path, title: str) -> None:
    """Bar chart of per-class F1 from a classification report."""
    scores = [report[label]["f1-score"] for label in labels]
    colors = [_VAL_COLOR if label in MINORITY_LABELS else _TRAIN_COLOR for label in labels]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(labels) + 2), 4.5))
    bars = ax.bar(range(len(labels)), scores, color=colors)
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        handles=[Patch(color=_TRAIN_COLOR, label="majority"), Patch(color=_VAL_COLOR, label="minority")],
        frameon=False, fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: list[list[int]] | np.ndarray,
    labels: list[str],
    save_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    """Confusion-matrix heatmap."""
    matrix = np.array(confusion_matrix, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n + 2), max(5.0, 0.6 * n + 1)))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8,
                color="white" if value > threshold else "black",
            )

    fig.colorbar(im, ax=ax, label="proportion of true-class row" if normalize else "count")
    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def plot_similarity_matrix(
    matrix: list[list[float]] | np.ndarray,
    labels: list[str],
    save_path: Path,
    title: str,
    cbar_label: str = "Jaccard index",
) -> None:
    """Symmetric class x class similarity heatmap."""
    matrix = np.array(matrix, dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n + 2), max(5.0, 0.6 * n + 1)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)


def plot_cv_search(cv_results: dict, best_C: float, save_path: Path, title: str) -> None:
    """Plot mean CV macro-F1 against C for the SVM grid search."""
    Cs = sorted(cv_results)
    scores = [cv_results[c] for c in Cs]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(Cs, scores, color=_TRAIN_COLOR, marker="o", markersize=5, linewidth=2)
    ax.axvline(best_C, color=_VAL_COLOR, linestyle="--", linewidth=1.5, label=f"best C={best_C}")
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("mean CV macro F1")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(save_path), dpi=150)
    plt.close(fig)
