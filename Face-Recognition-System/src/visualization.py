"""
src/visualization.py -- Confusion matrix, per-class metrics, and prediction panels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

logger = logging.getLogger(__name__)


# ─── Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Optional[Path] = None,
    figsize_per_class: float = 0.45,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a labeled confusion matrix.

    For large class counts the figure is auto-sized so labels remain readable.
    """
    n = len(class_names)
    fig_size = max(8, n * figsize_per_class)
    fig, ax = plt.subplots(figsize=(fig_size + 2, fig_size))

    # Normalised row-wise (recall per class)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=14, pad=12)

    tick_marks = np.arange(n)
    fontsize = max(4, min(9, 200 // n))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)

    # Annotate cells if class count is small enough
    if n <= 20:
        thresh = 0.5
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=fontsize)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved -> %s", save_path)

    return fig


# ─── Per-class F1 bar chart ──────────────────────────────────────────────────

def plot_per_class_f1(
    report_dict: dict,
    class_names: list[str],
    save_path: Optional[Path] = None,
    top_n: int = 40,
) -> plt.Figure:
    """
    Horizontal bar chart of per-class F1 scores, sorted descending.
    Shows only the top *top_n* classes when there are many.
    """
    f1_scores = []
    if isinstance(report_dict, dict):
        for name in class_names:
            if name in report_dict and isinstance(report_dict[name], dict):
                f1_scores.append(report_dict[name].get("f1-score", 0.0))
            else:
                f1_scores.append(0.0)
    else:
        f1_scores = [0.0] * len(class_names)

    pairs = sorted(zip(class_names, f1_scores), key=lambda x: -x[1])[:top_n]
    names, scores = zip(*pairs) if pairs else ([], [])

    fig_h = max(4, len(names) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colors = ["#2ecc71" if s >= 0.7 else "#e67e22" if s >= 0.4 else "#e74c3c" for s in scores]
    bars = ax.barh(names, scores, color=colors, height=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_title(f"Per-Class F1 Scores (top {len(names)})", fontsize=13)
    ax.axvline(0.7, color="green",  linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.4, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", fontsize=7)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Per-class F1 plot saved -> %s", save_path)

    return fig


# ─── Prediction panel ────────────────────────────────────────────────────────

def plot_prediction_panel(
    images: list[np.ndarray],
    gt_names: list[str],
    pred_results: list,        # list[PredictionResult]
    save_path: Optional[Path] = None,
    cols: int = 5,
    cell_size: float = 3.0,
) -> plt.Figure:
    """
    Display a grid of test images annotated with ground-truth and prediction.

    Color coding:
      ✅ Green border  -- correct prediction
      ❌ Red border    -- incorrect prediction
      ⬜ Yellow border -- uncertain / unknown
      ⬛ Gray border   -- no face detected
    """
    from src.predictor import RecognitionStatus

    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * (cell_size + 0.6)))
    axes = np.array(axes).flatten()

    for i, (img, gt, res) in enumerate(zip(images, gt_names, pred_results)):
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")

        if res.status == RecognitionStatus.SUCCESS:
            pred_name = res.predicted_identity
            is_correct = (pred_name == gt)
            border_color = "#2ecc71" if is_correct else "#e74c3c"
            icon = "OK" if is_correct else "✗"
            label_text = (
                f"{icon} Pred: {pred_name}\n"
                f"GT: {gt}\n"
                f"Conf: {res.confidence:.1%}"
            )
            label_color = border_color
        elif res.status == RecognitionStatus.UNKNOWN:
            border_color = "#e74c3c"
            label_text = f"✕ Unknown\nGT: {gt}\nConf: {res.confidence:.1%}"
            label_color = border_color
        elif res.status == RecognitionStatus.UNCERTAIN:
            border_color = "#f39c12"
            label_text = f"? Uncertain\nGT: {gt}\nConf: {res.confidence:.1%}"
            label_color = border_color
        elif res.status == RecognitionStatus.NO_FACE:
            border_color = "#95a5a6"
            label_text = f"No face\nGT: {gt}"
            label_color = border_color
        else:
            border_color = "#8e44ad"
            label_text = f"Error\nGT: {gt}"
            label_color = border_color

        # Border via spine colour
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        ax.set_title(label_text, fontsize=7, color=label_color,
                     pad=2, loc="center", wrap=True)

        # Draw bounding box if available
        if res.face_box is not None:
            x, y, w, h = res.face_box
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor=border_color, facecolor="none",
            )
            ax.add_patch(rect)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Face Recognition -- Test Predictions", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Prediction panel saved -> %s", save_path)

    return fig
