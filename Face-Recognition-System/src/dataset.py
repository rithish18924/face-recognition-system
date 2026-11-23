"""
src/dataset.py -- Dataset loading, validation, and splitting.

Supports two modes:
  1. LFW via sklearn.datasets.fetch_lfw_people (default).
  2. Custom directory organised as:  root/<person_name>/<image_file>

In both cases the output is a DatasetBundle namedtuple.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    """Holds raw pixel arrays + integer labels for train / val / test splits."""
    X_train: np.ndarray      # (N, H, W, 3) uint8 RGB
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray      # integer class indices
    y_val:   np.ndarray
    y_test:  np.ndarray
    label_encoder: LabelEncoder   # fitted encoder
    target_names: list[str]       # class names in index order
    image_shape: tuple[int, int]  # (H, W)
    summary: dict = field(default_factory=dict)


# ─── LFW loader ──────────────────────────────────────────────────────────────

def load_lfw(
    min_faces_per_person: int = 20,
    color: bool = True,
    resize: float = 0.5,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_seed: int = 42,
    data_home: Optional[str] = None,
) -> DatasetBundle:
    """
    Download (or load from cache) LFW, apply stratified splits, and return a
    DatasetBundle.

    Parameters
    ----------
    min_faces_per_person : int
        Minimum images per identity.  Increase to reduce class count.
    color : bool
        Load colour images (recommended).
    resize : float
        Downsample factor applied by sklearn (0.5 -> ~62×47 px).
    test_size / val_size : float
        Proportions for test and validation splits.
    random_seed : int
        Fixed seed for all random operations.
    data_home : str | None
        Override sklearn's default cache directory.
    """
    logger.info(
        "Fetching LFW dataset (min_faces_per_person=%d, color=%s, resize=%.2f) ...",
        min_faces_per_person, color, resize,
    )

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        color=color,
        resize=resize,
        data_home=data_home,
    )

    X = lfw.images          # (N, H, W) or (N, H, W, 3)
    y = lfw.target          # integer labels
    target_names = list(lfw.target_names)

    # Ensure colour axis
    if X.ndim == 3:
        X = np.stack([X, X, X], axis=-1)

    # sklearn returns float32 in [0, 1] for colour images -- convert to uint8
    if X.dtype != np.uint8:
        X = (np.clip(X, 0.0, 1.0) * 255).astype(np.uint8)

    H, W = X.shape[1], X.shape[2]
    n_total = len(X)
    n_classes = len(target_names)

    logger.info("Loaded %d images across %d identities (%dx%d px).", n_total, n_classes, H, W)

    # ── Fit label encoder (maps integer -> name) ───────────────────────────
    le = LabelEncoder()
    le.classes_ = np.array(target_names)   # already integer-indexed by sklearn

    # ── Stratified splits ─────────────────────────────────────────────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val,
        stratify=y_train_val,
        random_state=random_seed,
    )

    summary = {
        "source": "LFW (sklearn)",
        "total_images": n_total,
        "n_classes": n_classes,
        "image_shape_hw": (H, W),
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
        "min_faces_per_person": min_faces_per_person,
        "random_seed": random_seed,
    }

    logger.info(
        "Split -> train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=le,
        target_names=target_names,
        image_shape=(H, W),
        summary=summary,
    )


# ─── Custom directory loader ──────────────────────────────────────────────────

def load_custom_directory(
    root: str | Path,
    min_faces_per_person: int = 5,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_seed: int = 42,
) -> DatasetBundle:
    """
    Load a dataset organised as:
        root/
          <person_name>/
            img1.jpg
            img2.png
            ...

    Images are read as RGB uint8 arrays.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    logger.info("Scanning custom dataset directory: %s", root)

    images: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0
    empty_dirs = 0

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        img_paths = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if len(img_paths) < min_faces_per_person:
            empty_dirs += 1
            logger.debug("Skipping '%s': only %d image(s).", person_dir.name, len(img_paths))
            continue

        for p in img_paths:
            img = _safe_load_rgb(p)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(person_dir.name)

    if not images:
        raise ValueError(f"No usable images found under {root}.")

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(np.array(labels))
    target_names = list(le.classes_)

    # Stack images (may be different sizes -> resize to common shape)
    # Use the mode size for resizing
    shapes = [img.shape[:2] for img in images]
    H, W = max(set(shapes), key=shapes.count)
    X = np.stack([
        _resize_to(img, H, W) for img in images
    ], axis=0).astype(np.uint8)

    n_total = len(X)
    n_classes = len(target_names)
    logger.info("Loaded %d images across %d identities. Skipped %d.", n_total, n_classes, skipped)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_int,
        test_size=test_size,
        stratify=y_int,
        random_state=random_seed,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val,
        stratify=y_train_val,
        random_state=random_seed,
    )

    summary = {
        "source": str(root),
        "total_images": n_total,
        "n_classes": n_classes,
        "image_shape_hw": (H, W),
        "n_train": len(X_train),
        "n_val":   len(X_val),
        "n_test":  len(X_test),
        "skipped_images": skipped,
        "empty_dirs": empty_dirs,
        "random_seed": random_seed,
    }

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=le,
        target_names=target_names,
        image_shape=(H, W),
        summary=summary,
    )


# ─── Dataset summary printer ─────────────────────────────────────────────────

def print_dataset_summary(bundle: DatasetBundle) -> None:
    """Log a human-readable summary of the dataset bundle."""
    s = bundle.summary
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("  Source            : %s", s.get("source", "unknown"))
    logger.info("  Total images      : %d", s.get("total_images", 0))
    logger.info("  Identities        : %d", s.get("n_classes", 0))
    logger.info("  Image shape (H×W) : %s", s.get("image_shape_hw", "unknown"))
    logger.info("  Train / Val / Test: %d / %d / %d",
                s.get("n_train", 0), s.get("n_val", 0), s.get("n_test", 0))
    if "skipped_images" in s:
        logger.info("  Skipped (corrupt) : %d", s["skipped_images"])
    logger.info("  Random seed       : %s", s.get("random_seed", "?"))
    logger.info("=" * 60)

    # Per-class counts in training set
    unique, counts = np.unique(bundle.y_train, return_counts=True)
    logger.info("Per-class training counts (top 10 / bottom 5):")
    order = np.argsort(-counts)
    for idx in order[:10]:
        logger.info("  %-30s  %4d", bundle.target_names[unique[idx]], counts[idx])
    if len(order) > 10:
        logger.info("  ...")
        for idx in order[-5:]:
            logger.info("  %-30s  %4d", bundle.target_names[unique[idx]], counts[idx])


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _safe_load_rgb(path: Path) -> Optional[np.ndarray]:
    """Load an image as RGB uint8; return None on failure."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _resize_to(img: np.ndarray, H: int, W: int) -> np.ndarray:
    import cv2  # type: ignore
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
