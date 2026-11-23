"""
src/trainer.py -- Classifier training on top of pre-computed face embeddings.

Trains and compares multiple classifier types; selects the best on validation
accuracy; saves the model and label encoder.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.utils import save_pickle

logger = logging.getLogger(__name__)


# ─── Training result ─────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Result of a single classifier training run."""
    classifier_name: str
    pipeline: Pipeline        # scaler + classifier (or calibrated wrapper)
    val_accuracy: float
    train_accuracy: float
    train_time_seconds: float
    n_classes: int
    embedding_dim: int


# ─── Individual trainer helpers ───────────────────────────────────────────────

# ─── Centroid computation ────────────────────────────────────────────────────

def compute_class_centroids(X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute L2-normalized class prototypes (centroids) in embedding space."""
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_train)
    centroids = np.zeros((n_classes, X_train.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.any():
            centroids[c] = np.mean(X_norm[mask], axis=0)
    return normalize(centroids)


# ─── Individual trainer helpers ───────────────────────────────────────────────

def _train_one(
    name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> TrainResult:
    """Wrap a classifier in a StandardScaler pipeline, train, and evaluate."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    train_acc = pipeline.score(X_train, y_train)
    val_acc   = pipeline.score(X_val, y_val)

    logger.info(
        "  %-30s  train=%.4f  val=%.4f  time=%.1fs",
        name, train_acc, val_acc, elapsed,
    )

    return TrainResult(
        classifier_name=name,
        pipeline=pipeline,
        val_accuracy=val_acc,
        train_accuracy=train_acc,
        train_time_seconds=elapsed,
        n_classes=n_classes,
        embedding_dim=X_train.shape[1],
    )


# ─── Main training function ──────────────────────────────────────────────────

def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_encoder: LabelEncoder,
    random_seed: int = 42,
    classifier_type: str = "svm_linear",
) -> TrainResult:
    """
    Train one or more classifiers on pre-computed embeddings and return the
    best-performing one on the validation set.

    Parameters
    ----------
    X_train / X_val : (N, embedding_dim) float32 embeddings
    y_train / y_val : integer class indices
    label_encoder   : fitted LabelEncoder
    random_seed     : for reproducibility
    classifier_type : "svm_linear" | "svm_rbf" | "logistic_regression" |
                      "random_forest" | "all"
    """
    n_classes = len(label_encoder.classes_)
    dim = X_train.shape[1]

    logger.info("=" * 60)
    logger.info("TRAINING CLASSIFIER(S)")
    logger.info("  Embeddings  : %d-dim", dim)
    logger.info("  Train size  : %d", len(X_train))
    logger.info("  Val size    : %d", len(X_val))
    logger.info("  Classes     : %d", n_classes)
    logger.info("=" * 60)

    # ── Candidate classifiers with class_weight='balanced' ───────────────
    candidates: dict[str, object] = {}

    if classifier_type in ("svm_linear", "all"):
        svm = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=10000, class_weight="balanced", random_state=random_seed),
            cv=3,
            method="sigmoid",
        )
        candidates["LinearSVC (calibrated)"] = svm

    if classifier_type in ("logistic_regression", "all"):
        candidates["LogisticRegression"] = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_seed,
            solver="lbfgs",
        )

    if classifier_type in ("svm_rbf", "all"):
        candidates["RBF-SVM"] = SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_seed,
        )

    if classifier_type in ("random_forest", "all"):
        candidates["RandomForest"] = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )

    if not candidates:
        raise ValueError(f"No classifier for type={classifier_type!r}.")

    results: list[TrainResult] = []
    for name, clf in candidates.items():
        result = _train_one(name, clf, X_train, y_train, X_val, y_val, n_classes)
        results.append(result)

    best = max(results, key=lambda r: r.val_accuracy)
    logger.info("Best classifier: %s (val_acc=%.4f)", best.classifier_name, best.val_accuracy)

    return best


# ─── Save / load ─────────────────────────────────────────────────────────────

def save_model_artifact(
    result: TrainResult,
    label_encoder: LabelEncoder,
    extractor_strategy: str,
    model_path: Path,
    encoder_path: Path,
    metadata: Optional[dict] = None,
    centroids: Optional[np.ndarray] = None,
) -> dict:
    """
    Persist the trained pipeline, label encoder, and optional class centroids.

    Returns a metadata dict that documents the artifact.
    """
    model_path = Path(model_path)
    encoder_path = Path(encoder_path)

    save_pickle(result.pipeline, model_path)
    save_pickle(label_encoder, encoder_path)

    meta = {
        "classifier_name":    result.classifier_name,
        "val_accuracy":       round(result.val_accuracy, 6),
        "train_accuracy":     round(result.train_accuracy, 6),
        "train_time_seconds": round(result.train_time_seconds, 2),
        "n_classes":          result.n_classes,
        "embedding_dim":      result.embedding_dim,
        "extractor_strategy": extractor_strategy,
        "model_path":         str(model_path),
        "encoder_path":       str(encoder_path),
    }

    if centroids is not None:
        centroids_path = model_path.parent / "centroids.npy"
        np.save(centroids_path, centroids)
        meta["centroids_path"] = str(centroids_path)
        logger.info("Saved centroids -> %s", centroids_path)

    if metadata:
        meta.update(metadata)

    meta_path = model_path.parent / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved model  -> %s", model_path)
    logger.info("Saved encoder -> %s", encoder_path)
    logger.info("Saved metadata -> %s", meta_path)
    return meta
