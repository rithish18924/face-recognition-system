"""
evaluate.py -- Face Recognition Evaluation Pipeline.

Computes accuracy, macro/weighted F1, confusion matrix, per-class F1 chart,
and evaluates open-set unknown person rejection.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.evaluator import evaluate_on_dataset
from src.visualization import plot_confusion_matrix, plot_per_class_f1
from src.utils import get_logger, load_pickle
from app import get_predictor

logger = get_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Face Recognition System")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.UNKNOWN_THRESHOLD,
        help="Confidence threshold for unknown rejection",
    )
    parser.add_argument(
        "--use-detection",
        action="store_true",
        help="Run full MTCNN detector on test images instead of pre-cropped images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FACE RECOGNITION -- EVALUATION PIPELINE")
    logger.info("=" * 60)

    logger.info("Loading model ...")
    predictor = get_predictor()
    label_encoder = load_pickle(config.LABEL_ENCODER_PATH)
    class_names = list(label_encoder.classes_)

    test_emb_path = config.ARTIFACTS_DIR / "test_embeddings.npz"
    emb_cache_path = config.EMBEDDING_CACHE_PATH

    if test_emb_path.exists():
        logger.info("Loading test embeddings from %s ...", test_emb_path)
        data = np.load(test_emb_path)
        X_test = data["X_test"]
        y_test = data["y_test"]
    elif emb_cache_path.exists():
        logger.info("Loading test embeddings from %s ...", emb_cache_path)
        data = np.load(emb_cache_path)
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        raise FileNotFoundError("Test embeddings not found. Run  python train.py  first.")

    logger.info("Test samples: %d", len(X_test))

    report = evaluate_on_dataset(
        predictor=predictor,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        use_face_detection=args.use_detection,
        unknown_threshold=args.threshold,
        report_path=config.EVAL_REPORT_PATH,
    )

    # Confusion matrix
    cm = np.array(report["confusion_matrix"])
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=config.CONFUSION_MATRIX_PATH,
    )

    # Per-class F1 plot
    plot_per_class_f1(
        report_dict=report["classification_report"],
        class_names=class_names,
        save_path=config.METRICS_PLOT_PATH,
    )

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("  Accuracy           : %.4f", report["accuracy"])
    logger.info("  Macro F1           : %.4f", report.get("f1_macro", 0.0))
    logger.info("  Weighted F1        : %.4f", report.get("f1_weighted", 0.0))
    logger.info("  Test samples       : %d", report.get("n_total", 0))
    logger.info("  Correct            : %d", report.get("n_correct", 0))
    logger.info("  Incorrect          : %d", report.get("n_incorrect", 0))
    logger.info("  Avg latency        : %.1f ms/image", report.get("avg_latency_ms", 0.0))
    logger.info("Artifacts:")
    logger.info("  Confusion matrix   -> %s", config.CONFUSION_MATRIX_PATH)
    logger.info("  F1 chart           -> %s", config.METRICS_PLOT_PATH)
    logger.info("  Eval report (JSON) -> %s", config.EVAL_REPORT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
