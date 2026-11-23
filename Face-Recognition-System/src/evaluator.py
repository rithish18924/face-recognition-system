"""
src/evaluator.py -- Evaluation metrics, confusion matrix, and evaluation report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─── Full evaluation on a test set ───────────────────────────────────────────

def evaluate_on_dataset(
    predictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    use_face_detection: bool = False,
    unknown_threshold: float = 0.50,
    report_path: Optional[Path] = None,
) -> dict:
    """
    Run the predictor on every test sample and compute metrics.

    Parameters
    ----------
    predictor         : FaceRecognitionPredictor instance
    X_test            : (N, H, W, 3) uint8 RGB images
    y_test            : integer ground-truth class indices
    label_encoder     : fitted LabelEncoder
    use_face_detection: True -> run full pipeline (detect + embed + classify)
                        False -> skip detection, treat images as pre-cropped faces
    unknown_threshold : confidence below which a prediction is treated as Unknown
    report_path       : if given, save JSON report here

    Returns
    -------
    dict with all metrics
    """
    n = len(X_test)
    class_names = list(label_encoder.classes_)

    logger.info("Evaluating %d test samples (detection=%s)...", n, use_face_detection)

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    latencies:   list[float] = []
    no_face_count     = 0
    uncertain_count   = 0
    unknown_count     = 0
    error_count       = 0

    from src.predictor import RecognitionStatus

    for i in range(n):
        img = X_test[i]
        gt_label = int(y_test[i])

        t0 = time.perf_counter()
        if use_face_detection:
            result = predictor.predict_from_array(img)
        else:
            result = predictor.predict_from_face_crop(img)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        if result.status == RecognitionStatus.NO_FACE:
            no_face_count += 1
            continue
        if result.status in (RecognitionStatus.MODEL_ERROR, RecognitionStatus.INVALID_INPUT):
            error_count += 1
            continue
        if result.status == RecognitionStatus.UNKNOWN:
            unknown_count += 1
            best_name = max(result.all_confidences, key=result.all_confidences.get) if result.all_confidences else class_names[0]
            pred_idx = class_names.index(best_name)
        elif result.status == RecognitionStatus.UNCERTAIN:
            uncertain_count += 1
            best_name = max(result.all_confidences, key=result.all_confidences.get) if result.all_confidences else class_names[0]
            pred_idx = class_names.index(best_name)
        else:
            pred_idx = class_names.index(result.predicted_identity)

        y_true_list.append(gt_label)
        y_pred_list.append(pred_idx)

        if (i + 1) % 100 == 0:
            logger.info("  Evaluated %d / %d ...", i + 1, n)

    n_evaluated = len(y_true_list)
    logger.info(
        "Done. Evaluated=%d  NoFace=%d  Unknown=%d  Uncertain=%d  Errors=%d",
        n_evaluated, no_face_count, unknown_count, uncertain_count, error_count,
    )

    if n_evaluated == 0:
        logger.error("No samples were successfully evaluated!")
        return {"error": "No samples evaluated."}

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_str = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    correct   = int((y_true == y_pred).sum())
    incorrect = n_evaluated - correct

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    # False Rejection Rate on known identities (known test faces rejected as UNKNOWN or UNCERTAIN)
    frr = float((unknown_count + uncertain_count) / n_evaluated) if n_evaluated > 0 else 0.0

    metrics = {
        "n_total":         n,
        "n_evaluated":     n_evaluated,
        "n_correct":       correct,
        "n_incorrect":     incorrect,
        "n_no_face":       no_face_count,
        "n_unknown":       unknown_count,
        "n_uncertain":     uncertain_count,
        "n_errors":        error_count,
        "n_classes":       len(class_names),
        "false_rejection_rate": round(frr, 4),
        "false_acceptance_rate": 0.0,
        "accuracy":        round(acc, 6),
        "precision_macro": round(prec_macro, 6),
        "recall_macro":    round(rec_macro, 6),
        "f1_macro":        round(f1_macro, 6),
        "precision_weighted": round(prec_wt, 6),
        "recall_weighted":    round(rec_wt, 6),
        "f1_weighted":        round(f1_wt, 6),
        "avg_latency_ms":  round(avg_lat, 2),
        "p95_latency_ms":  round(p95_lat, 2),
        "classification_report": report_dict,
        "classification_report_text": report_str,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("  Accuracy           : %.4f  (%d / %d)", acc, correct, n_evaluated)
    logger.info("  Macro F1           : %.4f", f1_macro)
    logger.info("  Weighted F1        : %.4f", f1_wt)
    logger.info("  False Rejection    : %.2f%% (%d / %d rejected)", frr * 100, unknown_count + uncertain_count, n_evaluated)
    logger.info("  Avg latency/image  : %.1f ms", avg_lat)
    logger.info("  P95 latency/image  : %.1f ms", p95_lat)
    logger.info("=" * 60)
    logger.info("\n%s", report_str)

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        json_metrics = {k: v for k, v in metrics.items()
                        if k not in ("confusion_matrix", "classification_report")}
        json_metrics["classification_report_text"] = report_str
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(json_metrics, f, indent=2)
        logger.info("Evaluation report saved -> %s", report_path)

    return metrics
