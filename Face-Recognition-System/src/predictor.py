"""
src/predictor.py -- End-to-end inference pipeline.

Takes an RGB image (or file path), detects the face, extracts the embedding,
and returns a structured PredictionResult.

The classifier and label encoder are loaded once and reused across calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

from src.face_detector import FaceDetectorBase, FaceDetection
from src.feature_extractor import FeatureExtractorBase

logger = logging.getLogger(__name__)


# ─── Result types ─────────────────────────────────────────────────────────────

class RecognitionStatus(Enum):
    SUCCESS        = auto()   # Known identity: Match meets validated acceptance criteria
    UNKNOWN        = auto()   # Unknown person: No sufficiently reliable match
    UNCERTAIN      = auto()   # Uncertain prediction: Evidence is insufficient to establish identity
    NO_FACE        = auto()   # No face found in image
    INVALID_INPUT  = auto()   # Image could not be read or is malformed
    MODEL_ERROR    = auto()   # Model / pipeline failure


@dataclass
class PredictionResult:
    """Structured output of a single recognition request."""
    status: RecognitionStatus
    predicted_identity: Optional[str]   # None when status != SUCCESS
    confidence: float                   # max-probability of the predicted class
    all_confidences: dict[str, float]   # class -> probability mapping
    face_box: Optional[tuple[int, int, int, int]]  # (x,y,w,h) in original image
    face_detection: Optional[FaceDetection]
    inference_time_ms: float
    error_message: str = ""

    @property
    def is_recognized(self) -> bool:
        return self.status == RecognitionStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "status":               self.status.name,
            "predicted_identity":   self.predicted_identity,
            "confidence":           round(self.confidence, 4),
            "all_confidences":      {k: round(v, 4) for k, v in self.all_confidences.items()},
            "face_box":             self.face_box,
            "inference_time_ms":    round(self.inference_time_ms, 2),
            "error_message":        self.error_message,
        }


# ─── Predictor ───────────────────────────────────────────────────────────────

class FaceRecognitionPredictor:
    """
    Full prediction pipeline: image -> face crop -> embedding -> identity.

    Includes open-set unknown rejection, margin verification, and feature-space
    prototype verification.
    """

    def __init__(
        self,
        detector:   FaceDetectorBase,
        extractor:  FeatureExtractorBase,
        classifier_pipeline,  # sklearn Pipeline with .predict_proba()
        label_encoder,        # sklearn LabelEncoder
        unknown_threshold: float = 0.40,
        acceptance_threshold: float = 0.65,
        min_margin: float = 0.15,
        class_centroids: Optional[np.ndarray] = None,
        min_cosine_similarity: float = 0.82,
        unknown_cosine_similarity: float = 0.80,
    ) -> None:
        self._detector   = detector
        self._extractor  = extractor
        self._pipeline   = classifier_pipeline
        self._le         = label_encoder
        self._unknown_threshold = unknown_threshold
        self._acceptance_threshold = acceptance_threshold
        self._min_margin = min_margin
        self._centroids  = class_centroids
        self._min_cosine_sim = min_cosine_similarity
        self._unknown_cosine_sim = unknown_cosine_similarity
        self._threshold  = unknown_threshold  # Backwards compatibility

        logger.info(
            "FaceRecognitionPredictor ready. Classes=%d  UnknownThresh=%.2f  AcceptThresh=%.2f  MarginThresh=%.2f  CentroidSimThresh=%.2f",
            len(self._le.classes_), self._unknown_threshold, self._acceptance_threshold, self._min_margin, self._min_cosine_sim,
        )

    @property
    def label_encoder(self):
        """Return the fitted LabelEncoder."""
        return self._le

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_from_path(self, image_path: str | Path) -> PredictionResult:
        """Load image from disk and run recognition."""
        import cv2  # type: ignore
        path = Path(image_path)
        if not path.exists():
            return self._error(f"File not found: {path}", RecognitionStatus.INVALID_INPUT)

        img = cv2.imread(str(path))
        if img is None:
            return self._error(f"Cannot read image: {path}", RecognitionStatus.INVALID_INPUT)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.predict_from_array(img_rgb)

    def predict_from_array(self, image_rgb: np.ndarray) -> PredictionResult:
        """Run recognition on a (H, W, 3) uint8 RGB numpy array."""
        t0 = time.perf_counter()

        # ── 1. Validate input ────────────────────────────────────────────
        if image_rgb is None or not isinstance(image_rgb, np.ndarray):
            return self._error("Input is not a numpy array.", RecognitionStatus.INVALID_INPUT)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return self._error(
                f"Expected (H,W,3) array, got shape {image_rgb.shape}.",
                RecognitionStatus.INVALID_INPUT,
            )

        # ── 2. Face detection ────────────────────────────────────────────
        detection = self._detector.detect(image_rgb)
        if not detection.detected:
            elapsed = (time.perf_counter() - t0) * 1000
            return PredictionResult(
                status=RecognitionStatus.NO_FACE,
                predicted_identity=None,
                confidence=0.0,
                all_confidences={},
                face_box=None,
                face_detection=detection,
                inference_time_ms=elapsed,
                error_message=detection.error_message or "No face detected in image.",
            )

        # ── 3. Feature extraction ────────────────────────────────────────
        try:
            embedding = self._extractor.extract(detection.face_crop)
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                f"Embedding extraction failed: {exc}",
                RecognitionStatus.MODEL_ERROR,
                elapsed_ms=elapsed,
            )

        if not self._extractor.validate_embedding(embedding):
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                "Embedding contains non-finite values.",
                RecognitionStatus.MODEL_ERROR,
                elapsed_ms=elapsed,
            )

        # ── 4. Classification & Verification ─────────────────────────────
        return self._classify_and_verify(embedding, detection, t0)

    def predict_from_face_crop(self, face_crop_rgb_or_emb: np.ndarray) -> PredictionResult:
        """
        Run recognition on an already-cropped face or pre-extracted embedding.
        Useful when LFW images are pre-cropped or embeddings are pre-computed.
        """
        t0 = time.perf_counter()

        if face_crop_rgb_or_emb is None:
            return self._error("Invalid face crop input.", RecognitionStatus.INVALID_INPUT)

        if face_crop_rgb_or_emb.ndim == 1:
            # Pre-extracted 1D embedding
            embedding = face_crop_rgb_or_emb
        elif face_crop_rgb_or_emb.ndim == 3:
            try:
                embedding = self._extractor.extract(face_crop_rgb_or_emb)
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                return self._error(
                    f"Embedding extraction failed: {exc}", RecognitionStatus.MODEL_ERROR, elapsed_ms=elapsed
                )
        else:
            return self._error(
                f"Expected 1D embedding or 3D image, got shape {face_crop_rgb_or_emb.shape}.",
                RecognitionStatus.INVALID_INPUT,
            )

        if not self._extractor.validate_embedding(embedding):
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                "Embedding contains non-finite values.",
                RecognitionStatus.MODEL_ERROR,
                elapsed_ms=elapsed,
            )

        return self._classify_and_verify(embedding, None, t0)

    def _classify_and_verify(
        self,
        embedding: np.ndarray,
        detection: Optional[FaceDetection],
        t0: float,
    ) -> PredictionResult:
        """Classify embedding and apply unknown rejection / margin acceptance criteria."""
        try:
            proba = self._pipeline.predict_proba(embedding.reshape(1, -1))[0]
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return self._error(
                f"Classifier failed: {exc}",
                RecognitionStatus.MODEL_ERROR,
                elapsed_ms=elapsed,
            )

        class_names = list(self._le.classes_)
        all_conf = {name: float(p) for name, p in zip(class_names, proba)}

        sorted_indices = np.argsort(proba)[::-1]
        best_idx = int(sorted_indices[0])
        best_prob = float(proba[best_idx])
        best_name = class_names[best_idx]
        second_prob = float(proba[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = best_prob - second_prob

        # Centroid cosine similarity if class prototypes are loaded
        centroid_sim = None
        if self._centroids is not None and best_idx < len(self._centroids):
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
            centroid_sim = float(np.dot(emb_norm, self._centroids[best_idx]))

        elapsed = (time.perf_counter() - t0) * 1000
        box = detection.box if detection else None

        # ── Decision Logic ───────────────────────────────────────────────
        # Criteria 1: Unknown Person Rejection
        # A high softmax probability does NOT guarantee identity in closed-set
        # classifiers when tested on open-set (unknown) faces.
        # Rejected as UNKNOWN if:
        #   - Top probability is below unknown rejection threshold, OR
        #   - Feature space prototype similarity is below unknown threshold
        prob_too_low = (best_prob < self._unknown_threshold)
        sim_too_low = (centroid_sim is not None and centroid_sim < self._unknown_cosine_sim)
        is_unknown = prob_too_low or sim_too_low

        if is_unknown:
            reasons = []
            if prob_too_low:
                reasons.append(f"probability ({best_prob:.1%}) is below threshold ({self._unknown_threshold:.1%})")
            if sim_too_low:
                reasons.append(f"prototype similarity ({centroid_sim:.2f}) is below unknown cutoff ({self._unknown_cosine_sim:.2f})")
            reason_str = " and ".join(reasons) if reasons else f"criteria below ({self._unknown_threshold:.1%})"
            sim_str = f", prototype similarity {centroid_sim:.2f}" if centroid_sim is not None else ""
            return PredictionResult(
                status=RecognitionStatus.UNKNOWN,
                predicted_identity=None,
                confidence=best_prob,
                all_confidences=all_conf,
                face_box=box,
                face_detection=detection,
                inference_time_ms=elapsed,
                error_message=(
                    f"Unknown person: Match confidence ({best_prob:.1%}{sim_str}) rejected because {reason_str}. "
                    f"Closest candidate: {best_name}."
                ),
            )

        # Criteria 2: Uncertain Prediction
        # Flagged as UNCERTAIN if:
        #   - Confidence is between unknown and acceptance thresholds, OR
        #   - Margin between #1 and #2 candidate is too narrow (ambiguous match), OR
        #   - Prototype similarity to the top candidate is below verified acceptance criteria
        prob_uncertain = (best_prob < self._acceptance_threshold)
        margin_too_narrow = (margin < self._min_margin)
        sim_uncertain = (centroid_sim is not None and centroid_sim < self._min_cosine_sim)
        is_uncertain = prob_uncertain or margin_too_narrow or sim_uncertain

        if is_uncertain:
            reasons = []
            if prob_uncertain:
                reasons.append(f"probability ({best_prob:.1%}) is below acceptance threshold ({self._acceptance_threshold:.1%})")
            if margin_too_narrow:
                reasons.append(f"margin ({margin:.1%}) is below minimum margin ({self._min_margin:.1%})")
            if sim_uncertain:
                reasons.append(f"prototype similarity ({centroid_sim:.2f}) is below acceptance criteria ({self._min_cosine_sim:.2f})")
            reason_str = "; ".join(reasons) if reasons else f"margin or prototype similarity is below acceptance criteria ({self._acceptance_threshold:.1%})"
            sim_str = f", prototype similarity {centroid_sim:.2f}" if centroid_sim is not None else ""
            return PredictionResult(
                status=RecognitionStatus.UNCERTAIN,
                predicted_identity=None,
                confidence=best_prob,
                all_confidences=all_conf,
                face_box=box,
                face_detection=detection,
                inference_time_ms=elapsed,
                error_message=(
                    f"Uncertain prediction: Candidate is {best_name} ({best_prob:.1%}{sim_str}), "
                    f"but {reason_str}."
                ),
            )

        # Criteria 3: Verified Known Identity
        return PredictionResult(
            status=RecognitionStatus.SUCCESS,
            predicted_identity=best_name,
            confidence=best_prob,
            all_confidences=all_conf,
            face_box=box,
            face_detection=detection,
            inference_time_ms=elapsed,
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _error(
        msg: str,
        status: RecognitionStatus,
        elapsed_ms: float = 0.0,
    ) -> PredictionResult:
        logger.warning("Prediction error (%s): %s", status.name, msg)
        return PredictionResult(
            status=status,
            predicted_identity=None,
            confidence=0.0,
            all_confidences={},
            face_box=None,
            face_detection=None,
            inference_time_ms=elapsed_ms,
            error_message=msg,
        )
