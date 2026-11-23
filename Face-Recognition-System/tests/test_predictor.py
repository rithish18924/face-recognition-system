"""
tests/test_predictor.py -- Unit tests for the prediction pipeline.

Uses mocked detector / extractor / classifier so no real model download is needed.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import LabelEncoder
from src.predictor import (
    FaceRecognitionPredictor,
    PredictionResult,
    RecognitionStatus,
)
from src.face_detector import FaceDetection


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_label_encoder(names=("Alice", "Bob", "Carol")) -> LabelEncoder:
    le = LabelEncoder()
    le.classes_ = np.array(names)
    return le


def make_mock_detector(detected=True, confidence=0.99):
    """Return a mock FaceDetectorBase that always returns a fixed result."""
    face_crop = np.zeros((64, 64, 3), dtype=np.uint8)
    detection = FaceDetection(
        detected=detected,
        face_crop=face_crop if detected else None,
        box=(10, 10, 64, 64) if detected else None,
        confidence=confidence,
        n_faces_found=1 if detected else 0,
        error_message="" if detected else "No face detected.",
    )
    det = MagicMock()
    det.detect.return_value = detection
    return det


def make_mock_extractor(emb_dim=1280):
    ext = MagicMock()
    ext.embedding_dim = emb_dim
    ext.extract.return_value = np.ones(emb_dim, dtype=np.float32)
    ext.validate_embedding.return_value = True
    return ext


def make_mock_pipeline(n_classes=3, best_class=0):
    """Pipeline whose predict_proba puts most mass on *best_class*,
    with remaining probability spread evenly so all probs sum to 1.0."""
    proba = np.zeros(n_classes)
    proba[best_class] = 0.95
    remaining = (1.0 - 0.95) / max(1, n_classes - 1)
    for i in range(n_classes):
        if i != best_class:
            proba[i] = remaining
    pipe = MagicMock()
    pipe.predict_proba.return_value = np.array([proba])
    return pipe


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestPredictorSuccess:

    def setup_method(self):
        self.le       = make_label_encoder()
        self.detector = make_mock_detector(detected=True)
        self.extractor= make_mock_extractor()
        self.pipeline = make_mock_pipeline(n_classes=3, best_class=0)
        self.predictor = FaceRecognitionPredictor(
            self.detector, self.extractor, self.pipeline, self.le,
            unknown_threshold=0.50,
        )

    def test_success_status(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.SUCCESS

    def test_predicted_identity_correct(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        assert result.predicted_identity == "Alice"   # class 0

    def test_confidence_is_float(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_confidences_sum_to_one(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        total = sum(result.all_confidences.values())
        assert abs(total - 1.0) < 1e-5

    def test_face_box_present(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        assert result.face_box is not None
        assert len(result.face_box) == 4

    def test_inference_time_positive(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        assert result.inference_time_ms >= 0.0

    def test_to_dict_keys(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict_from_array(img)
        d = result.to_dict()
        for key in ("status", "predicted_identity", "confidence",
                    "all_confidences", "inference_time_ms", "error_message"):
            assert key in d


class TestPredictorNoFace:

    def test_no_face_status(self):
        le        = make_label_encoder()
        detector  = make_mock_detector(detected=False)
        extractor = make_mock_extractor()
        pipeline  = make_mock_pipeline()
        predictor = FaceRecognitionPredictor(detector, extractor, pipeline, le)

        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.NO_FACE
        assert result.predicted_identity is None
        assert result.confidence == 0.0


class TestPredictorUncertainAndUnknown:

    def test_uncertain_below_acceptance_threshold(self):
        le = make_label_encoder()
        detector = make_mock_detector(detected=True)
        extractor = make_mock_extractor()

        # Confidence is 0.50 (above unknown 0.35, below acceptance 0.65)
        proba = np.array([[0.50, 0.25, 0.25]])
        pipeline = MagicMock()
        pipeline.predict_proba.return_value = proba

        predictor = FaceRecognitionPredictor(
            detector, extractor, pipeline, le,
            unknown_threshold=0.35, acceptance_threshold=0.65, min_margin=0.15
        )
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.UNCERTAIN
        assert result.predicted_identity is None

    def test_uncertain_due_to_narrow_margin(self):
        le = make_label_encoder()
        detector = make_mock_detector(detected=True)
        extractor = make_mock_extractor()

        # Confidence is high (0.70) but runner-up is 0.60 -> margin is only 0.10 (< 0.15)
        proba = np.array([[0.52, 0.46, 0.02]])
        pipeline = MagicMock()
        pipeline.predict_proba.return_value = proba

        predictor = FaceRecognitionPredictor(
            detector, extractor, pipeline, le,
            unknown_threshold=0.35, acceptance_threshold=0.50, min_margin=0.15
        )
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.UNCERTAIN
        assert result.predicted_identity is None

    def test_unknown_person_below_unknown_threshold(self):
        le = make_label_encoder()
        detector = make_mock_detector(detected=True)
        extractor = make_mock_extractor()

        # Very low confidence (0.25) across 3 classes
        proba = np.array([[0.25, 0.25, 0.50]])  # best is 0.50, but if unknown_threshold=0.60 -> unknown
        pipeline = MagicMock()
        pipeline.predict_proba.return_value = np.array([[0.20, 0.20, 0.20]]) # sum doesn't matter for mock
        pipeline.predict_proba.return_value = np.array([[0.28, 0.27, 0.25]])

        predictor = FaceRecognitionPredictor(
            detector, extractor, pipeline, le,
            unknown_threshold=0.35, acceptance_threshold=0.65
        )
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.UNKNOWN
        assert result.predicted_identity is None
        assert "Unknown person" in result.error_message


class TestPredictorInvalidInput:

    def _predictor(self):
        return FaceRecognitionPredictor(
            make_mock_detector(), make_mock_extractor(),
            make_mock_pipeline(), make_label_encoder(),
        )

    def test_none_input(self):
        result = self._predictor().predict_from_array(None)
        assert result.status == RecognitionStatus.INVALID_INPUT

    def test_1d_array_input(self):
        result = self._predictor().predict_from_array(np.zeros(100))
        assert result.status == RecognitionStatus.INVALID_INPUT

    def test_grayscale_2d_input(self):
        result = self._predictor().predict_from_array(np.zeros((64, 64)))
        assert result.status == RecognitionStatus.INVALID_INPUT

    def test_missing_file(self, tmp_path):
        result = self._predictor().predict_from_path(tmp_path / "missing.jpg")
        assert result.status == RecognitionStatus.INVALID_INPUT


class TestPredictorModelError:

    def test_embedding_failure(self):
        le = make_label_encoder()
        detector = make_mock_detector(detected=True)
        extractor = MagicMock()
        extractor.embedding_dim = 1280
        extractor.extract.side_effect = RuntimeError("GPU OOM")
        extractor.validate_embedding.return_value = True
        pipeline = make_mock_pipeline()

        predictor = FaceRecognitionPredictor(detector, extractor, pipeline, le)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.MODEL_ERROR

    def test_nan_embedding_caught(self):
        le = make_label_encoder()
        detector = make_mock_detector(detected=True)
        extractor = MagicMock()
        extractor.embedding_dim = 1280
        extractor.extract.return_value = np.full(1280, np.nan, dtype=np.float32)
        extractor.validate_embedding.return_value = False
        pipeline = make_mock_pipeline()

        predictor = FaceRecognitionPredictor(detector, extractor, pipeline, le)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_from_array(img)
        assert result.status == RecognitionStatus.MODEL_ERROR
