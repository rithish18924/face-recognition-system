"""
tests/test_advanced_features.py -- Tests for multiple-face handling, prototype verification,
and open-set unknown person rejection.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from src.face_detector import FaceDetection, FaceDetectorBase
from src.feature_extractor import FeatureExtractorBase
from src.predictor import (
    FaceRecognitionPredictor,
    PredictionResult,
    RecognitionStatus,
)


def make_le(names=("Alice", "Bob", "Carol")) -> LabelEncoder:
    le = LabelEncoder()
    le.classes_ = np.array(names)
    return le


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestMultipleFaces:
    """Tests for multiple-face detection, selection, and response metadata."""

    def test_detector_returns_all_boxes(self):
        """Mock detector with 2 faces returns n_faces_found=2 and all_boxes."""
        det = MagicMock(spec=FaceDetectorBase)
        det.detect.return_value = FaceDetection(
            detected=True,
            face_crop=np.zeros((64, 64, 3), dtype=np.uint8),
            box=(10, 10, 80, 80),
            confidence=0.98,
            n_faces_found=2,
            all_boxes=[(10, 10, 80, 80), (120, 20, 50, 50)],
        )

        ext = MagicMock(spec=FeatureExtractorBase)
        ext.embedding_dim = 1280
        ext.extract.return_value = np.ones(1280, dtype=np.float32)
        ext.validate_embedding.return_value = True

        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.85, 0.10, 0.05]])

        le = make_le()
        predictor = FaceRecognitionPredictor(det, ext, pipe, le)
        result = predictor.predict_from_array(np.zeros((200, 200, 3), dtype=np.uint8))

        assert result.status == RecognitionStatus.SUCCESS
        assert result.face_detection.n_faces_found == 2
        assert len(result.face_detection.all_boxes) == 2

    def test_predict_endpoint_multiple_faces(self, client):
        """API endpoint should return n_faces_detected=2 and faces_note when multiple faces exist."""
        mock_pred = MagicMock()
        detection = FaceDetection(
            detected=True,
            face_crop=np.zeros((64, 64, 3), dtype=np.uint8),
            box=(10, 10, 80, 80),
            confidence=0.98,
            n_faces_found=2,
            all_boxes=[(10, 10, 80, 80), (120, 20, 50, 50)],
        )
        mock_result = PredictionResult(
            status=RecognitionStatus.SUCCESS,
            predicted_identity="Colin Powell",
            confidence=0.91,
            all_confidences={"Colin Powell": 0.91, "George W Bush": 0.09},
            face_box=(10, 10, 80, 80),
            face_detection=detection,
            inference_time_ms=15.0,
        )
        mock_pred.predict_from_array.return_value = mock_result

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("app.get_predictor", lambda: mock_pred)
            import cv2
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".png", img)
            data = {"image": (io.BytesIO(buf.tobytes()), "group.png")}
            res = client.post("/predict", data=data, content_type="multipart/form-data")
            assert res.status_code == 200
            d = res.get_json()
            assert d["n_faces_detected"] == 2
            assert "2 faces detected" in d["faces_note"]
            assert d["status"] == "SUCCESS"


class TestUnknownPersonRejectionAndPrototypeVerification:
    """
    Test open-set unknown person rejection.
    Verify that a high softmax confidence score alone does NOT guarantee identity
    if feature-space prototype similarity is low.
    """

    def test_high_prob_rejected_as_unknown_when_prototype_dissimilar(self):
        """Even with 92% softmax probability, low centroid similarity MUST reject as UNKNOWN."""
        det = MagicMock(spec=FaceDetectorBase)
        det.detect.return_value = FaceDetection(
            detected=True,
            face_crop=np.zeros((64, 64, 3), dtype=np.uint8),
            box=(0, 0, 64, 64),
            confidence=0.99,
            n_faces_found=1,
        )

        ext = MagicMock(spec=FeatureExtractorBase)
        ext.embedding_dim = 1280
        # Unit vector along axis 0
        emb = np.zeros(1280, dtype=np.float32)
        emb[0] = 1.0
        ext.extract.return_value = emb
        ext.validate_embedding.return_value = True

        # Classifier outputs high confidence (0.92) for Alice (index 0)
        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.92, 0.05, 0.03]])

        # Class centroids where Alice's centroid is along axis 1 (orthogonal -> cosine sim = 0.0)
        centroids = np.zeros((3, 1280), dtype=np.float32)
        centroids[0, 1] = 1.0  # Alice centroid
        centroids[1, 2] = 1.0  # Bob centroid
        centroids[2, 3] = 1.0  # Carol centroid

        le = make_le()
        predictor = FaceRecognitionPredictor(
            det, ext, pipe, le,
            unknown_threshold=0.40,
            acceptance_threshold=0.65,
            min_margin=0.15,
            class_centroids=centroids,
            min_cosine_similarity=0.82,
            unknown_cosine_similarity=0.80,
        )

        result = predictor.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
        # Must be rejected as UNKNOWN because prototype similarity is 0.0 < 0.80
        assert result.status == RecognitionStatus.UNKNOWN
        assert result.predicted_identity is None
        assert "Unknown person" in result.error_message

    def test_uncertain_when_prototype_similarity_borderline(self):
        """When probability is high and margin is wide, but prototype similarity is between 0.80 and 0.82."""
        det = MagicMock(spec=FaceDetectorBase)
        det.detect.return_value = FaceDetection(
            detected=True,
            face_crop=np.zeros((64, 64, 3), dtype=np.uint8),
            box=(0, 0, 64, 64),
            confidence=0.99,
            n_faces_found=1,
        )

        ext = MagicMock(spec=FeatureExtractorBase)
        ext.embedding_dim = 1280

        # Vector with cosine similarity exactly 0.81 to Alice's centroid
        # emb = [cos(theta), sin(theta), 0, ...]
        cos_sim = 0.81
        sin_sim = np.sqrt(1.0 - cos_sim ** 2)
        emb = np.zeros(1280, dtype=np.float32)
        emb[0] = cos_sim
        emb[1] = sin_sim
        ext.extract.return_value = emb
        ext.validate_embedding.return_value = True

        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.88, 0.08, 0.04]])

        centroids = np.zeros((3, 1280), dtype=np.float32)
        centroids[0, 0] = 1.0  # Alice centroid along axis 0

        le = make_le()
        predictor = FaceRecognitionPredictor(
            det, ext, pipe, le,
            unknown_threshold=0.40,
            acceptance_threshold=0.65,
            min_margin=0.15,
            class_centroids=centroids,
            min_cosine_similarity=0.82,
            unknown_cosine_similarity=0.80,
        )

        result = predictor.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
        # 0.81 is >= unknown (0.80) but < acceptance (0.82) -> UNCERTAIN
        assert result.status == RecognitionStatus.UNCERTAIN
        assert result.predicted_identity is None
        assert "Uncertain prediction" in result.error_message

    def test_success_when_all_criteria_met(self):
        """When probability is high, margin is wide, and prototype similarity is high -> SUCCESS."""
        det = MagicMock(spec=FaceDetectorBase)
        det.detect.return_value = FaceDetection(
            detected=True,
            face_crop=np.zeros((64, 64, 3), dtype=np.uint8),
            box=(0, 0, 64, 64),
            confidence=0.99,
            n_faces_found=1,
        )

        ext = MagicMock(spec=FeatureExtractorBase)
        ext.embedding_dim = 1280
        # High similarity (0.95)
        cos_sim = 0.95
        sin_sim = np.sqrt(1.0 - cos_sim ** 2)
        emb = np.zeros(1280, dtype=np.float32)
        emb[0] = cos_sim
        emb[1] = sin_sim
        ext.extract.return_value = emb
        ext.validate_embedding.return_value = True

        pipe = MagicMock()
        pipe.predict_proba.return_value = np.array([[0.90, 0.06, 0.04]])

        centroids = np.zeros((3, 1280), dtype=np.float32)
        centroids[0, 0] = 1.0  # Alice centroid along axis 0

        le = make_le()
        predictor = FaceRecognitionPredictor(
            det, ext, pipe, le,
            unknown_threshold=0.40,
            acceptance_threshold=0.65,
            min_margin=0.15,
            class_centroids=centroids,
            min_cosine_similarity=0.82,
            unknown_cosine_similarity=0.80,
        )

        result = predictor.predict_from_array(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result.status == RecognitionStatus.SUCCESS
        assert result.predicted_identity == "Alice"
        assert result.confidence == 0.90
