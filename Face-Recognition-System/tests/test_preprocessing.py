"""
tests/test_preprocessing.py -- Tests for face detection and feature extraction.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_rgb(h=120, w=120, val=128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def make_face_image() -> np.ndarray:
    """
    Return a plausible face-sized image (random, not a real face).
    MTCNN may not detect it -- tests that need a real detection use HaarCascade.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


# ─── Face Detector ───────────────────────────────────────────────────────────

class TestFaceDetectorBuilding:

    def test_build_haar_detector(self):
        """HaarCascadeDetector should build without raising, even if OpenCV 5
        has moved or removed CascadeClassifier."""
        from src.face_detector import build_detector
        # Should not raise -- graceful degradation on OpenCV 5
        det = build_detector("haar")
        assert det is not None

    def test_invalid_strategy_raises(self):
        from src.face_detector import build_detector
        with pytest.raises(ValueError):
            build_detector("nonexistent_strategy")

    def test_detect_returns_no_face_on_blank_image(self):
        """Blank image should return detected=False (either no face or cascade unavailable)."""
        from src.face_detector import build_detector
        det = build_detector("haar")
        img = make_rgb(64, 64, 200)
        result = det.detect(img)
        # Either no face detected or cascade unavailable -- both are valid False
        assert result.detected is False
        assert hasattr(result, "face_crop")
        assert hasattr(result, "confidence")

    def test_detect_returns_false_on_none(self):
        from src.face_detector import build_detector
        det = build_detector("haar")
        result = det.detect(None)
        assert result.detected is False
        assert len(result.error_message) > 0

    def test_detect_returns_false_on_wrong_shape(self):
        from src.face_detector import build_detector
        det = build_detector("haar")
        img2d = np.zeros((64, 64), dtype=np.uint8)
        result = det.detect(img2d)
        assert result.detected is False

    def test_clip_box_stays_in_image(self):
        """_clip_box is a pure function on the base class -- test it directly."""
        from src.face_detector import MTCNNDetector, FaceDetectorBase
        # Use FaceDetectorBase directly (no __init__ needed for _clip_box)
        base = FaceDetectorBase.__new__(FaceDetectorBase)
        # Box extends outside image boundary
        x, y, w, h = base._clip_box(-5, -10, 200, 300, img_h=100, img_w=100)
        assert x >= 0 and y >= 0
        assert x + w <= 100
        assert y + h <= 100


class TestFeatureExtractor:

    def test_mobilenetv2_builds(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        assert ext is not None
        assert ext.embedding_dim == 1280
        assert ext.input_shape == (224, 224)

    def test_mobilenetv2_preprocess_output_shape(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        img = make_rgb(100, 80)
        out = ext.preprocess(img)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32

    def test_mobilenetv2_preprocess_range(self):
        """MobileNetV2 preprocess_input should produce values roughly in [-1, 1]."""
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        out = ext.preprocess(img)
        assert out.min() >= -1.1
        assert out.max() <=  1.1

    def test_extract_single_image_shape(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        img = make_rgb(80, 80)
        emb = ext.extract(img)
        assert emb.shape == (1280,)
        assert emb.dtype == np.float32

    def test_extract_single_image_finite(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        img = make_rgb(80, 80)
        emb = ext.extract(img)
        assert np.isfinite(emb).all(), "Embedding contains non-finite values"

    def test_extract_batch_shape(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        imgs = np.stack([make_rgb(64, 64) for _ in range(4)])
        embs = ext.extract_batch(imgs)
        assert embs.shape == (4, 1280)

    def test_validate_embedding_good(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        good = np.ones(1280, dtype=np.float32)
        assert ext.validate_embedding(good) is True

    def test_validate_embedding_nan(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        bad = np.full(1280, np.nan, dtype=np.float32)
        assert ext.validate_embedding(bad) is False

    def test_validate_embedding_wrong_dim(self):
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        wrong = np.ones(512, dtype=np.float32)
        assert ext.validate_embedding(wrong) is False

    def test_batch_embedding_consistent(self):
        """Same image extracted individually vs in batch should give identical embeddings."""
        from src.feature_extractor import build_extractor
        ext = build_extractor("mobilenetv2")
        img = make_rgb(100, 100, val=77)

        emb_single = ext.extract(img)
        emb_batch  = ext.extract_batch(np.expand_dims(img, 0))[0]
        np.testing.assert_allclose(emb_single, emb_batch, rtol=1e-4)
