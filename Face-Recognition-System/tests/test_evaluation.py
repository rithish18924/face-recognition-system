"""
tests/test_evaluation.py -- Tests for evaluator and model serialization.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import LabelEncoder
from src.predictor import RecognitionStatus, PredictionResult


# ─── Helper factories ─────────────────────────────────────────────────────────

def make_le(names=("Alice", "Bob", "Carol")) -> LabelEncoder:
    le = LabelEncoder()
    le.classes_ = np.array(names)
    return le


def make_success_result(identity: str, confidence: float = 0.85) -> PredictionResult:
    return PredictionResult(
        status=RecognitionStatus.SUCCESS,
        predicted_identity=identity,
        confidence=confidence,
        all_confidences={identity: confidence},
        face_box=(0, 0, 64, 64),
        face_detection=None,
        inference_time_ms=12.0,
    )


def make_no_face_result() -> PredictionResult:
    return PredictionResult(
        status=RecognitionStatus.NO_FACE,
        predicted_identity=None,
        confidence=0.0,
        all_confidences={},
        face_box=None,
        face_detection=None,
        inference_time_ms=5.0,
        error_message="No face detected.",
    )


# ─── Metrics tests ───────────────────────────────────────────────────────────

class TestMetricsCalculation:

    def _run_eval(self, y_true, y_pred_names, le, report_path=None):
        """
        Build a mock predictor that returns pre-set results,
        then call evaluate_on_dataset.
        """
        from src.evaluator import evaluate_on_dataset

        results = []
        for name in y_pred_names:
            if name is None:
                results.append(make_no_face_result())
            else:
                results.append(make_success_result(name, 0.9))

        it = iter(results)

        class MockPredictor:
            def predict_from_face_crop(self, _img):
                return next(it)

        n = len(y_true)
        X = np.zeros((n, 10), dtype=np.float32)   # fake embeddings

        return evaluate_on_dataset(
            predictor=MockPredictor(),
            X_test=X,
            y_test=np.array(y_true),
            label_encoder=le,
            use_face_detection=False,
            report_path=report_path,
        )

    def test_perfect_accuracy(self):
        le = make_le(["Alice", "Bob"])
        y_true = [0, 0, 1, 1]
        y_pred = ["Alice", "Alice", "Bob", "Bob"]
        metrics = self._run_eval(y_true, y_pred, le)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        le = make_le(["Alice", "Bob"])
        y_true = [0, 0, 1, 1]
        y_pred = ["Bob", "Bob", "Alice", "Alice"]
        metrics = self._run_eval(y_true, y_pred, le)
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_partial_accuracy(self):
        le = make_le(["Alice", "Bob"])
        y_true = [0, 0, 1, 1]
        y_pred = ["Alice", "Bob", "Bob", "Alice"]
        metrics = self._run_eval(y_true, y_pred, le)
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_no_face_excluded_from_accuracy(self):
        le = make_le(["Alice", "Bob"])
        # 4 samples, 1 has no face -- evaluated on 3
        y_true = [0, 0, 1, 1]
        y_pred = ["Alice", None, "Bob", "Bob"]
        metrics = self._run_eval(y_true, y_pred, le)
        assert metrics["n_no_face"] == 1
        assert metrics["n_evaluated"] == 3

    def test_metrics_keys_present(self):
        le = make_le(["Alice", "Bob"])
        y_true = [0, 1]
        y_pred = ["Alice", "Bob"]
        metrics = self._run_eval(y_true, y_pred, le)
        for key in ("accuracy", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro",
                    "n_evaluated", "n_correct", "n_incorrect"):
            assert key in metrics

    def test_report_saved_to_disk(self, tmp_path):
        import json
        le = make_le(["Alice", "Bob"])
        y_true = [0, 1]
        y_pred = ["Alice", "Bob"]
        report_path = tmp_path / "report.json"
        self._run_eval(y_true, y_pred, le, report_path=report_path)
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "accuracy" in data


class TestSerializationRoundTrip:

    def test_pickle_roundtrip(self, tmp_path):
        from src.utils import save_pickle, load_pickle

        obj = {"key": [1, 2, 3], "nested": {"a": np.array([1.0, 2.0])}}
        p = tmp_path / "test.pkl"
        save_pickle(obj, p)
        loaded = load_pickle(p)
        assert loaded["key"] == [1, 2, 3]
        np.testing.assert_array_equal(loaded["nested"]["a"], obj["nested"]["a"])

    def test_load_missing_raises(self, tmp_path):
        from src.utils import load_pickle
        with pytest.raises(FileNotFoundError):
            load_pickle(tmp_path / "missing.pkl")

    def test_label_encoder_roundtrip(self, tmp_path):
        from src.utils import save_pickle, load_pickle

        le = LabelEncoder()
        le.classes_ = np.array(["Alice", "Bob", "Carol"])
        p = tmp_path / "le.pkl"
        save_pickle(le, p)
        le2 = load_pickle(p)
        np.testing.assert_array_equal(le.classes_, le2.classes_)
