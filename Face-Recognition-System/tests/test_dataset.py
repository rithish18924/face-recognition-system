"""
tests/test_dataset.py -- Unit tests for src/dataset.py
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Helper: tiny in-memory image array ──────────────────────────────────────

def make_rgb_images(n: int = 10, h: int = 62, w: int = 47) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, h, w, 3), dtype=np.uint8)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestDatasetSplit:
    """Test that train/val/test splits are correctly sized and non-overlapping."""

    def _make_bundle(self, n=100, n_classes=5):
        """Create a minimal DatasetBundle with synthetic data."""
        from src.dataset import DatasetBundle
        from sklearn.preprocessing import LabelEncoder

        X = make_rgb_images(n)
        y = np.tile(np.arange(n_classes), n // n_classes + 1)[:n]

        le = LabelEncoder()
        y_enc = le.fit_transform([str(i) for i in y])

        from sklearn.model_selection import train_test_split
        X_tv, X_test, y_tv, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.125, stratify=y_tv, random_state=42)

        return DatasetBundle(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            label_encoder=le,
            target_names=list(le.classes_),
            image_shape=(62, 47),
            summary={"source": "synthetic", "total_images": n, "n_classes": n_classes,
                     "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                     "random_seed": 42},
        )

    def test_split_sizes_sum_to_total(self):
        bundle = self._make_bundle(n=100)
        total = len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test)
        assert total == 100

    def test_splits_have_matching_labels(self):
        bundle = self._make_bundle(n=100)
        assert len(bundle.X_train) == len(bundle.y_train)
        assert len(bundle.X_val)   == len(bundle.y_val)
        assert len(bundle.X_test)  == len(bundle.y_test)

    def test_label_encoder_classes_count(self):
        bundle = self._make_bundle(n=100, n_classes=5)
        assert len(bundle.label_encoder.classes_) == 5

    def test_images_are_uint8_rgb(self):
        bundle = self._make_bundle(n=50)
        assert bundle.X_train.dtype == np.uint8
        assert bundle.X_train.ndim == 4
        assert bundle.X_train.shape[-1] == 3

    def test_labels_in_valid_range(self):
        bundle = self._make_bundle(n=100, n_classes=5)
        n_classes = len(bundle.label_encoder.classes_)
        for arr in [bundle.y_train, bundle.y_val, bundle.y_test]:
            assert arr.min() >= 0
            assert arr.max() < n_classes


class TestCustomDirectoryLoader:
    """Test custom dataset directory loading with a temp filesystem."""

    def test_missing_root_raises(self, tmp_path):
        from src.dataset import load_custom_directory
        with pytest.raises(FileNotFoundError):
            load_custom_directory(tmp_path / "nonexistent")

    def test_empty_root_raises(self, tmp_path):
        from src.dataset import load_custom_directory
        with pytest.raises(ValueError):
            load_custom_directory(tmp_path, min_faces_per_person=1)

    def test_loads_valid_directory(self, tmp_path):
        """Create 3 persons × 10 images each using synthetic PNG data."""
        import cv2

        for person in ["Alice", "Bob", "Carol"]:
            d = tmp_path / person
            d.mkdir()
            for i in range(10):
                img = np.full((48, 48, 3), (i * 25) % 255, dtype=np.uint8)
                cv2.imwrite(str(d / f"img_{i:02d}.jpg"), img)

        from src.dataset import load_custom_directory
        bundle = load_custom_directory(tmp_path, min_faces_per_person=5, random_seed=42)

        assert bundle.summary["n_classes"] == 3
        assert bundle.summary["total_images"] == 30
        assert len(bundle.target_names) == 3

    def test_filters_small_classes(self, tmp_path):
        import cv2

        # Alice: 10 images (kept), Bob: 2 images (filtered)
        for person, count in [("Alice", 10), ("Bob", 2)]:
            d = tmp_path / person
            d.mkdir()
            for i in range(count):
                img = np.zeros((48, 48, 3), dtype=np.uint8)
                cv2.imwrite(str(d / f"img_{i}.jpg"), img)

        from src.dataset import load_custom_directory
        bundle = load_custom_directory(tmp_path, min_faces_per_person=5, random_seed=42)
        assert bundle.summary["n_classes"] == 1
        assert "Alice" in bundle.target_names
