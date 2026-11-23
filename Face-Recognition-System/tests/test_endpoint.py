"""
tests/test_endpoint.py -- Comprehensive tests for the Flask web endpoints.

Tests:
  - GET / (Upload page)
  - GET /health (Health check)
  - POST /predict with no file / empty filename / invalid extension / corrupt file
  - POST /predict with no-face image
  - POST /predict with valid face image (successful recognition)
  - GET /predict (Method Not Allowed - returns JSON)
  - Annotated image retrieval (/annotated/<filename> and /static/uploads/<filename>)
  - Server error handling on API route (returns JSON, not HTML)
  - 404 on API route (returns JSON, not HTML)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from app import app
import config


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_face_bytes() -> bytes:
    """Return bytes of a real face image."""
    test_face_path = Path(__file__).parent / "test_face.jpg"
    if test_face_path.exists():
        with open(test_face_path, "rb") as f:
            return f.read()

    # Fallback to LFW funneled dataset if present
    lfw_colin = Path(r"C:\Users\sures\scikit_learn_data\lfw_home\lfw_funneled\Colin_Powell\Colin_Powell_0001.jpg")
    if lfw_colin.exists():
        with open(lfw_colin, "rb") as f:
            return f.read()

    # Synthetic fallback
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(img, (100, 100), (70, 90), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(img, (70, 80), 14, (50, 50, 50), -1)
    cv2.circle(img, (130, 80), 14, (50, 50, 50), -1)
    cv2.ellipse(img, (100, 130), (30, 16), 0, 0, 180, (150, 80, 80), 4)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def no_face_image_bytes() -> bytes:
    """Return bytes of a plain black image with no face."""
    blank = np.zeros((150, 150, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", blank)
    return buf.tobytes()


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_index_page(client):
    """GET / should render the upload page with status 200."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "drop-zone" in html
    assert "file-input" in html
    assert "recognize-btn" in html
    assert "result-card" in html


def test_health(client):
    """GET /health should return JSON status ok."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.is_json
    data = res.get_json()
    assert data.get("status") == "ok"
    assert "model_loaded" in data


def test_predict_no_image_uploaded(client):
    """POST /predict without multipart file should return 400 JSON."""
    res = client.post("/predict", data={})
    assert res.status_code == 400
    assert res.is_json
    data = res.get_json()
    assert "error" in data
    assert "No image file uploaded" in data["error"]


def test_predict_empty_filename(client):
    """POST /predict with empty filename should return 400 JSON."""
    data = {"image": (io.BytesIO(b"dummy"), "")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.is_json
    assert "error" in res.get_json()


def test_predict_invalid_extension(client):
    """POST /predict with unsupported file type should return 400 JSON."""
    data = {"image": (io.BytesIO(b"dummy text"), "notes.txt")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.is_json
    data = res.get_json()
    assert "error" in data
    assert "Allowed formats" in data["error"]


def test_predict_corrupt_image(client):
    """POST /predict with corrupted image bytes should return 400 JSON."""
    data = {"image": (io.BytesIO(b"not a valid image format content"), "corrupt.png")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert res.is_json
    data = res.get_json()
    assert "error" in data
    assert "Could not decode image" in data["error"]


def test_predict_no_face_image(client, no_face_image_bytes):
    """POST /predict with an image containing no face should return 200 with NO_FACE status."""
    data = {"image": (io.BytesIO(no_face_image_bytes), "noface.png")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.is_json
    d = res.get_json()
    assert d.get("status") == "NO_FACE"
    assert d.get("predicted_identity") is None
    assert d.get("confidence") == 0.0
    assert d.get("face_box") is None
    assert "No face detected" in (d.get("error_message") or "")
    assert d.get("annotated_image") is not None
    assert d.get("annotated_image_url") is not None


def test_predict_successful_recognition(client, sample_face_bytes):
    """POST /predict with a valid face image should return 200 with recognition details."""
    data = {"image": (io.BytesIO(sample_face_bytes), "face.jpg")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.is_json
    d = res.get_json()
    assert d.get("status") in ("SUCCESS", "UNCERTAIN", "UNKNOWN")
    assert d.get("confidence") > 0.0
    assert d.get("face_box") is not None
    assert len(d.get("face_box")) == 4
    assert d.get("annotated_image") is not None
    assert len(d.get("annotated_image")) > 50
    assert d.get("annotated_image_url") is not None


def test_predict_trailing_slash_returns_json(client, sample_face_bytes):
    """POST /predict/ with trailing slash should work directly without HTML error redirect."""
    data = {"image": (io.BytesIO(sample_face_bytes), "face.jpg")}
    res = client.post("/predict/", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.is_json
    d = res.get_json()
    assert d.get("status") in ("SUCCESS", "UNCERTAIN", "UNKNOWN")
    assert d.get("image_url") is not None
    assert "inference_time_ms" in d


def test_annotated_image_retrieval(client, sample_face_bytes):
    """Annotated image URL returned by /predict should be retrievable with HTTP 200."""
    data = {"image": (io.BytesIO(sample_face_bytes), "face.jpg")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    d = res.get_json()

    annotated_url = d.get("annotated_image_url")
    assert annotated_url is not None

    # Retrieve via the URL returned
    img_res = client.get(annotated_url)
    assert img_res.status_code == 200
    assert len(img_res.data) > 0
    # PNG signature check: \x89PNG\r\n\x1a\n
    assert img_res.data.startswith(b"\x89PNG")

    # Also test retrieval via /annotated/<filename>
    filename = Path(annotated_url).name
    direct_res = client.get(f"/annotated/{filename}")
    assert direct_res.status_code == 200
    assert direct_res.data.startswith(b"\x89PNG")


def test_annotated_image_not_found(client):
    """Retrieving a non-existent annotated image should return 404 JSON."""
    res = client.get("/annotated/non_existent_file_xyz.png")
    assert res.status_code == 404
    assert res.is_json
    assert "error" in res.get_json()


def test_predict_get_method_not_allowed(client):
    """GET /predict should return 405 JSON instead of HTML."""
    res = client.get("/predict")
    assert res.status_code == 405
    assert res.is_json
    data = res.get_json()
    assert "error" in data
    assert "Method Not Allowed" in data["error"]


def test_server_error_returns_json(client):
    """When an unhandled exception occurs in /predict, it must return JSON, not HTML."""
    with patch("app.get_predictor", side_effect=RuntimeError("Simulated failure")):
        data = {"image": (io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "test.png")}
        # Mock cv2.imdecode so it passes initial decode
        with patch("cv2.imdecode", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            res = client.post("/predict", data=data, content_type="multipart/form-data")
            assert res.status_code in (500, 503)
            assert res.is_json
            assert "error" in res.get_json()


def test_evaluate_page(client):
    """GET /evaluate should return HTTP 200."""
    res = client.get("/evaluate")
    assert res.status_code == 200


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
