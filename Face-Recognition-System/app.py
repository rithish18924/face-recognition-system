"""
app.py -- Flask web interface for the Face Recognition System.

Routes
------
GET  /             -- Upload page
POST /predict      -- Recognition endpoint (JSON response)
GET  /evaluate     -- Shows evaluation summary
GET  /health       -- Health check
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.utils import get_logger

logger = get_logger("app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
app.secret_key = os.urandom(24).hex()

_predictor = None


def get_predictor():
    """Lazily load and cache the predictor instance."""
    global _predictor
    if _predictor is not None:
        return _predictor

    model_path = config.MODEL_ARTIFACT_PATH
    encoder_path = config.LABEL_ENCODER_PATH

    if not model_path.exists() or not encoder_path.exists():
        raise RuntimeError("Model artifacts not found. Run  python train.py  first.")

    import numpy as np
    from src.utils import load_pickle
    from src.feature_extractor import build_extractor
    from src.face_detector import build_detector
    from src.predictor import FaceRecognitionPredictor

    logger.info("Loading model artifacts ...")
    pipeline = load_pickle(model_path)
    le = load_pickle(encoder_path)
    extractor = build_extractor(config.FEATURE_EXTRACTOR)
    detector = build_detector(config.FACE_DETECTOR_STRATEGY)

    centroids_path = config.CENTROIDS_PATH
    centroids = np.load(centroids_path) if centroids_path.exists() else None

    _predictor = FaceRecognitionPredictor(
        detector=detector,
        extractor=extractor,
        classifier_pipeline=pipeline,
        label_encoder=le,
        unknown_threshold=config.UNKNOWN_THRESHOLD,
        acceptance_threshold=config.ACCEPTANCE_THRESHOLD,
        min_margin=config.MIN_MARGIN_THRESHOLD,
        class_centroids=centroids,
        min_cosine_similarity=config.MIN_COSINE_SIMILARITY,
        unknown_cosine_similarity=getattr(config, "UNKNOWN_COSINE_SIMILARITY", 0.8),
    )

    logger.info("Model ready. Classes: %d (centroids loaded: %s)", len(le.classes_), centroids is not None)
    return _predictor


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def _draw_result(image_rgb, result):
    """Draw bounding box, labels, and secondary faces on image."""
    import cv2
    import numpy as np
    from src.predictor import RecognitionStatus

    img = image_rgb.copy()

    # Draw secondary faces if present
    if result.face_detection and getattr(result.face_detection, "all_boxes", None):
        primary_box = result.face_box
        for box in result.face_detection.all_boxes:
            if primary_box and box == primary_box:
                continue
            bx, by, bw, bh = box
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (200, 180, 50), 1)
            cv2.putText(
                img,
                "Secondary face",
                (bx + 2, max(by - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 180, 50),
                1,
            )

    # Primary face box and label
    if result.face_box:
        x, y, w, h = result.face_box
        if result.status == RecognitionStatus.SUCCESS:
            color = (34, 197, 94)    # Green
            label = f"{result.predicted_identity} ({result.confidence:.1%})"
        elif result.status == RecognitionStatus.UNCERTAIN:
            color = (245, 158, 11)   # Yellow / Amber
            label = f"Uncertain ({result.confidence:.1%})"
        elif result.status == RecognitionStatus.UNKNOWN:
            color = (239, 68, 68)    # Red
            label = "Unknown Person"
        else:
            color = (239, 68, 68)    # Red
            label = result.status.name

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        y_text = max(y - 8, text_h + 4)
        cv2.rectangle(
            img,
            (x, y_text - text_h - 4),
            (x + text_w + 6, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x + 3, y_text - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Convert to BGR for cv2 encoding
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr_img)
    raw_bytes = buf.tobytes()
    b64_str = base64.b64encode(raw_bytes).decode("ascii")

    filename = f"annotated_{uuid.uuid4().hex}.png"
    filepath = config.UPLOAD_DIR / filename
    try:
        with open(filepath, "wb") as f_out:
            f_out.write(raw_bytes)
        url = f"/annotated/{filename}"
    except Exception as exc:
        logger.warning("Could not save annotated image: %s", exc)
        url = f"data:image/png;base64,{b64_str}"

    return b64_str, url


@app.route("/", methods=["GET"])
def index():
    identities = []
    try:
        predictor = get_predictor()
        if predictor and hasattr(predictor, "label_encoder") and predictor.label_encoder:
            identities = sorted([str(cls) for cls in predictor.label_encoder.classes_])
    except Exception as exc:
        logger.warning("Could not load identities for index page: %s", exc)
    return render_template("index.html", identities=identities, total_identities=len(identities))


@app.route("/api/identities", methods=["GET"])
def get_identities():
    try:
        predictor = get_predictor()
        identities = sorted([str(cls) for cls in predictor.label_encoder.classes_])
        return jsonify({
            "status": "ok",
            "total_identities": len(identities),
            "identities": identities,
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/predict", methods=["GET", "POST"])
@app.route("/predict/", methods=["POST"])
def predict():
    import cv2
    import numpy as np

    if request.method == "GET":
        return jsonify({"error": "Method Not Allowed. Use POST to upload an image."}), 405

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    f = request.files["image"]
    if f.filename == "" or not _allowed_file(f.filename):
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Invalid file. Allowed formats: {allowed}."}), 400

    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    if file_bytes.size == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Corrupted or invalid format."}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        predictor = get_predictor()
    except Exception as exc:
        logger.exception("Model loading failed.")
        return jsonify({"error": f"Model not ready: {exc}"}), 503

    try:
        result = predictor.predict_from_array(img_rgb)
        annotated_b64, annotated_url = _draw_result(img_rgb, result)

        n_faces = result.face_detection.n_faces_found if result.face_detection else 0
        faces_note = f"{n_faces} faces detected; selected largest" if n_faces > 1 else ""

        # Top confidences
        top_conf = dict(
            sorted(
                result.all_confidences.items(),
                key=lambda x: -x[1],
            )[:5]
        )

        return jsonify({
            "status": result.status.name,
            "predicted_identity": result.predicted_identity,
            "confidence": result.confidence,
            "all_confidences": top_conf,
            "face_box": list(result.face_box) if result.face_box else None,
            "n_faces_detected": n_faces,
            "faces_note": faces_note,
            "inference_time_ms": result.inference_time_ms,
            "error_message": result.error_message,
            "annotated_image": annotated_b64,
            "annotated_image_url": annotated_url,
            "image_url": annotated_url,
        })
    except Exception as exc:
        logger.exception("Unhandled error in /predict.")
        return jsonify({"error": f"Internal error: {type(exc).__name__}: {exc}"}), 500


@app.route("/annotated/<filename>", methods=["GET"])
@app.route("/static/uploads/<filename>", methods=["GET"])
def get_annotated_image(filename: str):
    safe_name = Path(filename).name
    file_path = config.UPLOAD_DIR / safe_name
    if not file_path.is_file():
        return jsonify({"error": "Annotated image not found."}), 404
    return send_from_directory(config.UPLOAD_DIR, safe_name)


@app.route("/artifacts/<filename>", methods=["GET"])
def get_artifact(filename: str):
    safe_name = Path(filename).name
    file_path = config.ARTIFACTS_DIR / safe_name
    if not file_path.is_file():
        return jsonify({"error": "Artifact not found."}), 404
    return send_from_directory(config.ARTIFACTS_DIR, safe_name)


@app.route("/evaluate", methods=["GET"])
def evaluate_page():
    report_path = config.EVAL_REPORT_PATH
    if not report_path.exists():
        return render_template("result.html", error="Evaluation report not found. Run python evaluate.py first.")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    return render_template("result.html", report=report)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _predictor is not None or config.MODEL_ARTIFACT_PATH.exists(),
    })


def _wants_json() -> bool:
    if request.path.startswith(("/predict", "/health", "/annotated", "/artifacts", "/api")):
        return True
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    if "application/json" in accept:
        return True
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    best = request.accept_mimetypes.best
    return best == "application/json"


@app.errorhandler(400)
def bad_request(exc):
    if _wants_json():
        return jsonify({"error": getattr(exc, "description", "Bad request.")}), 400
    return render_template("error.html", error="Bad request."), 400


@app.errorhandler(404)
def not_found(exc):
    if _wants_json():
        return jsonify({"error": "Resource not found."}), 404
    return render_template("error.html", error="Page not found."), 404


@app.errorhandler(405)
def method_not_allowed(exc):
    if _wants_json():
        return jsonify({"error": "Method not allowed. Use POST to upload an image."}), 405
    return render_template("error.html", error="Method not allowed."), 405


@app.errorhandler(413)
def too_large(_):
    return jsonify({
        "error": f"File too large. Maximum size: {config.MAX_UPLOAD_BYTES // 1048576} MB."
    }), 413


@app.errorhandler(500)
def server_error(exc):
    logger.exception("Unhandled server error: %s", exc)
    if _wants_json():
        return jsonify({"error": "Internal server error."}), 500
    return render_template("error.html", error="Internal server error."), 500


if __name__ == "__main__":
    logger.info("Starting Flask on %s:%d", config.FLASK_HOST, config.FLASK_PORT)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
