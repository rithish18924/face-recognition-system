"""
src/face_detector.py -- Face detection with consistent bounding-box policy.

Primary  : MTCNN (deep-learning based, high accuracy)
Fallback : OpenCV Haar cascade (always available, lower accuracy)

The same detector instance MUST be used during training AND inference so that
the face-selection policy is identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """Result of detecting faces in a single RGB image."""
    detected: bool
    face_crop: Optional[np.ndarray]  # (H, W, 3) uint8 RGB crop, or None
    box: Optional[tuple[int, int, int, int]]  # (x, y, w, h) in original image
    confidence: float
    n_faces_found: int
    error_message: str = ""
    all_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# ─── Detector base class ─────────────────────────────────────────────────────

class FaceDetectorBase:
    """Common interface for all face detectors."""

    def detect(self, image_rgb: np.ndarray) -> FaceDetection:
        raise NotImplementedError

    def _clip_box(
        self, x: int, y: int, w: int, h: int, img_h: int, img_w: int
    ) -> tuple[int, int, int, int]:
        """Clip box to image boundaries."""
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_w, int(x + w))
        y2 = min(img_h, int(y + h))
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    def _crop(self, img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = box
        return img[y : y + h, x : x + w].copy()

    def _select_face(
        self,
        detections: list[dict],
        policy: str = "largest",
    ) -> dict:
        """Select one face from a list using *policy*."""
        if not detections:
            raise ValueError("No detections to select from.")
        if len(detections) == 1 or policy == "first":
            return detections[0]
        if policy == "largest":
            def area(d: dict) -> int:
                b = d["box"]
                return b[2] * b[3]
            return max(detections, key=area)
        return detections[0]


# ─── MTCNN Detector ──────────────────────────────────────────────────────────

class MTCNNDetector(FaceDetectorBase):
    """
    MTCNN-based face detector.

    Input must be RGB uint8.
    MTCNN returns:  list of {'box': [x,y,w,h], 'confidence': float, 'keypoints': ...}
    """

    def __init__(
        self,
        min_face_size: int = 20,
        confidence_threshold: float = 0.90,
        select_policy: str = "largest",
    ) -> None:
        from mtcnn import MTCNN  # type: ignore
        import inspect
        sig = inspect.signature(MTCNN.__init__)
        if "min_face_size" in sig.parameters:
            self._detector = MTCNN(min_face_size=min_face_size)
        else:
            self._detector = MTCNN()
        self._conf_thresh = confidence_threshold
        self._select_policy = select_policy
        logger.info("MTCNNDetector initialised (conf=%.2f).", confidence_threshold)

    def detect(self, image_rgb: np.ndarray) -> FaceDetection:
        """Detect faces in *image_rgb* (H×W×3 uint8 RGB)."""
        if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3:
            return FaceDetection(False, None, None, 0.0, 0, "Invalid image input.")

        # Ensure uint8
        if np.issubdtype(image_rgb.dtype, np.floating) and image_rgb.max() <= 1.0:
            image_rgb = (np.clip(image_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        elif image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        img_h, img_w = image_rgb.shape[:2]

        try:
            raw = self._detector.detect_faces(image_rgb)
        except Exception as exc:
            return FaceDetection(False, None, None, 0.0, 0, f"MTCNN error: {exc}")

        if not raw:
            return FaceDetection(False, None, None, 0.0, 0, "No face detected.")

        # Normalise output: MTCNN 1.0.0 stores confidence in
        # result['confidence'] (float) or result['score'] depending on build.
        def _conf(d):
            c = d.get("confidence", d.get("score", 0.0))
            return float(c) if not isinstance(c, (list, dict)) else 0.0

        # Filter by confidence
        confident = [d for d in raw if _conf(d) >= self._conf_thresh]
        if not confident:
            # Fall back to best available
            confident = sorted(raw, key=lambda d: -_conf(d))[:1]
            logger.debug("All faces below confidence threshold -- using best (%.3f).",
                         _conf(confident[0]))

        # Clip all detected bounding boxes
        all_boxes = [self._clip_box(*d["box"], img_h, img_w) for d in confident]
        all_boxes = [b for b in all_boxes if b[2] >= 10 and b[3] >= 10]

        n_found = len(all_boxes) if all_boxes else len(confident)
        chosen = self._select_face(confident, self._select_policy)
        chosen_conf = _conf(chosen)

        x, y, w, h = chosen["box"]
        x, y, w, h = self._clip_box(x, y, w, h, img_h, img_w)

        if w < 10 or h < 10:
            return FaceDetection(False, None, None, chosen_conf, n_found,
                                 "Face bounding box too small after clipping.",
                                 all_boxes=all_boxes)

        crop = self._crop(image_rgb, (x, y, w, h))
        return FaceDetection(
            detected=True,
            face_crop=crop,
            box=(x, y, w, h),
            confidence=chosen_conf,
            n_faces_found=n_found,
            all_boxes=all_boxes,
        )


# ─── OpenCV Haar Cascade Fallback ────────────────────────────────────────────

class HaarCascadeDetector(FaceDetectorBase):
    """
    OpenCV Haar Cascade face detector -- always available, lower accuracy.
    Used as a fallback when MTCNN is unavailable.
    """

    def __init__(self, select_policy: str = "largest") -> None:
        import cv2  # type: ignore
        import os

        # OpenCV 5.x moved CascadeClassifier to cv2.legacy.
        # Try both locations gracefully.
        CascadeClassifier = (
            getattr(cv2, "CascadeClassifier", None)
            or getattr(getattr(cv2, "legacy", None), "CascadeClassifier", None)
        )
        if CascadeClassifier is None:
            logger.warning("CascadeClassifier not found in cv2 or cv2.legacy; "
                           "Haar detector disabled -- no detections will be returned.")
            self._cascade = None
        else:
            haarcascades_dir = getattr(
                getattr(cv2, "data", None), "haarcascades", None
            )
            if haarcascades_dir is None:
                logger.warning("cv2.data.haarcascades path not found; Haar detector disabled.")
                self._cascade = None
            else:
                cascade_path = os.path.join(
                    haarcascades_dir, "haarcascade_frontalface_default.xml"
                )
                self._cascade = CascadeClassifier(cascade_path)
        self._select_policy = select_policy
        logger.warning("Using HaarCascadeDetector as fallback -- accuracy will be lower than MTCNN.")

    def detect(self, image_rgb: np.ndarray) -> FaceDetection:
        import cv2  # type: ignore

        if image_rgb is None or image_rgb.ndim != 3:
            return FaceDetection(False, None, None, 0.0, 0, "Invalid image input.")

        if self._cascade is None:
            return FaceDetection(False, None, None, 0.0, 0,
                                 "Haar cascade unavailable (OpenCV 5 compatibility issue).")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        img_h, img_w = image_rgb.shape[:2]

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return FaceDetection(False, None, None, 0.0, 0, "No face detected (Haar).")

        detections = [{"box": list(map(int, f)), "confidence": 1.0} for f in faces]
        all_boxes = [self._clip_box(*d["box"], img_h, img_w) for d in detections]
        all_boxes = [b for b in all_boxes if b[2] >= 10 and b[3] >= 10]
        n_found = len(all_boxes) if all_boxes else len(detections)

        chosen = self._select_face(detections, self._select_policy)
        x, y, w, h = self._clip_box(*chosen["box"], img_h, img_w)

        if w < 10 or h < 10:
            return FaceDetection(False, None, None, 1.0, n_found,
                                 "Face bounding box too small after clipping.",
                                 all_boxes=all_boxes)

        crop = self._crop(image_rgb, (x, y, w, h))
        return FaceDetection(
            detected=True,
            face_crop=crop,
            box=(x, y, w, h),
            confidence=1.0,
            n_faces_found=n_found,
            all_boxes=all_boxes,
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_detector(
    strategy: str = "mtcnn",
    min_face_size: int = 20,
    confidence_threshold: float = 0.90,
    select_policy: str = "largest",
) -> FaceDetectorBase:
    """
    Instantiate a face detector by *strategy*.

    Falls back to HaarCascade if MTCNN cannot be imported.
    """
    if strategy in ("mtcnn", "auto"):
        try:
            return MTCNNDetector(
                min_face_size=min_face_size,
                confidence_threshold=confidence_threshold,
                select_policy=select_policy,
            )
        except (ImportError, Exception) as exc:
            logger.warning("MTCNN not available (%s) -- falling back to Haar cascade.", exc)
            return HaarCascadeDetector(select_policy=select_policy)

    if strategy == "haar":
        return HaarCascadeDetector(select_policy=select_policy)

    raise ValueError(f"Unknown face detector strategy: {strategy!r}. "
                     "Choose 'mtcnn' or 'haar'.")
