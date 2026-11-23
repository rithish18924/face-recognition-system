"""
src/utils.py -- Shared utilities: logging setup, image I/O, serialization helpers.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for this project (idempotent)."""
    from config import LOG_LEVEL, LOG_FILE  # lazy import so utils can be imported early

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — force UTF-8 so Unicode chars never crash on Windows cp1252
    import io
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass  # Silently skip file logging if path is not writable

    return logger


# ─── Serialization ───────────────────────────────────────────────────────────

def save_pickle(obj: Any, path: Path | str) -> None:
    """Serialize *obj* to *path* using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path | str) -> Any:
    """Deserialize an object from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Image helpers ───────────────────────────────────────────────────────────

def load_image_rgb(path: Path | str) -> np.ndarray | None:
    """
    Load an image from *path* and return it as an RGB uint8 ndarray.
    Returns None on read failure.
    """
    import cv2  # type: ignore

    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def is_valid_image(path: Path | str) -> bool:
    """Return True if *path* can be read as an image."""
    return load_image_rgb(path) is not None


def resize_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize *img* (H×W×C) to *target_size* = (W, H) using bilinear interpolation."""
    import cv2  # type: ignore

    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def clip_box(box: tuple[int, int, int, int], img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """
    Clip a bounding box (x, y, w, h) to image boundaries.
    Returns (x, y, w, h) safe for numpy slicing.
    """
    x, y, w, h = box
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)
