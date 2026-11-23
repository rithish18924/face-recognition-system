"""
src/feature_extractor.py -- Face embedding extraction.

Primary  : MobileNetV2 (tf.keras.applications) -- no extra download needed,
           always compatible with TensorFlow 2.x.
           Input: RGB uint8 or float32 [0–255] -> internally scaled to [-1,1]
           Output: 1280-dim embedding (GlobalAveragePooling2D)

Secondary: EfficientNet Lite0 via TensorFlow Hub (requires network on first run)
           Input: float32 in [0,1] after resize to 224×224
           Output: 1280-dim embedding

The preprocessor used at training time MUST match inference time.
All preprocessing is encapsulated here and applied identically everywhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Ensure KERAS_HOME and MPLCONFIGDIR point inside the project directory
_ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_ROOT / ".keras")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(_ROOT / ".matplotlib")

import numpy as np

logger = logging.getLogger(__name__)

# Input sizes
MOBILENETV2_INPUT = (224, 224)
EFFICIENTNET_INPUT = (224, 224)


# ─── Base ────────────────────────────────────────────────────────────────────

class FeatureExtractorBase:
    """Common interface for all feature extractors."""

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @property
    def input_shape(self) -> tuple[int, int]:
        raise NotImplementedError

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Preprocess a single RGB image for this extractor."""
        raise NotImplementedError

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract a 1-D embedding from a single RGB image."""
        batch = self.extract_batch(np.expand_dims(image_rgb, axis=0))
        return batch[0]

    def extract_batch(self, images_rgb: np.ndarray) -> np.ndarray:
        """Extract embeddings from a batch (N, H, W, 3) of RGB images."""
        raise NotImplementedError

    def validate_embedding(self, emb: np.ndarray) -> bool:
        """Return True if the embedding is finite and has the expected dimension."""
        return bool(
            emb.ndim == 1
            and emb.shape[0] == self.embedding_dim
            and np.isfinite(emb).all()
        )


# ─── MobileNetV2 ─────────────────────────────────────────────────────────────

class MobileNetV2Extractor(FeatureExtractorBase):
    """
    Feature extraction with MobileNetV2 + GlobalAveragePooling2D.

    Preprocessing contract (MobileNetV2):
      - Input: float32 values in [0, 255]  (or uint8, auto-converted)
      - tf.keras.applications.mobilenet_v2.preprocess_input rescales to [-1, 1]
      - Input spatial size: 224 × 224
      - Output: 1280-dimensional L2-normalised embedding
    """

    def __init__(self) -> None:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras import Model

        logger.info("Building MobileNetV2 feature extractor ...")
        base = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        base.trainable = False
        self._model = base
        self._preprocess = preprocess_input
        self._tf = tf
        logger.info("MobileNetV2 loaded. Output dim: %d.", self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return 1280

    @property
    def input_shape(self) -> tuple[int, int]:
        return MOBILENETV2_INPUT

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Resize to 224×224, convert to float32 in [0,255], apply MobileNetV2
        preprocess_input (-> [-1,1]).
        Returns (224, 224, 3) float32.
        """
        import cv2  # type: ignore

        img = image_rgb
        if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
            img = img * 255.0
        img = cv2.resize(img.astype(np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
        img = self._preprocess(img)           # -> [-1, 1]
        return img

    def extract_batch(self, images_rgb: np.ndarray) -> np.ndarray:
        """
        images_rgb : (N, H, W, 3) uint8 or float32 RGB
        Returns     : (N, 1280) float32 embeddings
        """
        import cv2  # type: ignore

        processed = np.stack([
            self.preprocess(images_rgb[i]) for i in range(len(images_rgb))
        ], axis=0)

        embeddings = self._model.predict(processed, verbose=0)
        return embeddings.astype(np.float32)


# ─── EfficientNet Lite0 via TensorFlow Hub ───────────────────────────────────

class EfficientNetHubExtractor(FeatureExtractorBase):
    """
    Feature extraction using EfficientNet Lite0 from TensorFlow Hub.

    Preprocessing contract (EfficientNet Lite0 feature-vector):
      - Input: float32 in [0, 1]  (NOT [-1,1], NOT uint8)
      - Input spatial size: 224 × 224
      - Output: 1280-dimensional embedding
    """

    def __init__(
        self,
        hub_url: str = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2",
    ) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub  # type: ignore

        logger.info("Loading EfficientNet Lite0 from TF Hub ...")
        self._model = hub.KerasLayer(hub_url, trainable=False)
        # Warm-up to trigger weight download
        dummy = tf.zeros([1, 224, 224, 3])
        self._model(dummy)
        self._tf = tf
        logger.info("EfficientNet Lite0 loaded. Output dim: %d.", self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return 1280

    @property
    def input_shape(self) -> tuple[int, int]:
        return EFFICIENTNET_INPUT

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Resize to 224×224, convert to float32 in [0, 1].
        Returns (224, 224, 3) float32.
        """
        import cv2  # type: ignore

        img = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0    # -> [0, 1]
        return img

    def extract_batch(self, images_rgb: np.ndarray) -> np.ndarray:
        """
        images_rgb : (N, H, W, 3) uint8 or float32 RGB
        Returns     : (N, 1280) float32 embeddings
        """
        import tensorflow as tf

        processed = np.stack([
            self.preprocess(images_rgb[i]) for i in range(len(images_rgb))
        ], axis=0)

        tensor = tf.constant(processed, dtype=tf.float32)
        embeddings = self._model(tensor).numpy()
        return embeddings.astype(np.float32)


# ─── Batch embedding helper ───────────────────────────────────────────────────

def extract_embeddings_in_batches(
    extractor: FeatureExtractorBase,
    images: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run feature extraction in fixed-size batches to avoid OOM.

    Parameters
    ----------
    extractor  : FeatureExtractorBase instance
    images     : (N, H, W, 3) array of RGB images
    batch_size : number of images per GPU/CPU batch

    Returns
    -------
    (N, embedding_dim) float32 numpy array
    """
    n = len(images)
    all_embeddings: list[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]
        embs = extractor.extract_batch(batch)
        all_embeddings.append(embs)
        logger.debug("Embedded %d / %d images.", end, n)

    return np.vstack(all_embeddings)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_extractor(strategy: str = "mobilenetv2") -> FeatureExtractorBase:
    """
    Instantiate a feature extractor.

    Parameters
    ----------
    strategy : "mobilenetv2" | "efficientnet_hub"
    """
    strategy = strategy.lower()

    if strategy == "mobilenetv2":
        return MobileNetV2Extractor()

    if strategy in ("efficientnet_hub", "efficientnet"):
        try:
            return EfficientNetHubExtractor()
        except Exception as exc:
            logger.error("Failed to load EfficientNet Hub model: %s", exc)
            logger.warning("Falling back to MobileNetV2.")
            return MobileNetV2Extractor()

    raise ValueError(
        f"Unknown extractor strategy: {strategy!r}. "
        "Choose 'mobilenetv2' or 'efficientnet_hub'."
    )
