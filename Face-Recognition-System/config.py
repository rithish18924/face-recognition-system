"""
config.py -- Central configuration for the Face Recognition System.
All paths, hyperparameters, and constants live here.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Ensure Keras and Matplotlib store configurations and caches inside the workspace
os.environ.setdefault("KERAS_HOME", str(BASE_DIR / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

for _d in (MODELS_DIR, ARTIFACTS_DIR, DATA_DIR, LOGS_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# LFW settings
LFW_MIN_FACES_PER_PERSON = 20
LFW_RESIZE = 0.5
LFW_COLOR = True

# Custom dataset directory (e.g. DATA_DIR / "custom_faces")
CUSTOM_DATASET_DIR: str | None = None

RANDOM_SEED = 42

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Face detector settings
FACE_DETECTOR_STRATEGY = "mtcnn"
FACE_MIN_SIZE = 20
FACE_CONFIDENCE_THRESH = 0.9
FACE_SELECT_POLICY = "largest"

# Feature extractor settings
FEATURE_EXTRACTOR = "mobilenetv2"
INPUT_SHAPE = (224, 224)
EMBEDDING_DIM = 1280

EFFICIENTNET_HUB_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
EFFICIENTNET_INPUT_SHAPE = (224, 224)

PIXEL_NORMALIZATION = "mobilenetv2"

# Classifier settings
CLASSIFIER_TYPE = "logistic_regression"
SVM_C = 1.0
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000

# Unknown person rejection & verification thresholds
UNKNOWN_THRESHOLD = 0.40
UNKNOWN_COSINE_SIMILARITY = 0.80
ACCEPTANCE_THRESHOLD = 0.65
MIN_MARGIN_THRESHOLD = 0.15
MIN_COSINE_SIMILARITY = 0.82

# Artifact paths
MODEL_ARTIFACT_PATH = MODELS_DIR / "face_recognition_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
CENTROIDS_PATH = MODELS_DIR / "centroids.npy"

EMBEDDING_CACHE_PATH = ARTIFACTS_DIR / "embeddings_cache.npz"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
METRICS_PLOT_PATH = ARTIFACTS_DIR / "per_class_metrics.png"
PANEL_PLOT_PATH = ARTIFACTS_DIR / "prediction_panel.png"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
MAX_UPLOAD_BYTES = 8388608  # 8 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "face_recognition.log"

# Pipeline caching
USE_EMBEDDING_CACHE = True
BATCH_SIZE = 32
