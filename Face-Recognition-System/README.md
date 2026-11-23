# Face Recognition System

A production-quality face recognition pipeline built on **MobileNetV2** embeddings, **LinearSVC**, and the **LFW (Labeled Faces in the Wild)** dataset — with a Flask web interface for live demonstrations.

---

## Architecture & Data Flow

```
Input Image
    │
    ▼
Face Detection (MTCNN)
    │  ┌── No face → reject / "unknown"
    ▼
Face Crop (aligned, clipped)
    │
    ▼
MobileNetV2 Feature Extractor
    │  ImageNet pretrained, frozen
    │  Preprocessing: resize 224×224 → preprocess_input() → [-1,1]
    │  Output: 1280-dim embedding
    ▼
StandardScaler
    │
    ▼
LinearSVC (calibrated probabilities)
    │  Trained only on train split embeddings
    ▼
Prediction + Confidence
    │  Confidence < threshold → "Uncertain / Unknown"
    ▼
Result (identity, confidence, box, status)
```

---

## Dataset

**LFW (Labeled Faces in the Wild)** — downloaded automatically via scikit-learn.

| Property | Value |
|---|---|
| Source | `sklearn.datasets.fetch_lfw_people` |
| Default filter | ≥ 20 images per person |
| Image size | ~62 × 47 px (sklearn resize=0.5) |
| Colour | Yes (RGB) |
| Splits | 70% train / 10% val / 20% test (stratified) |
| Random seed | 42 (fixed for reproducibility) |

---

## Project Structure

```
face-recognition-system/
├── app.py                  Flask web application
├── train.py                Training entry point
├── evaluate.py             Evaluation entry point
├── config.py               All settings in one place
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── dataset.py          LFW + custom directory loading
│   ├── face_detector.py    MTCNN detector (Haar cascade fallback)
│   ├── feature_extractor.py  MobileNetV2 / EfficientNet embeddings
│   ├── trainer.py          Classifier training & comparison
│   ├── predictor.py        End-to-end inference pipeline
│   ├── evaluator.py        Metrics & evaluation report
│   ├── visualization.py    Confusion matrix, F1 chart, prediction panel
│   └── utils.py            Logging, serialization, image helpers
├── models/                 Saved model artifacts (git-ignored)
├── artifacts/              Plots, reports, embedding cache (git-ignored)
├── data/                   LFW cache (git-ignored)
├── logs/
├── tests/
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_predictor.py
│   └── test_evaluation.py
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Installation

```bash
# 1. Clone / download the project
cd "Face Recognition"

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

**Python ≥ 3.10** required. Tested on Python 3.13.3, TensorFlow 2.21.

---

## Training

```bash
# Download LFW and train with default settings
python train.py

# Options
python train.py --min-faces 30          # stricter (fewer, larger classes)
python train.py --classifier all        # compare LinearSVC, LR, RBF-SVM, RF
python train.py --extractor mobilenetv2 # (default)
python train.py --no-cache              # force recompute embeddings
```

Training downloads ~200 MB of LFW data on first run (cached afterwards).
Embedding extraction takes ~5–15 minutes on CPU for the full dataset.

**Artifacts saved:**
- `models/face_recognition_model.pkl` — trained classifier pipeline
- `models/label_encoder.pkl` — fitted LabelEncoder
- `models/model_metadata.json` — training run metadata
- `artifacts/embeddings_cache.npz` — cached embeddings (speeds up retraining)

---

## Evaluation

```bash
python evaluate.py

# Options
python evaluate.py --show-panel 30      # show 30 sample predictions
python evaluate.py --threshold 0.6     # stricter confidence threshold
python evaluate.py --use-detection     # run full MTCNN on test images
```

**Artifacts generated:**
- `artifacts/confusion_matrix.png`
- `artifacts/per_class_metrics.png`
- `artifacts/prediction_panel.png`
- `artifacts/evaluation_report.json`

---

## Inference (Python API)

```python
import sys; sys.path.insert(0, ".")
from src.utils import load_pickle
from src.feature_extractor import build_extractor
from src.face_detector import build_detector
from src.predictor import FaceRecognitionPredictor
import config

pipeline  = load_pickle(config.MODEL_ARTIFACT_PATH)
le        = load_pickle(config.LABEL_ENCODER_PATH)
detector  = build_detector("mtcnn")
extractor = build_extractor("mobilenetv2")

predictor = FaceRecognitionPredictor(
    detector, extractor, pipeline, le,
    unknown_threshold=0.50,
)

result = predictor.predict_from_path("photo.jpg")
print(result.status.name, result.predicted_identity, f"{result.confidence:.1%}")
```

---

## Flask Web Interface

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

| Route | Method | Description |
|---|---|---|
| `/` | GET | Upload page |
| `/predict` | POST | Recognition (JSON + annotated image) |
| `/evaluate` | GET | Evaluation metrics page |
| `/health` | GET | Health check |

**`/predict` response:**
```json
{
  "status": "SUCCESS",
  "predicted_identity": "George W Bush",
  "confidence": 0.8742,
  "all_confidences": {"George W Bush": 0.8742, "Colin Powell": 0.07},
  "face_box": [34, 12, 110, 120],
  "inference_time_ms": 142.3,
  "annotated_image": "<base64 PNG>"
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use **mocked** model components and do **not** require downloading LFW or model weights.

---

## Performance Results

> Results are generated by `python evaluate.py` and reflect actual model predictions.
> Numbers here are placeholders — run the pipeline to get your measured values.

| Metric | Value |
|---|---|
| Accuracy | *run evaluate.py* |
| Macro F1 | *run evaluate.py* |
| Weighted F1 | *run evaluate.py* |
| Avg. inference latency | *run evaluate.py* |

---

## Configuration

Edit [`config.py`](config.py) to change any setting:

| Key | Default | Description |
|---|---|---|
| `LFW_MIN_FACES_PER_PERSON` | 20 | Min images per identity |
| `FEATURE_EXTRACTOR` | `"mobilenetv2"` | `"mobilenetv2"` or `"efficientnet_hub"` |
| `CLASSIFIER_TYPE` | `"svm_linear"` | `"svm_linear"`, `"logistic_regression"`, etc. |
| `UNKNOWN_THRESHOLD` | 0.50 | Confidence below this → "Uncertain" |
| `RANDOM_SEED` | 42 | Fixed for reproducibility |
| `FACE_DETECTOR_STRATEGY` | `"mtcnn"` | `"mtcnn"` or `"haar"` |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Model artifacts not found` | Run `python train.py` first |
| MTCNN import error | `pip install mtcnn` |
| TF Hub download fails | Set `FEATURE_EXTRACTOR = "mobilenetv2"` in config.py |
| Slow training (CPU) | Use `--no-cache` once, then embeddings cache speeds up retries |
| Low accuracy | Increase `LFW_MIN_FACES_PER_PERSON` to 40+ for more data per class |
| Flask file too large | Adjust `MAX_UPLOAD_BYTES` in config.py |

---

## Limitations

1. **Closed-set recognition only** — can only identify people seen during training.
2. **LFW bias** — LFW images are mostly of public figures, frontal, well-lit.
3. **Generic embeddings** — MobileNetV2 was trained on ImageNet, not faces specifically. A face-specific model (FaceNet, ArcFace) would give better results.
4. **Calibration** — probability scores are sigmoid-calibrated, not perfectly calibrated probabilities.
5. **No liveness detection** — system can be fooled by printed photographs.

---

## Privacy Considerations

- This system is for **educational demonstration only**.
- Do not deploy for biometric identification of individuals without consent.
- LFW images are of public figures and licensed for research use.
- Never store uploaded images or recognition results without user consent.
- Model outputs are probabilistic and should not be treated as authoritative identification.

---

## License

MIT License — see `LICENSE` for details.
