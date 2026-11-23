# Models Directory

Trained model artifacts are stored here.

## Files

| File | Description |
|---|---|
| `face_recognition_model.pkl` | Trained sklearn Pipeline (StandardScaler + Calibrated LinearSVC) |
| `label_encoder.pkl` | Fitted LabelEncoder mapping integer indices → person names |
| `model_metadata.json` | Training run metadata: classifier, accuracy, dataset info |

## Generating artifacts

```bash
python train.py
```

These files are excluded from git (see `.gitignore`).
To share a trained model, copy `models/` and `artifacts/embeddings_cache.npz`
to the target machine before running `evaluate.py`.
