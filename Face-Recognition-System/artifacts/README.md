# Artifacts Directory

Generated evaluation outputs are stored here.

## Files

| File | Description |
|---|---|
| `confusion_matrix.png` | Normalized confusion matrix (true-label rows, predicted columns) |
| `per_class_metrics.png` | Horizontal bar chart of per-class F1 scores |
| `prediction_panel.png` | Grid of test predictions with color-coded correct/incorrect labels |
| `evaluation_report.json` | Full evaluation metrics in machine-readable JSON |
| `embeddings_cache.npz` | Pre-computed train/val/test embeddings (speeds up retraining) |
| `test_embeddings.npz` | Test-set embeddings + labels for fast evaluation |

## Generating artifacts

```bash
python evaluate.py
```

These files are excluded from git (see `.gitignore`).
