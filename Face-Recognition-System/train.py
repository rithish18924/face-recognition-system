"""
train.py -- Face Recognition Training Pipeline.

Supports:
  1. LFW dataset (sklearn.datasets.fetch_lfw_people)
  2. Custom dataset directory (organized as root/<person_name>/<image_file>)
  3. Hybrid mode (LFW + custom identities such as Ratan Tata)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.dataset import load_lfw, load_custom_directory, print_dataset_summary, DatasetBundle
from src.feature_extractor import build_extractor, extract_embeddings_in_batches
from src.trainer import train_classifiers, compute_class_centroids, save_model_artifact
from src.utils import get_logger

logger = get_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Face Recognition Classifier")
    parser.add_argument(
        "--dataset",
        choices=["lfw", "custom", "hybrid"],
        default="lfw" if config.CUSTOM_DATASET_DIR is None else "hybrid",
        help="Dataset type to load (default: lfw, or hybrid if custom dataset exists)",
    )
    parser.add_argument(
        "--custom-dir",
        type=str,
        default=str(config.DATA_DIR / "custom_faces"),
        help="Path to custom directory containing <person_name>/ folders",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=config.LFW_MIN_FACES_PER_PERSON,
        help="Minimum faces per person to include",
    )
    parser.add_argument(
        "--classifier",
        choices=["svm_linear", "logistic_regression", "svm_rbf", "random_forest", "all"],
        default=config.CLASSIFIER_TYPE,
        help="Classifier type to train",
    )
    parser.add_argument(
        "--extractor",
        choices=["mobilenetv2", "efficientnet_hub"],
        default=config.FEATURE_EXTRACTOR,
        help="Feature extractor backbone",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputing embeddings even if cache exists",
    )
    return parser.parse_args()


def load_data(args) -> DatasetBundle:
    custom_path = Path(args.custom_dir)
    has_custom = custom_path.is_dir() and any(p.is_dir() for p in custom_path.iterdir())

    if args.dataset == "custom":
        if not has_custom:
            raise FileNotFoundError(f"Custom dataset directory not found or empty: {custom_path}")
        return load_custom_directory(custom_path, min_faces_per_person=args.min_faces, random_seed=config.RANDOM_SEED)

    if args.dataset == "hybrid":
        lfw_bundle = load_lfw(
            min_faces_per_person=args.min_faces,
            color=config.LFW_COLOR,
            resize=config.LFW_RESIZE,
            random_seed=config.RANDOM_SEED,
        )
        if not has_custom:
            logger.warning("No custom directory found at %s. Falling back to pure LFW.", custom_path)
            return lfw_bundle

        custom_bundle = load_custom_directory(custom_path, min_faces_per_person=3, random_seed=config.RANDOM_SEED)
        return _merge_bundles(lfw_bundle, custom_bundle)

    # Pure LFW
    return load_lfw(
        min_faces_per_person=args.min_faces,
        color=config.LFW_COLOR,
        resize=config.LFW_RESIZE,
        random_seed=config.RANDOM_SEED,
    )


def _merge_bundles(b1: DatasetBundle, b2: DatasetBundle) -> DatasetBundle:
    """Merge two DatasetBundles (e.g. LFW + custom identities)."""
    import cv2
    from sklearn.preprocessing import LabelEncoder

    H, W = 112, 92  # common standard resolution for raw crops
    def _rescale(arr):
        return np.stack([cv2.resize(img, (W, H)) for img in arr], axis=0)

    X_train1, X_train2 = _rescale(b1.X_train), _rescale(b2.X_train)
    X_val1, X_val2 = _rescale(b1.X_val), _rescale(b2.X_val)
    X_test1, X_test2 = _rescale(b1.X_test), _rescale(b2.X_test)

    # Names
    y1_train_names = [b1.target_names[i] for i in b1.y_train]
    y1_val_names = [b1.target_names[i] for i in b1.y_val]
    y1_test_names = [b1.target_names[i] for i in b1.y_test]

    y2_train_names = [b2.target_names[i] for i in b2.y_train]
    y2_val_names = [b2.target_names[i] for i in b2.y_val]
    y2_test_names = [b2.target_names[i] for i in b2.y_test]

    all_names = sorted(set(b1.target_names + b2.target_names))
    le = LabelEncoder()
    le.fit(all_names)

    X_train = np.concatenate([X_train1, X_train2], axis=0)
    X_val = np.concatenate([X_val1, X_val2], axis=0)
    X_test = np.concatenate([X_test1, X_test2], axis=0)

    y_train = le.transform(y1_train_names + y2_train_names)
    y_val = le.transform(y1_val_names + y2_val_names)
    y_test = le.transform(y1_test_names + y2_test_names)

    summary = {
        "source": f"Hybrid ({b1.summary.get('source')} + {b2.summary.get('source')})",
        "total_images": len(X_train) + len(X_val) + len(X_test),
        "n_classes": len(all_names),
        "image_shape_hw": (H, W),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=le,
        target_names=all_names,
        image_shape=(H, W),
        summary=summary,
    )


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("FACE RECOGNITION -- TRAINING PIPELINE")
    logger.info("  Extractor  : %s", args.extractor)
    logger.info("  Classifier : %s", args.classifier)
    logger.info("  Dataset    : %s", args.dataset)
    logger.info("  Seed       : %d", config.RANDOM_SEED)
    logger.info("=" * 60)

    cache_path = config.EMBEDDING_CACHE_PATH

    if config.USE_EMBEDDING_CACHE and cache_path.exists() and not args.no_cache and args.dataset == "lfw":
        logger.info("Loading cached embeddings from %s ...", cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        X_train = cache["X_train"]
        y_train = cache["y_train"]
        X_val = cache["X_val"]
        y_val = cache["y_val"]
        X_test = cache["X_test"]
        y_test = cache["y_test"]

        from src.utils import load_pickle
        label_encoder = load_pickle(config.LABEL_ENCODER_PATH)
        n_classes = len(label_encoder.classes_)
        logger.info("Cache loaded: train=%d  val=%d  test=%d  classes=%d", len(X_train), len(X_val), len(X_test), n_classes)
    else:
        bundle = load_data(args)
        print_dataset_summary(bundle)
        label_encoder = bundle.label_encoder
        n_classes = len(label_encoder.classes_)

        logger.info("Building feature extractor (%s) ...", args.extractor)
        extractor = build_extractor(args.extractor)

        logger.info("Extracting training embeddings (%d images) ...", len(bundle.X_train))
        X_train = extract_embeddings_in_batches(extractor, bundle.X_train, batch_size=config.BATCH_SIZE)

        logger.info("Extracting validation embeddings (%d images) ...", len(bundle.X_val))
        X_val = extract_embeddings_in_batches(extractor, bundle.X_val, batch_size=config.BATCH_SIZE)

        logger.info("Extracting test embeddings (%d images) ...", len(bundle.X_test))
        X_test = extract_embeddings_in_batches(extractor, bundle.X_test, batch_size=config.BATCH_SIZE)

        y_train = bundle.y_train
        y_val = bundle.y_val
        y_test = bundle.y_test

        # Save embeddings cache
        np.savez_compressed(
            cache_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
        )
        logger.info("Saved embedding cache -> %s", cache_path)

        # Save test embeddings for evaluation
        test_emb_path = config.ARTIFACTS_DIR / "test_embeddings.npz"
        np.savez_compressed(test_emb_path, X_test=X_test, y_test=y_test)

    # Compute class centroids for prototype verification
    logger.info("Computing %d class centroids ...", n_classes)
    centroids = compute_class_centroids(X_train, y_train, n_classes)

    # Train classifiers
    best_result = train_classifiers(
        X_train, y_train, X_val, y_val,
        label_encoder=label_encoder,
        random_seed=config.RANDOM_SEED,
        classifier_type=args.classifier,
    )

    # Save trained model artifact
    save_model_artifact(
        result=best_result,
        label_encoder=label_encoder,
        extractor_strategy=args.extractor,
        model_path=config.MODEL_ARTIFACT_PATH,
        encoder_path=config.LABEL_ENCODER_PATH,
        centroids=centroids,
    )

    logger.info("Training complete! Model ready for inference.")


if __name__ == "__main__":
    main()
