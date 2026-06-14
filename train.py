from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

from src.config import ALL_LABELS, DEFAULT_ARCHIVE_PATH, EPOCHS, LABEL_TO_INDEX, LOG_DIR, MANIFEST_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR, SEED
from src.dataset import build_manifests, create_tf_dataset, ensure_dataset_ready
from src.model import build_keyword_model

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_records(path: Path, limit: int | None) -> int:
    total = sum(1 for _ in open(path, encoding="utf-8")) - 1
    return min(total, limit) if limit else total


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["sparse_categorical_accuracy"], label="train")
    axes[1].plot(history.history["val_sparse_categorical_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(metrics: dict, output_path: Path) -> None:
    lines = [
        "# Speech Commands Project Summary",
        "",
        "## Final Metrics",
        f"- Test accuracy: {metrics['test_accuracy']:.4f}",
        f"- Test loss: {metrics['test_loss']:.4f}",
        f"- Training samples: {metrics['train_examples']}",
        f"- Validation samples: {metrics['val_examples']}",
        f"- Test samples: {metrics['test_examples']}",
        "",
        "## Outputs",
        "- Model: `artifacts/models/best_model.keras`",
        "- Training curves: `artifacts/plots/training_history.png`",
        "- Confusion matrix: `artifacts/plots/confusion_matrix.png`",
        "- Metrics JSON: `artifacts/reports/metrics.json`",
        "- Classification report: `artifacts/reports/classification_report.txt`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    for directory in (MODEL_DIR, PLOT_DIR, REPORT_DIR, LOG_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_dir = ensure_dataset_ready(args.archive)
    manifest_paths = build_manifests(dataset_dir=dataset_dir)

    train_ds = create_tf_dataset(manifest_paths["train"], training=True, batch_size=args.batch_size, limit=args.limit_train)
    val_ds = create_tf_dataset(manifest_paths["val"], training=False, batch_size=args.batch_size, limit=args.limit_val)
    test_ds = create_tf_dataset(manifest_paths["test"], training=False, batch_size=args.batch_size, limit=args.limit_test)

    sample_features, _ = next(iter(train_ds.take(1)))
    input_shape = tuple(sample_features.shape[1:])

    model = build_keyword_model(input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_DIR / "best_model.keras", monitor="val_sparse_categorical_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.CSVLogger(LOG_DIR / "training_log.csv"),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
    plot_history(history, PLOT_DIR / "training_history.png")

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_LABELS))))
    report_text = classification_report(y_true, y_pred, target_names=ALL_LABELS, digits=4, zero_division=0)
    plot_confusion(cm, ALL_LABELS, PLOT_DIR / "confusion_matrix.png")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "sklearn_accuracy": float(accuracy_score(y_true, y_pred)),
        "train_examples": count_records(manifest_paths["train"], args.limit_train),
        "val_examples": count_records(manifest_paths["val"], args.limit_val),
        "test_examples": count_records(manifest_paths["test"], args.limit_test),
        "labels": ALL_LABELS,
        "label_to_index": LABEL_TO_INDEX,
        "input_shape": list(input_shape),
        "epochs_completed": len(history.history["loss"]),
    }

    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_summary(metrics, REPORT_DIR / "project_summary.md")

    print(json.dumps(metrics, indent=2))
    print(report_text)


if __name__ == "__main__":
    main()