import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf

from src.config import ALL_LABELS, DEFAULT_ARCHIVE_PATH, MODEL_DIR, PLOT_DIR, REPORT_DIR
from src.dataset import build_manifests, create_tf_dataset, ensure_dataset_ready


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "best_model.keras")
    args = parser.parse_args()

    ensure_dataset_ready(args.archive)
    manifest_paths = build_manifests()
    test_ds = create_tf_dataset(manifest_paths["test"], training=False)

    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = probabilities.argmax(axis=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ALL_LABELS))))
    report_text = classification_report(y_true, y_pred, target_names=ALL_LABELS, digits=4, zero_division=0)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "num_examples": int(len(y_true))}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "evaluation_classification_report.txt").write_text(report_text, encoding="utf-8")
    plot_confusion(cm, ALL_LABELS, PLOT_DIR / "evaluation_confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(report_text)


if __name__ == "__main__":
    main()
