from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from src.config import DEFAULT_ARCHIVE_PATH, MANIFEST_DIR, MODEL_DIR
from src.dataset import build_manifests, create_tf_dataset, ensure_dataset_ready


DEFAULT_KERAS_MODEL = MODEL_DIR / "best_model.keras"
DEFAULT_FLOAT32_TFLITE = MODEL_DIR / "model_float32.tflite"
DEFAULT_INT8_TFLITE = MODEL_DIR / "model_int8.tflite"
DEFAULT_INT4_TFLITE = MODEL_DIR / "model_int4.tflite"
EXPORT_DIR = MODEL_DIR / "tmp_saved_model_for_tflite"


def representative_dataset_from_tf_dataset(
    train_ds: tf.data.Dataset,
    num_samples: int = 100,
) -> Iterable[list[np.ndarray]]:
    """Yield float32 samples for TFLite calibration.

    This is the boilerplate representative dataset generator you can reuse with your
    existing train_ds from src.dataset.create_tf_dataset(...).
    """
    yielded = 0
    for features, _ in train_ds:
        batch = features.numpy().astype(np.float32)
        for sample in batch:
            yield [np.expand_dims(sample, axis=0)]
            yielded += 1
            if yielded >= num_samples:
                return


def build_default_representative_dataset(
    archive_path: Path,
    num_samples: int = 100,
    limit_train: int = 512,
) -> callable:
    """Create a representative dataset generator using the existing project pipeline."""
    dataset_dir = ensure_dataset_ready(archive_path)
    manifest_paths = build_manifests(dataset_dir=dataset_dir, manifest_dir=MANIFEST_DIR)
    train_ds = create_tf_dataset(manifest_paths["train"], training=False, batch_size=1, limit=limit_train)
    return lambda: representative_dataset_from_tf_dataset(train_ds, num_samples=num_samples)


def export_saved_model(keras_model_path: Path, export_dir: Path) -> Path:
    """Export a Keras model to SavedModel because Keras 3 -> TFLite direct conversion can fail."""
    export_dir = Path(export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    model = tf.keras.models.load_model(keras_model_path)
    model.export(str(export_dir))
    return export_dir


def convert_float32(saved_model_dir: Path, output_path: Path) -> Path:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def convert_int8(saved_model_dir: Path, output_path: Path, representative_dataset: callable) -> Path:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def convert_int4(saved_model_dir: Path, output_path: Path, representative_dataset: callable) -> tuple[Path, str]:
    """Best-effort experimental Int4 conversion.

    TensorFlow Lite does not expose a fully stable public Int4 PTQ API in all TF 2.14+
    builds. This function first tries experimental low-bit conversion flags. If the local
    TensorFlow build does not support it, an informative RuntimeError is raised.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    experimental_flags = []
    if hasattr(converter, "_experimental_low_bit_qat"):
        converter._experimental_low_bit_qat = True
        experimental_flags.append("_experimental_low_bit_qat")
    if hasattr(converter, "_experimental_reduce_type_precision"):
        converter._experimental_reduce_type_precision = True
        experimental_flags.append("_experimental_reduce_type_precision")

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        raise RuntimeError(
            "Experimental Int4 conversion failed in the installed TensorFlow build. "
            "Keep the script, but expect this path to require either a newer TF/TFLite build, "
            "TensorFlow Model Optimization Toolkit, or weight-only low-bit tooling."
        ) from exc

    output_path.write_bytes(tflite_model)
    return output_path, ", ".join(experimental_flags) if experimental_flags else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the trained KWS model into TFLite variants.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--keras-model", type=Path, default=DEFAULT_KERAS_MODEL)
    parser.add_argument("--num-calibration-samples", type=int, default=100)
    parser.add_argument("--limit-train", type=int, default=512)
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    saved_model_dir = export_saved_model(args.keras_model, EXPORT_DIR)
    representative_dataset = build_default_representative_dataset(
        archive_path=args.archive,
        num_samples=args.num_calibration_samples,
        limit_train=args.limit_train,
    )

    outputs = {}
    outputs["float32"] = str(convert_float32(saved_model_dir, DEFAULT_FLOAT32_TFLITE))
    outputs["int8"] = str(convert_int8(saved_model_dir, DEFAULT_INT8_TFLITE, representative_dataset))

    int4_metadata = {"status": "not_run"}
    try:
        int4_path, flags_used = convert_int4(saved_model_dir, DEFAULT_INT4_TFLITE, representative_dataset)
        outputs["int4"] = str(int4_path)
        int4_metadata = {"status": "ok", "flags": flags_used}
    except RuntimeError as exc:
        int4_metadata = {"status": "failed", "reason": str(exc)}

    metadata = {
        "keras_model": str(args.keras_model),
        "outputs": outputs,
        "int4_metadata": int4_metadata,
    }
    (MODEL_DIR / "quantization_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()