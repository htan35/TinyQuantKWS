from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import DEFAULT_ARCHIVE_PATH, DESIRED_SAMPLES, MANIFEST_DIR, MODEL_DIR, REPORT_DIR, SAMPLE_RATE
from src.dataset import build_manifests, ensure_dataset_ready, waveform_to_log_mel
from src.noise import load_audio_mono, pad_or_trim_audio, sample_and_mix_noise


DEFAULT_NOISE_DIR = Path(__file__).resolve().parent / "data" / "noise_samples"
DEFAULT_FLOAT32_TFLITE = MODEL_DIR / "model_float32.tflite"
DEFAULT_INT8_TFLITE = MODEL_DIR / "model_int8.tflite"
DEFAULT_INT4_TFLITE = MODEL_DIR / "model_int4.tflite"
DEFAULT_SNR_LEVELS = [20, 10, 0, -5]
OUTPUT_TABLE_DIR = Path(__file__).resolve().parent / "results" / "tables"


class TFLiteRunner:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.file_size_kb = self.model_path.stat().st_size / 1024.0

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        input_tensor = np.expand_dims(features.astype(np.float32), axis=0)
        scale, zero_point = self.input_details["quantization"]
        if self.input_details["dtype"] in (np.int8, np.uint8) and scale > 0:
            input_tensor = np.round(input_tensor / scale + zero_point).astype(self.input_details["dtype"])
        else:
            input_tensor = input_tensor.astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        start = time.perf_counter()
        self.interpreter.invoke()
        latency_ms = (time.perf_counter() - start) * 1000.0

        output = self.interpreter.get_tensor(self.output_details["index"])
        out_scale, out_zero_point = self.output_details["quantization"]
        if self.output_details["dtype"] in (np.int8, np.uint8) and out_scale > 0:
            output = (output.astype(np.float32) - out_zero_point) * out_scale
        predicted_index = int(np.argmax(output, axis=1)[0])
        return predicted_index, latency_ms


def load_manifest_rows(manifest_path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    if limit:
        frame = frame.iloc[:limit].copy()
    return frame.reset_index(drop=True)


def load_manifest_waveform(row: pd.Series) -> np.ndarray:
    waveform = load_audio_mono(Path(row["filepath"]), SAMPLE_RATE)
    if int(row.get("is_silence", 0)) == 1:
        start = int(row.get("start_sample", 0))
        waveform = waveform[start : start + DESIRED_SAMPLES]
    return pad_or_trim_audio(waveform, DESIRED_SAMPLES)


def waveform_to_features(waveform: np.ndarray) -> np.ndarray:
    tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    return waveform_to_log_mel(tensor).numpy().astype(np.float32)


def evaluate_condition(
    manifest_frame: pd.DataFrame,
    runners: dict[str, TFLiteRunner],
    snr_label: str,
    noise_dir: Path,
    rng: random.Random,
) -> list[dict]:
    stats = {model_name: {"correct": 0, "total": 0, "latencies": []} for model_name in runners}

    for _, row in manifest_frame.iterrows():
        clean_waveform = load_manifest_waveform(row)
        if snr_label == "clean" or int(row.get("is_silence", 0)) == 1:
            waveform = clean_waveform
        else:
            snr_db = float(snr_label)
            waveform, _ = sample_and_mix_noise(clean_waveform, noise_dir, snr_db, SAMPLE_RATE, rng=rng)

        features = waveform_to_features(waveform)
        label_index = int(row["label_index"])

        for model_name, runner in runners.items():
            pred, latency_ms = runner.predict(features)
            stats[model_name]["correct"] += int(pred == label_index)
            stats[model_name]["total"] += 1
            stats[model_name]["latencies"].append(latency_ms)

    rows = []
    for model_name, values in stats.items():
        accuracy = values["correct"] / max(values["total"], 1)
        avg_latency = float(np.mean(values["latencies"])) if values["latencies"] else float("nan")
        rows.append(
            {
                "model": model_name,
                "condition": snr_label,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
                "avg_latency_ms": avg_latency,
                "num_samples": values["total"],
                "model_size_kb": runners[model_name].file_size_kb,
            }
        )
    return rows


def summarize_results(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    accuracy_table = results.pivot(index="condition", columns="model", values="accuracy_percent").reset_index()
    latency_table = results.pivot(index="condition", columns="model", values="avg_latency_ms").reset_index()

    clean_map = results[results["condition"] == "clean"].set_index("model")["accuracy"]
    robustness_rows = []
    for condition in [c for c in results["condition"].unique() if c != "clean"]:
        for model_name in results["model"].unique():
            current_acc = results[(results["condition"] == condition) & (results["model"] == model_name)]["accuracy"].iloc[0]
            robustness_rows.append(
                {
                    "condition": condition,
                    "model": model_name,
                    "robustness_drop": float(clean_map[model_name] - current_acc),
                    "robustness_drop_percent_points": float((clean_map[model_name] - current_acc) * 100.0),
                }
            )
    robustness_long = pd.DataFrame(robustness_rows)
    robustness_drop_table = robustness_long.pivot(index="condition", columns="model", values="robustness_drop_percent_points").reset_index()

    deployment_rows = []
    for model_name, group in results.groupby("model"):
        clean_acc = float(clean_map.get(model_name, np.nan))
        noisy_group = group[group["condition"] != "clean"]
        worst_case_acc = float(noisy_group["accuracy"].min()) if not noisy_group.empty else clean_acc
        deployment_rows.append(
            {
                "model": model_name,
                "model_size_kb": float(group["model_size_kb"].iloc[0]),
                "avg_latency_ms": float(group["avg_latency_ms"].mean()),
                "clean_accuracy": clean_acc,
                "clean_accuracy_percent": clean_acc * 100.0,
                "worst_case_accuracy": worst_case_acc,
                "worst_case_accuracy_percent": worst_case_acc * 100.0,
                "max_robustness_drop": float(clean_acc - worst_case_acc),
                "max_robustness_drop_percent_points": float((clean_acc - worst_case_acc) * 100.0),
            }
        )
    deployment_table = pd.DataFrame(deployment_rows)

    detailed_variation = results.copy()
    detailed_variation["accuracy_change_vs_clean_percent_points"] = 0.0
    for model_name in detailed_variation["model"].unique():
        clean_value = float(clean_map[model_name] * 100.0)
        mask = detailed_variation["model"] == model_name
        detailed_variation.loc[mask, "accuracy_change_vs_clean_percent_points"] = detailed_variation.loc[mask, "accuracy_percent"] - clean_value

    return {
        "accuracy_table": accuracy_table,
        "latency_table": latency_table,
        "robustness_drop_table": robustness_drop_table,
        "deployment_table": deployment_table,
        "detailed_variation_table": detailed_variation,
    }


def save_tables(tables: dict[str, pd.DataFrame], results: pd.DataFrame, conditions: list[str], models: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    file_map = {
        "robustness_results.csv": results,
        "robustness_accuracy_table.csv": tables["accuracy_table"],
        "robustness_latency_table.csv": tables["latency_table"],
        "robustness_drop_table.csv": tables["robustness_drop_table"],
        "deployment_summary_table.csv": tables["deployment_table"],
        "noise_variation_table.csv": tables["detailed_variation_table"],
    }

    for filename, frame in file_map.items():
        frame.to_csv(REPORT_DIR / filename, index=False)
        frame.to_csv(OUTPUT_TABLE_DIR / filename, index=False)

    summary = {
        "conditions": conditions,
        "models": models,
        "report_dir": str(REPORT_DIR),
        "results_table_dir": str(OUTPUT_TABLE_DIR),
        "generated_files": list(file_map.keys()),
    }
    (REPORT_DIR / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_TABLE_DIR / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Float32, Int8, and Int4 TFLite KWS models under clean and noisy conditions.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    parser.add_argument("--float32-model", type=Path, default=DEFAULT_FLOAT32_TFLITE)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_TFLITE)
    parser.add_argument("--int4-model", type=Path, default=DEFAULT_INT4_TFLITE)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-levels", type=int, nargs="*", default=DEFAULT_SNR_LEVELS)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    dataset_dir = ensure_dataset_ready(args.archive)
    manifest_paths = build_manifests(dataset_dir=dataset_dir, manifest_dir=MANIFEST_DIR)
    manifest_frame = load_manifest_rows(manifest_paths["test"], limit=args.limit_test)

    from src.config import LABEL_TO_INDEX
    manifest_frame["label_index"] = manifest_frame["label"].map(LABEL_TO_INDEX)

    model_paths = {"float32": args.float32_model, "int8": args.int8_model}
    if args.int4_model.exists():
        model_paths["int4"] = args.int4_model

    runners = {name: TFLiteRunner(path) for name, path in model_paths.items()}
    conditions = ["clean"] + [str(level) for level in args.snr_levels]

    rows = []
    for condition in conditions:
        rows.extend(evaluate_condition(manifest_frame, runners, condition, args.noise_dir, rng))

    results = pd.DataFrame(rows)
    tables = summarize_results(results)
    save_tables(tables, results, conditions, list(runners.keys()))

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    print("\nAccuracy by condition (%)")
    print(tables["accuracy_table"].to_string(index=False))
    print("\nLatency by condition (ms)")
    print(tables["latency_table"].to_string(index=False))
    print("\nRobustness drop from clean (percentage points)")
    print(tables["robustness_drop_table"].to_string(index=False))
    print("\nDeployment summary")
    print(tables["deployment_table"].to_string(index=False))
    print(f"\nSaved report CSV files to: {REPORT_DIR}")
    print(f"Saved paper-ready table CSV files to: {OUTPUT_TABLE_DIR}")


if __name__ == "__main__":
    main()