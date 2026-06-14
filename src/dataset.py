from __future__ import annotations

import csv
import hashlib
import random
import tarfile
import wave
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    ALL_LABELS,
    BATCH_SIZE,
    COMMAND_LABELS,
    DEFAULT_ARCHIVE_PATH,
    DESIRED_SAMPLES,
    EXTRACTED_DATA_DIR,
    FFT_LENGTH,
    FRAME_LENGTH,
    FRAME_STEP,
    LABEL_TO_INDEX,
    LOWER_HZ,
    MANIFEST_DIR,
    MEL_BINS,
    PLOT_DIR,
    SAMPLE_RATE,
    SEED,
    SILENCE_RATIO,
    UNKNOWN_RATIO,
    UPPER_HZ,
)

AUTOTUNE = tf.data.AUTOTUNE
BACKGROUND_DIR_NAME = "_background_noise_"


def _resolve_dataset_dir(extracted_dir: Path) -> Path:
    if (extracted_dir / "validation_list.txt").exists():
        return extracted_dir
    if (extracted_dir.parent / "validation_list.txt").exists():
        return extracted_dir.parent
    return extracted_dir


def ensure_dataset_ready(archive_path: Path = DEFAULT_ARCHIVE_PATH, extracted_dir: Path = EXTRACTED_DATA_DIR) -> Path:
    extracted_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_dataset_dir(extracted_dir)
    if (resolved / "validation_list.txt").exists():
        return resolved
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extracted_dir.parent)
    resolved = _resolve_dataset_dir(extracted_dir)
    if not (resolved / "validation_list.txt").exists():
        raise FileNotFoundError(f"Speech Commands files were not found after extraction under {extracted_dir.parent}")
    return resolved


def _read_split_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().replace('\\', '/') for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _hash_split(rel_path: str) -> str:
    token = rel_path.split("_nohash_")[0]
    hashed = hashlib.sha1(token.encode("utf-8")).hexdigest()
    bucket = int(hashed, 16) % 100
    if bucket < 10:
        return "val"
    if bucket < 20:
        return "test"
    return "train"


def _wav_duration_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def _collect_audio_files(dataset_dir: Path) -> tuple[list[dict], list[Path]]:
    validation = _read_split_file(dataset_dir / "validation_list.txt")
    testing = _read_split_file(dataset_dir / "testing_list.txt")
    examples: list[dict] = []
    noise_files: list[Path] = []

    for label_dir in sorted(dataset_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name == BACKGROUND_DIR_NAME:
            noise_files.extend(sorted(label_dir.glob("*.wav")))
            continue
        for wav_path in sorted(label_dir.glob("*.wav")):
            rel_path = wav_path.relative_to(dataset_dir).as_posix()
            if rel_path in validation:
                split = "val"
            elif rel_path in testing:
                split = "test"
            else:
                split = _hash_split(rel_path)
            examples.append(
                {
                    "filepath": str(wav_path),
                    "label": label_dir.name,
                    "split": split,
                    "source_label": label_dir.name,
                    "is_silence": 0,
                    "start_sample": 0,
                }
            )
    return examples, noise_files


def _sample_silence_entries(noise_files: list[Path], count: int, split: str, rng: random.Random) -> list[dict]:
    entries: list[dict] = []
    if not noise_files or count <= 0:
        return entries
    for _ in range(count):
        noise_path = rng.choice(noise_files)
        total_samples = _wav_duration_samples(noise_path)
        max_start = max(total_samples - DESIRED_SAMPLES, 0)
        start_sample = rng.randint(0, max_start) if max_start else 0
        entries.append(
            {
                "filepath": str(noise_path),
                "label": "silence",
                "split": split,
                "source_label": BACKGROUND_DIR_NAME,
                "is_silence": 1,
                "start_sample": start_sample,
            }
        )
    return entries


def _plot_class_distribution(records: list[dict]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter(record["label"] for record in records)
    ordered_counts = [counts[label] for label in ALL_LABELS]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ALL_LABELS, ordered_counts, color="#1f77b4")
    ax.set_title("Training Manifest Class Distribution")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def build_manifests(
    dataset_dir: Path = EXTRACTED_DATA_DIR,
    manifest_dir: Path = MANIFEST_DIR,
    unknown_ratio: float = UNKNOWN_RATIO,
    silence_ratio: float = SILENCE_RATIO,
    seed: int = SEED,
) -> dict[str, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: manifest_dir / f"{split}.csv" for split in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return paths

    dataset_dir = _resolve_dataset_dir(dataset_dir)
    examples, noise_files = _collect_audio_files(dataset_dir)
    rng = random.Random(seed)
    split_buckets = {"train": [], "val": [], "test": []}
    unknown_buckets = {"train": [], "val": [], "test": []}

    for example in examples:
        split = example["split"]
        if example["label"] in COMMAND_LABELS:
            example["label"] = example["source_label"]
            split_buckets[split].append(example)
        else:
            example["label"] = "unknown"
            unknown_buckets[split].append(example)

    final_buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        known_examples = list(split_buckets[split])
        final_records = list(known_examples)
        target_unknown = min(len(unknown_buckets[split]), int(len(known_examples) * unknown_ratio))
        if target_unknown > 0:
            final_records.extend(rng.sample(unknown_buckets[split], target_unknown))
        silence_count = int(len(known_examples) * silence_ratio)
        final_records.extend(_sample_silence_entries(noise_files, silence_count, split, rng))
        rng.shuffle(final_records)
        final_buckets[split] = final_records

        with paths[split].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "split", "source_label", "is_silence", "start_sample"])
            writer.writeheader()
            writer.writerows(final_records)

    pd.concat([pd.DataFrame(final_buckets[split]) for split in ("train", "val", "test")], ignore_index=True).to_csv(
        manifest_dir / "all_examples.csv", index=False
    )
    _plot_class_distribution(final_buckets["train"])
    return paths


def _pad_or_trim(waveform: tf.Tensor) -> tf.Tensor:
    waveform = waveform[:DESIRED_SAMPLES]
    padding = DESIRED_SAMPLES - tf.shape(waveform)[0]
    waveform = tf.pad(waveform, [[0, padding]])
    waveform.set_shape([DESIRED_SAMPLES])
    return waveform


def decode_audio_file(filepath: tf.Tensor, start_sample: tf.Tensor, is_silence: tf.Tensor) -> tf.Tensor:
    audio_binary = tf.io.read_file(filepath)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    sample_rate = tf.cast(sample_rate, tf.int32)
    tf.debugging.assert_equal(sample_rate, SAMPLE_RATE, message="Expected 16kHz audio")

    start_sample = tf.cast(start_sample, tf.int32)
    is_silence = tf.cast(is_silence, tf.bool)

    def _slice_silence() -> tf.Tensor:
        end_sample = tf.minimum(start_sample + DESIRED_SAMPLES, tf.shape(waveform)[0])
        sliced = waveform[start_sample:end_sample]
        return _pad_or_trim(sliced)

    return tf.cond(is_silence, _slice_silence, lambda: _pad_or_trim(waveform))


def augment_waveform(waveform: tf.Tensor) -> tf.Tensor:
    shift = tf.random.uniform([], minval=-1600, maxval=1601, dtype=tf.int32)
    waveform = tf.roll(waveform, shift=shift, axis=0)
    gain = tf.random.uniform([], 0.85, 1.15)
    waveform = waveform * gain
    noise = tf.random.normal(tf.shape(waveform), stddev=0.003)
    waveform = waveform + noise
    return tf.clip_by_value(waveform, -1.0, 1.0)


def waveform_to_log_mel(waveform: tf.Tensor) -> tf.Tensor:
    stft = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    spectrogram = tf.abs(stft) ** 2
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BINS,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ,
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, axes=1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))
    log_mel = tf.math.log(mel_spectrogram + 1e-6)
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)
    normalized = (log_mel - mean) / (std + 1e-6)
    return tf.expand_dims(normalized, axis=-1)


def _prepare_example(record: dict[str, tf.Tensor], training: bool) -> tuple[tf.Tensor, tf.Tensor]:
    waveform = decode_audio_file(record["filepath"], record["start_sample"], record["is_silence"])
    if training:
        waveform = augment_waveform(waveform)
    features = waveform_to_log_mel(waveform)
    label = tf.cast(record["label_index"], tf.int32)
    return features, label


def create_tf_dataset(manifest_path: Path, training: bool, batch_size: int = BATCH_SIZE, limit: int | None = None) -> tf.data.Dataset:
    frame = pd.read_csv(manifest_path)
    frame["label_index"] = frame["label"].map(LABEL_TO_INDEX)
    if limit:
        frame = frame.iloc[:limit].copy()

    records = {
        "filepath": frame["filepath"].astype(str).tolist(),
        "start_sample": frame["start_sample"].astype(np.int32).tolist(),
        "is_silence": frame["is_silence"].astype(np.int32).tolist(),
        "label_index": frame["label_index"].astype(np.int32).tolist(),
    }
    ds = tf.data.Dataset.from_tensor_slices(records)
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda record: _prepare_example(record, training), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds