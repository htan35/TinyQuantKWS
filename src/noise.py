from __future__ import annotations

from pathlib import Path
import random

import librosa
import numpy as np


def pad_or_trim_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim a waveform to a fixed number of samples."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.shape[0] >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - audio.shape[0]), mode="constant").astype(np.float32)


def load_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Load a mono waveform at the requested sample rate."""
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def mix_noise_at_snr(clean_audio: np.ndarray, noise_audio: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mix clean speech with noise at a target SNR in dB.

    Both inputs are assumed to be 1-D float arrays. The returned waveform is clipped
    to [-1, 1] for compatibility with the existing feature extraction pipeline.
    """
    clean = np.asarray(clean_audio, dtype=np.float32).flatten()
    noise = np.asarray(noise_audio, dtype=np.float32).flatten()
    if clean.size == 0:
        raise ValueError("clean_audio must not be empty")
    if noise.size == 0:
        raise ValueError("noise_audio must not be empty")

    if noise.shape[0] < clean.shape[0]:
        repeats = int(np.ceil(clean.shape[0] / noise.shape[0]))
        noise = np.tile(noise, repeats)
    if noise.shape[0] > clean.shape[0]:
        start_max = noise.shape[0] - clean.shape[0]
        start = np.random.randint(0, start_max + 1) if start_max > 0 else 0
        noise = noise[start : start + clean.shape[0]]

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if clean_power <= 1e-12:
        return np.zeros_like(clean)
    if noise_power <= 1e-12:
        return clean.copy()

    desired_noise_power = clean_power / (10.0 ** (target_snr_db / 10.0))
    scale = np.sqrt(desired_noise_power / (noise_power + 1e-12))
    mixed = clean + scale * noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def sample_and_mix_noise(
    clean_audio: np.ndarray,
    noise_dir: Path,
    target_snr_db: float,
    sample_rate: int,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, Path]:
    """Randomly select a local noise file and mix it with the clean waveform."""
    noise_dir = Path(noise_dir)
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(f"No noise WAV files found in {noise_dir}")
    chooser = rng.choice if rng is not None else random.choice
    noise_path = chooser(noise_files)
    noise_audio = load_audio_mono(noise_path, sample_rate)
    mixed = mix_noise_at_snr(clean_audio, noise_audio, target_snr_db)
    return mixed, noise_path