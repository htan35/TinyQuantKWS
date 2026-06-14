import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.config import ALL_LABELS, DEFAULT_ARCHIVE_PATH, MODEL_DIR, PLOT_DIR
from src.dataset import decode_audio_file, ensure_dataset_ready, waveform_to_log_mel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "best_model.keras")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not args.wav.exists():
        raise FileNotFoundError(
            f"WAV file not found: {args.wav}\n"
            "Pass a real file path, for example:\n"
            "python predict.py --wav \"D:\\CVprojects\\DL-Project\\data\\yes\\004ae714_nohash_0.wav\" --plot"
        )
    if not args.wav.is_file():
        raise FileNotFoundError(f"Expected a WAV file but got: {args.wav}")

    ensure_dataset_ready(args.archive)
    model = tf.keras.models.load_model(args.model)

    waveform = decode_audio_file(tf.constant(str(args.wav)), tf.constant(0), tf.constant(False))
    features = waveform_to_log_mel(waveform)
    probabilities = model.predict(tf.expand_dims(features, axis=0), verbose=0)[0]

    ranked = sorted(zip(ALL_LABELS, probabilities.tolist()), key=lambda item: item[1], reverse=True)
    print("Top predictions:")
    for label, score in ranked[:5]:
        print(f"{label:>8}: {score:.4f}")

    if args.plot:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(np.squeeze(features.numpy()).T, origin="lower", aspect="auto", cmap="magma")
        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        fig.tight_layout()
        output_path = PLOT_DIR / "prediction_spectrogram.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()