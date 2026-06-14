# Deep Learning Speech Commands Project

End-to-end keyword spotting project built on the Google Speech Commands dataset. The project supports:
- training a baseline KWS model
- evaluation on the test split
- single-file prediction
- TensorFlow Lite export for Float32, Int8, and experimental Int4 variants
- robustness evaluation under local noise samples at multiple SNR levels

The 12 output classes are:
`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`.

## Project Files

- `train.py`: trains the baseline Keras model and saves metrics/plots.
- `evaluate.py`: evaluates the saved Keras model on the clean test split.
- `predict.py`: predicts the class for one WAV file.
- `quantize.py`: exports the trained model to Float32, Int8, and experimental Int4 TFLite formats.
- `evaluate_robustness.py`: evaluates TFLite models on clean audio and noisy audio at multiple SNR levels.
- `src/config.py`: paths and model/audio constants.
- `src/dataset.py`: dataset extraction, manifests, TensorFlow datasets, and log-mel preprocessing.
- `src/model.py`: CNN model definition.
- `src/noise.py`: local noise loading and SNR-based audio mixing helpers.

## Dataset Requirements

You need:
1. The Google Speech Commands archive.
2. Optional local noise files for robustness testing.

Default archive path expected by the project:
`D:\CVprojects\project\speech_commands_v0.02.tar.gz`

Local noise folder expected by the robustness script:
`D:\CVprojects\DL-Project\data\noise_samples`

Example noise files already supported:
- `Babble_1.wav`
- `Cafe_1.wav`
- `Traffic_1.wav`
- `AirConditioner_1.wav`

## Step-by-Step: Run Locally in VS Code Terminal

Open the project folder in VS Code and make sure the terminal is already inside:
`D:\CVprojects\DL-Project`

### Step 1: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Train the baseline model

Quick verification run:

```powershell
python train.py --epochs 3 --limit-train 6000 --limit-val 1200 --limit-test 1200
```

Full training run:

```powershell
python train.py --epochs 12
```

This saves:
- `artifacts/models/best_model.keras`
- `artifacts/plots/training_history.png`
- `artifacts/plots/confusion_matrix.png`
- `artifacts/reports/metrics.json`
- `artifacts/reports/classification_report.txt`

### Step 3: Evaluate the baseline Keras model

```powershell
python evaluate.py
```

### Step 4: Run single-file prediction

```powershell
python predict.py --wav "D:\CVprojects\DL-Project\data\yes\004ae714_nohash_0.wav" --plot
```

### Step 5: Export TensorFlow Lite models

```powershell
python quantize.py
```

This writes:
- `artifacts/models/model_float32.tflite`
- `artifacts/models/model_int8.tflite`
- `artifacts/models/model_int4.tflite`
- `artifacts/models/quantization_summary.json`

Note: the Int4 path is experimental and depends on TensorFlow Lite support in your local build.

### Step 6: Run robustness evaluation

Quick demo run:

```powershell
python evaluate_robustness.py --limit-test 100
```

Full robustness run:

```powershell
python evaluate_robustness.py
```

This writes:
- `artifacts/reports/robustness_results.csv`
- `artifacts/reports/robustness_accuracy_table.csv`
- `artifacts/reports/deployment_summary_table.csv`
- `artifacts/reports/robustness_summary.json`

## Best Demo Order for Instructor

If you want to demonstrate the full project quickly, run these in order:

```powershell
python quantize.py
```

```powershell
python evaluate_robustness.py --limit-test 100
```

If your instructor wants baseline training too, run this before the above two commands:

```powershell
python train.py --epochs 3 --limit-train 6000 --limit-val 1200 --limit-test 1200
```

## Important Output Files to Show

- `artifacts/models/best_model.keras`
- `artifacts/models/model_float32.tflite`
- `artifacts/models/model_int8.tflite`
- `artifacts/models/model_int4.tflite`
- `artifacts/plots/training_history.png`
- `artifacts/plots/confusion_matrix.png`
- `artifacts/reports/metrics.json`
- `artifacts/reports/classification_report.txt`
- `artifacts/reports/robustness_accuracy_table.csv`
- `artifacts/reports/deployment_summary_table.csv`

## Notes

- Audio is processed as 16 kHz mono and padded or trimmed to 1 second.
- Features are normalized log-mel spectrograms.
- The baseline model is a CNN trained with TensorFlow/Keras.
- The robustness script evaluates clean audio and noisy audio at 20 dB, 10 dB, 0 dB, and -5 dB SNR.
- The TFLite Int4 path should be treated as experimental unless separately validated in your exact TensorFlow build.