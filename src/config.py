from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
EXTRACTED_DATA_DIR = DATA_ROOT / "speech_commands_v0.02"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_DIR = ARTIFACTS_DIR / "manifests"
MODEL_DIR = ARTIFACTS_DIR / "models"
PLOT_DIR = ARTIFACTS_DIR / "plots"
REPORT_DIR = ARTIFACTS_DIR / "reports"
LOG_DIR = ARTIFACTS_DIR / "logs"

DEFAULT_ARCHIVE_PATH = Path(r"D:\CVprojects\project\speech_commands_v0.02.tar.gz")

COMMAND_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
ALL_LABELS = COMMAND_LABELS + ["unknown", "silence"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(ALL_LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
DESIRED_SAMPLES = SAMPLE_RATE * CLIP_DURATION_MS // 1000
FRAME_LENGTH = 480
FRAME_STEP = 160
FFT_LENGTH = 512
MEL_BINS = 40
LOWER_HZ = 20.0
UPPER_HZ = 4000.0

BATCH_SIZE = 64
EPOCHS = 12
SEED = 42
UNKNOWN_RATIO = 0.25
SILENCE_RATIO = 0.10
