from __future__ import annotations
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

CONTENT_DIR = Path(
    os.environ.get(
        "MANZAI_CONTENT_DIR",
        WORKSPACE_ROOT / "web" / "src" / "content" / "manzai",
    )
)
WORK_DIR = REPO_ROOT / "work"
LEXICON_PATH = REPO_ROOT / "lexicon.yaml"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

# ASR_BACKEND: auto | sensevoice | mlx | faster
#   auto = sensevoice (best for zh/ja, multilingual, fast on small GPUs)
#   sensevoice = FunAudioLLM SenseVoiceSmall via funasr (zh/ja/en/yue/ko)
#   mlx = Apple Silicon GPU via mlx-whisper
#   faster = faster-whisper (CTranslate2, CPU/CUDA)
# WHISPER_BACKEND retained as alias for backward compat.
ASR_BACKEND = os.environ.get(
    "ASR_BACKEND", os.environ.get("WHISPER_BACKEND", "auto")
).lower()
if ASR_BACKEND == "auto":
    ASR_BACKEND = "sensevoice"
WHISPER_BACKEND = ASR_BACKEND  # alias

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
# MLX model repo override; default uses the mlx-community pre-converted model
MLX_WHISPER_REPO = os.environ.get(
    "MLX_WHISPER_REPO",
    f"mlx-community/whisper-{WHISPER_MODEL}-mlx",
)

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Diarization device for pyannote (PyTorch). Auto-pick MPS / CUDA / CPU.
PYANNOTE_DEVICE = os.environ.get("PYANNOTE_DEVICE", "auto").lower()

DEFAULT_CONTRIBUTOR = os.environ.get("MANZAI_CONTRIBUTOR", "wheatfox")
