from __future__ import annotations
import os
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
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
