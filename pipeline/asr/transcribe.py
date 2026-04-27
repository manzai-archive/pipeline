from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline import config


@dataclass
class Word:
    start: float
    end: float
    text: str


_model = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(
    audio: Path,
    language: str = "ja",
    initial_prompt: Optional[str] = None,
) -> list[Word]:
    model = _load()
    segments, _info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt or None,
        vad_filter=True,
        beam_size=5,
    )
    out: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            out.append(Word(start=float(w.start), end=float(w.end), text=w.word))
    return out
