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


_faster_model = None


def _load_faster():
    global _faster_model
    if _faster_model is None:
        from faster_whisper import WhisperModel

        _faster_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _faster_model


def _transcribe_faster(audio: Path, language, initial_prompt) -> tuple[list[Word], str]:
    model = _load_faster()
    segments, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt or None,
        vad_filter=True,
        beam_size=5,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(start=float(w.start), end=float(w.end), text=w.word))
    detected = getattr(info, "language", language) or language or "ja"
    return words, detected


def _transcribe_mlx(audio: Path, language, initial_prompt) -> tuple[list[Word], str]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(audio),
        path_or_hf_repo=config.MLX_WHISPER_REPO,
        word_timestamps=True,
        initial_prompt=initial_prompt or None,
        language=language,
        verbose=False,
    )
    detected = result.get("language") or language or "ja"
    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words") or []:
            words.append(Word(
                start=float(w["start"]),
                end=float(w["end"]),
                text=w.get("word", ""),
            ))
    return words, detected


def transcribe(
    audio: Path,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> tuple[list[Word], str]:
    """Return (words, detected_language). language=None → auto-detect."""
    if config.WHISPER_BACKEND == "mlx":
        return _transcribe_mlx(audio, language, initial_prompt)
    return _transcribe_faster(audio, language, initial_prompt)
