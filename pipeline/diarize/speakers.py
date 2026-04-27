from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline import config


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        if not config.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN required for diarization. "
                "Get a token at https://hf.co/settings/tokens and accept "
                "https://hf.co/pyannote/speaker-diarization-3.1"
            )
        from pyannote.audio import Pipeline

        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.HF_TOKEN,
        )
    return _pipeline


def diarize(audio: Path, num_speakers: Optional[int] = 2) -> list[Turn]:
    pl = _load()
    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    diar = pl(str(audio), **kwargs)
    out: list[Turn] = []
    for segment, _, speaker in diar.itertracks(yield_label=True):
        out.append(
            Turn(start=float(segment.start), end=float(segment.end), speaker=speaker)
        )
    return out
