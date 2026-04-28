from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline import config


@dataclass
class Word:
    """A unit of transcribed audio with start/end timestamps. Despite the
    name, for VAD-segmented backends (SenseVoice) each Word may actually
    be a full sentence — postprocess.format groups by speaker turn either
    way."""

    start: float
    end: float
    text: str


# ---------- SenseVoice (default; multilingual, punctuation-aware) ----------

_sense_model = None


def _load_sense():
    global _sense_model
    if _sense_model is None:
        from funasr import AutoModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sense_model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            hub="hf",
            disable_update=True,
        )
    return _sense_model


_SENSE_LANG_MAP = {
    "zh": "zn",
    "zh-cn": "zn",
    "zh-tw": "zn",
    "yue": "yue",
    "ja": "ja",
    "en": "en",
    "ko": "ko",
}


def _transcribe_sensevoice(
    audio: Path, language: Optional[str], _initial_prompt: Optional[str]
) -> tuple[list[Word], str]:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    model = _load_sense()
    sense_lang = _SENSE_LANG_MAP.get((language or "auto").lower(), "auto")
    res = model.generate(
        input=str(audio),
        language=sense_lang,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        output_timestamp=True,
    )

    words: list[Word] = []
    detected = language or "auto"
    for r in res:
        # Try to use sentence-level timestamps if present.
        sentences = r.get("sentence_info") or []
        if sentences:
            for s in sentences:
                txt = rich_transcription_postprocess(s.get("text", ""))
                if not txt.strip():
                    continue
                words.append(
                    Word(
                        start=float(s.get("start", 0)) / 1000.0,
                        end=float(s.get("end", 0)) / 1000.0,
                        text=txt,
                    )
                )
        else:
            txt = rich_transcription_postprocess(r.get("text", ""))
            if txt.strip():
                words.append(Word(start=0.0, end=0.0, text=txt))
        if "language" in r:
            detected = r["language"]
    return words, detected


# ---------- mlx-whisper (Apple Silicon GPU) ----------

def _transcribe_mlx(audio, language, initial_prompt) -> tuple[list[Word], str]:
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
            words.append(
                Word(start=float(w["start"]), end=float(w["end"]), text=w.get("word", ""))
            )
    return words, detected


# ---------- faster-whisper (CTranslate2; CPU or CUDA fallback) ----------

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


def _transcribe_faster(audio, language, initial_prompt) -> tuple[list[Word], str]:
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


def transcribe(
    audio: Path,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> tuple[list[Word], str]:
    """Return (words, detected_language). language=None → auto-detect."""
    backend = config.ASR_BACKEND
    if backend == "sensevoice":
        return _transcribe_sensevoice(audio, language, initial_prompt)
    if backend == "mlx":
        return _transcribe_mlx(audio, language, initial_prompt)
    return _transcribe_faster(audio, language, initial_prompt)
