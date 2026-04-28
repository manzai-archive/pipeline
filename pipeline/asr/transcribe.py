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


_SENT_END_CHARS = set("。.！!?？…")
_TAG_LANG_RE = __import__("re").compile(r"<\|(zh|zn|ja|en|ko|yue)\|>")


def _transcribe_sensevoice(
    audio: Path, language: Optional[str], _initial_prompt: Optional[str]
) -> tuple[list[Word], str]:
    """SenseVoice returns one big text per VAD-merge group plus character-
    level timestamps. We split that into sentence-level segments by walking
    characters and breaking on terminal punctuation (。.！!?？…)."""
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
        # Pull detected language from in-text tags (the only place SenseVoice exposes it).
        m = _TAG_LANG_RE.search(r.get("text") or "")
        if m:
            code = m.group(1)
            detected = "zh" if code == "zn" else code

        chars = r.get("words") or []
        ts = r.get("timestamp") or []
        if len(chars) != len(ts):
            # Last-resort fallback: dump entire result as one chunk.
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            txt = rich_transcription_postprocess(r.get("text", "")).strip()
            if txt:
                words.append(Word(start=0.0, end=0.0, text=txt))
            continue

        cur_chars: list[str] = []
        cur_start: Optional[float] = None
        for ch, (s_ms, e_ms) in zip(chars, ts):
            if cur_start is None:
                cur_start = s_ms / 1000.0
            cur_chars.append(ch)
            if ch in _SENT_END_CHARS:
                sent = "".join(cur_chars).strip()
                if sent:
                    words.append(Word(start=cur_start, end=e_ms / 1000.0, text=sent))
                cur_chars = []
                cur_start = None
        if cur_chars and cur_start is not None:
            sent = "".join(cur_chars).strip()
            if sent:
                words.append(Word(start=cur_start, end=ts[-1][1] / 1000.0, text=sent))

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


def _resolve_backend(language: Optional[str]) -> str:
    """Backend selection.

    ASR_BACKEND=auto picks per-language local backends:
    - ja → mlx-whisper (Apple Silicon) or faster-whisper (CUDA/CPU)
    - zh / other → SenseVoice (CUDA, multilingual, fast)

    qwen-omni is no longer auto-selected — output truncation can drop
    content for long performances. Use explicit ASR_BACKEND=qwen-omni
    if you want it.
    """
    b = config.ASR_BACKEND
    if b != "auto":
        return b
    lang = (language or "").lower()
    if lang.startswith("ja"):
        return "mlx" if config.IS_APPLE_SILICON else "faster"
    return "sensevoice"


def transcribe(
    audio: Path,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    group_slug: Optional[str] = None,
    content_dir: Optional[Path] = None,
    raw_title: str = "",
) -> tuple[list[Word], str, Optional[list]]:
    """Return (words, detected_language, optional_turns).

    Backends that produce speaker-tagged output (e.g. qwen-omni) return a
    populated `turns` list with member names already assigned, in which case
    the caller should skip pyannote diarization. Other backends return None.
    """
    backend = _resolve_backend(language)
    if backend == "qwen-omni":
        from pipeline.asr.qwen_omni import transcribe_qwen_omni
        return transcribe_qwen_omni(
            audio, language, group_slug or "unknown", content_dir, raw_title=raw_title
        )
    if backend == "qwen":
        from pipeline.asr.qwen import transcribe_qwen
        words, lang = transcribe_qwen(audio, language, initial_prompt)
        return words, lang, None
    if backend == "sensevoice":
        words, lang = _transcribe_sensevoice(audio, language, initial_prompt)
        return words, lang, None
    if backend == "mlx":
        words, lang = _transcribe_mlx(audio, language, initial_prompt)
        return words, lang, None
    words, lang = _transcribe_faster(audio, language, initial_prompt)
    return words, lang, None
