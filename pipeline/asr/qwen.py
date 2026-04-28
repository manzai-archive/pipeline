from __future__ import annotations
import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .transcribe import Word


_QWEN_LANG_HINT = {
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh",
    "ja": "ja",
    "en": "en",
    "ko": "ko",
    "yue": "yue",
}

# qwen3-asr-flash hard cap is 5 minutes / 10 MB per request.
_CHUNK_SECONDS = 240  # 4 min, leaves headroom


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip() or 0.0)


def _slice(audio: Path, start: float, end: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start}",
            "-to", f"{end}",
            "-i", str(audio),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            str(dst),
        ],
        check=True,
    )


def _b64_data_url(path: Path, mime: str = "audio/wav") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _client():
    from openai import OpenAI

    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("VLM_API_KEY")
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError("QWEN_API_KEY (or VLM_API_KEY) not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def _model_name() -> str:
    return os.environ.get("QWEN_MODEL") or os.environ.get("VLM_MODEL") or "qwen3-asr-flash"


def _transcribe_chunk(client, audio: Path, language: Optional[str]) -> str:
    """One ≤5-min audio chunk → transcript text via qwen3-asr-flash."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": _b64_data_url(audio),
                        "format": "wav",
                    },
                }
            ],
        }
    ]
    extra_body = {}
    lang_hint = _QWEN_LANG_HINT.get((language or "").lower())
    if lang_hint:
        # DashScope wants a 2-letter code in asr_options.language; if absent,
        # the model auto-detects.
        extra_body["asr_options"] = {"language": lang_hint}

    completion = client.chat.completions.create(
        model=_model_name(),
        messages=messages,
        extra_body=extra_body or None,
    )
    return (completion.choices[0].message.content or "").strip()


def transcribe_qwen(
    audio: Path, language: Optional[str] = None, _initial_prompt: Optional[str] = None
) -> tuple[list[Word], str]:
    """Chunk to ≤4min slices, call qwen3-asr-flash per chunk, split each
    chunk's text into sentence-level Word entries with timestamps estimated
    proportional to character position within the chunk."""
    client = _client()
    duration = _ffprobe_duration(audio)
    if duration <= 0:
        raise RuntimeError(f"ffprobe failed for {audio}")

    chunk_starts = [s for s in range(0, int(duration), _CHUNK_SECONDS)]
    words: list[Word] = []

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + _CHUNK_SECONDS, duration)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _slice(audio, chunk_start, chunk_end, tmp_path)
            text = _transcribe_chunk(client, tmp_path, language)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not text:
            continue
        # Split into sentences on terminal punctuation
        sentences = _split_sentences(text)
        if not sentences:
            words.append(Word(start=float(chunk_start), end=float(chunk_end), text=text))
            continue

        chunk_dur = chunk_end - chunk_start
        total_chars = sum(len(s) for s in sentences) or 1
        char_cursor = 0
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            seg_start = chunk_start + (char_cursor / total_chars) * chunk_dur
            char_cursor += len(s)
            seg_end = chunk_start + (char_cursor / total_chars) * chunk_dur
            words.append(Word(start=seg_start, end=seg_end, text=s_clean))

    return words, language or "auto"


_SENT_END = "。.！!?？…"


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text:
        cur.append(ch)
        if ch in _SENT_END:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return [s for s in out if s.strip()]
