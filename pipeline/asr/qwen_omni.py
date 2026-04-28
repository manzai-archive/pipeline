"""Qwen3-Omni multimodal: audio + member context → speaker-tagged JSON.

The model listens to the whole audio (≤20 min for qwen3-omni-flash) and
returns segments tagged with the actual performer's name, using both
voice characteristics and linguistic cues (e.g. ツッコミ catchphrases).

Returns (words, detected_language, turns) where turns is pre-populated
so the pipeline can skip pyannote diarization for this backend.
"""
from __future__ import annotations
import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from .transcribe import Word


def _to_mp3(src: Path, bitrate: str = "64k") -> Path:
    """Compress to mono 16kHz mp3 to stay well under the 28 MB JSON-string
    limit on DashScope's input_audio data URI."""
    fd, out = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ac", "1", "-ar", "16000",
            "-b:a", bitrate,
            "-loglevel", "error",
            out,
        ],
        check=True,
    )
    return Path(out)


def _client():
    from openai import OpenAI

    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("VLM_API_KEY")
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError("QWEN_API_KEY / VLM_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=base_url)


def _model_name() -> str:
    # Don't fall through to QWEN_MODEL/VLM_MODEL — those are for the
    # ASR-task model (qwen3-asr-flash) which doesn't accept this prompt
    # format. qwen-omni needs the multimodal-instruction model.
    return os.environ.get("QWEN_OMNI_MODEL") or "qwen3-omni-flash"


def _b64(path: Path, mime: str = "audio/mpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _build_prompt(group_slug: str, content_dir: Path, language: Optional[str]) -> str:
    members_block = ""
    member_names: list[str] = []
    try:
        group_yaml = content_dir.parent / "performers" / f"{group_slug}.yaml"
        if group_yaml.exists():
            data = yaml.safe_load(group_yaml.read_text()) or {}
            members = data.get("members") or []
            for m in members:
                name = m.get("name") or ""
                role = m.get("role") or ""
                if name:
                    member_names.append(name)
                    members_block += f"- {name}"
                    if role:
                        members_block += f" ({role})"
                    members_block += "\n"
    except Exception:
        pass

    if not members_block:
        members_block = "- SPEAKER_A\n- SPEAKER_B\n"
        member_names = ["SPEAKER_A", "SPEAKER_B"]

    cue_hint = ""
    lang = (language or "").lower()
    if lang.startswith("ja"):
        cue_hint = (
            "Use ツッコミ catchphrases (なんでやねん / ちゃうやろ / おかしいやろ / "
            "なに言うてんねん) to identify the ツッコミ when role is given. "
            "ボケ provides setups and absurd statements."
        )
    elif lang.startswith("zh"):
        cue_hint = (
            "用搭档间 setup / 吐槽 的对比，加上声音特征区分两位演员。"
        )

    return f"""You are an expert at transcribing comedy duo (manzai / 漫才) performances.

Performers:
{members_block}
{cue_hint}

Task:
1. Listen to the entire audio.
2. Output a single JSON object (no other text, no markdown fences):
   {{"title": "<short clean title>", "segments": [...]}}
3. The "title" should be the bit's name only — the Japanese 演目名 or
   Chinese 段子名 — without the show name, year, host channel, hashtags,
   or "by <performer>" preamble. Examples of GOOD titles:
   - 「保険の契約」
   - 「人間ドック」
   - 「骗假不留」
   - 「听不懂话的理发师」
   Use the same script (kanji / hangzi) as the audio; surround with 「」.
4. Each segment MUST be: {{"speaker": "<one of the names above>",
   "start": <seconds float>, "end": <seconds float>,
   "text": "<verbatim transcript including 语气词 / fillers / interjections>"}}
5. Split into natural sentence-sized turns (≤30s each). Preserve filler
   words (ええ, あー, 嗯, 哎呀, etc.).
6. Use voice characteristics first; fall back to role-based linguistic
   cues if voices are similar.
7. Do NOT translate; transcribe in the original spoken language."""


# Stash for the most-recent qwen-omni call's title (read by cli.py).
LAST_TITLE: str = ""


def transcribe_qwen_omni(
    audio: Path,
    language: Optional[str] = None,
    group_slug: str = "unknown",
    content_dir: Optional[Path] = None,
) -> tuple[list[Word], str, list]:
    """Returns (words, detected_lang, turns) where turns has speaker == member
    name. Caller should bypass pyannote when turns is non-empty.

    Side-effect: sets module-level LAST_TITLE to the model-generated short
    title; cli.py reads this and uses as title_override when present."""
    from pipeline.diarize.speakers import Turn

    client = _client()
    prompt = _build_prompt(group_slug, content_dir or Path("."), language)

    mp3 = _to_mp3(audio)
    try:
        completion = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {
                    "role": "system",
                    "content": "Output strictly valid JSON. No markdown. No commentary.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _b64(mp3), "format": "mp3"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            modalities=["text"],
        )
    finally:
        mp3.unlink(missing_ok=True)
    raw = (completion.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a top-level object first, then array fallback
        obj_match = re.search(r"\{[\s\S]*\}\s*$", raw)
        arr_match = re.search(r"\[[\s\S]*\]", raw)
        candidate = obj_match.group(0) if obj_match else (arr_match.group(0) if arr_match else None)
        if not candidate:
            raise RuntimeError(f"qwen-omni did not return JSON: {raw[:300]}")
        data = json.loads(candidate)

    title = ""
    if isinstance(data, dict):
        title = str(data.get("title") or "").strip()
        segments = data.get("segments") or []
    else:
        segments = data

    words: list[Word] = []
    turns: list[Turn] = []
    for item in segments:
        speaker = str(item.get("speaker") or "unknown").strip()
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        words.append(Word(start=start, end=end, text=text))
        turns.append(Turn(start=start, end=end, speaker=speaker))

    global LAST_TITLE
    LAST_TITLE = title
    return words, (language or "auto"), turns
