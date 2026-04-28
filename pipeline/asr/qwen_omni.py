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


_CHUNK_SEC = 180  # 3-min chunks; one ~16k-token JSON response per chunk fits


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


def _to_mp3(src: Path, bitrate: str = "64k", start: float = 0.0, end: Optional[float] = None) -> Path:
    """Compress to mono 16kHz mp3, optionally slicing [start, end). Used to
    stay under the 28 MB JSON-string limit on DashScope's input_audio data URI
    and to keep each chunk's transcript output within max_tokens."""
    fd, out = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    args = ["ffmpeg", "-y"]
    if start > 0:
        args += ["-ss", f"{start}"]
    if end is not None:
        args += ["-to", f"{end}"]
    args += [
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-b:a", bitrate,
        "-loglevel", "error",
        out,
    ]
    subprocess.run(args, check=True)
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


def _salvage_partial(raw: str) -> dict:
    """Best-effort recovery from truncated JSON: extract whatever
    title / form / roles / segments parsed cleanly before the cutoff."""
    title_m = re.search(r'"title"\s*:\s*"([^"]*)"', raw)
    title = title_m.group(1) if title_m else ""
    form_m = re.search(r'"form"\s*:\s*"([^"]*)"', raw)
    form = form_m.group(1) if form_m else ""
    # roles: capture whatever appears between "roles": { ... } until first }
    roles: dict[str, str] = {}
    roles_block = re.search(r'"roles"\s*:\s*\{([^}]*)\}', raw)
    if roles_block:
        for km in re.finditer(r'"([^"]+)"\s*:\s*"([^"]*)"', roles_block.group(1)):
            roles[km.group(1)] = km.group(2)
    seg_pat = re.compile(
        r'\{"speaker"\s*:\s*"([^"]+)"\s*,\s*"start"\s*:\s*([0-9.]+)\s*,\s*"end"\s*:\s*([0-9.]+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    )
    segs = []
    for m in seg_pat.finditer(raw):
        segs.append({
            "speaker": m.group(1),
            "start": float(m.group(2)),
            "end": float(m.group(3)),
            # Re-quote captured fragment as JSON string for proper escape
            # handling without mangling multi-byte UTF-8.
            "text": json.loads('"' + m.group(4) + '"'),
        })
    return {"title": title, "form": form, "roles": roles, "segments": segs}


def _model_name() -> str:
    # Don't fall through to QWEN_MODEL/VLM_MODEL — those are for the
    # ASR-task model (qwen3-asr-flash) which doesn't accept this prompt
    # format. qwen-omni needs the multimodal-instruction model.
    return os.environ.get("QWEN_OMNI_MODEL") or "qwen3-omni-flash"


def _b64(path: Path, mime: str = "audio/mpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _build_prompt(
    group_slug: str,
    content_dir: Path,
    language: Optional[str],
    raw_title: str = "",
) -> str:
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

    title_block = ""
    if raw_title:
        title_block = f'\nPublished video title: "{raw_title}"\n'

    return f"""You are an expert at transcribing comedy performances
(manzai / 漫才, 脱口秀, コント, 相声, 落語, etc.).

Performers:
{members_block}
{cue_hint}{title_block}
Task:
1. Listen to the entire audio.
2. Output a single JSON object (no other text, no markdown fences):
   {{"title": "<bit name>",
     "form": "<comedy form>",
     "roles": {{"<performer>": "<role>", ...}},
     "segments": [...]}}
3. "title" MUST be derived from the published video title above.
   Extract just the bit name (演目名 / 段子名), removing hashtags, channel
   name, year suffixes, show brackets, "by <performer>" prefixes.
   Keep the original wording exactly — do NOT paraphrase. Wrap in 「」.
   If the published title has no extractable bit name, infer one from
   content. Examples:
   - "中川家の寄席2024 「保険の契約」" → "「保険の契約」"
   - "徐浩伦 谭湘文演绎《骗假不留》#脱口秀" → "「骗假不留」"
4. "form" MUST be one of:
   - "manzai"     — Japanese 漫才 (boke + tsukkomi duo)
   - "xiangsheng" — Chinese 相声 (捧哏 + 逗哏)
   - "standup"    — solo standup / スタンダップ / 脱口秀 (single performer)
   - "sketch"     — コント / 小品 / character-driven scene
   - "rakugo"     — 落語 (single seated performer, multiple voiced characters)
   - "other"      — anything else
   Note: Chinese "脱口秀" duos that follow boke/tsukkomi structure are
   "manzai" (e.g. 漫才兄弟 are categorized as manzai even though the show
   is called 脱口秀大会).
5. "roles" maps each performer name to their role in THIS specific bit:
   - manzai: "ツッコミ" / "ボケ" (or for Chinese: "吐槽" / "搞笑" / "捧哏" / "逗哏")
   - standup: "ソロ" (single performer; only one entry)
   - sketch: character role/name they play
   - xiangsheng: "捧哏" / "逗哏"
   Always use the performer's real name (from the list above) as the key.
6. Each segment MUST be: {{"speaker": "<performer name>",
   "start": <seconds float>, "end": <seconds float>,
   "text": "<verbatim transcript including 语气词 / fillers / interjections>"}}
   "speaker" MUST be the literal performer NAME, NEVER a role label.
7. Split into natural sentence-sized turns (≤30s each). Preserve filler
   words (ええ, あー, 嗯, 哎呀).
8. Use voice characteristics first; fall back to role-based linguistic
   cues (ツッコミ catchphrases) if voices are similar.
9. Do NOT translate; transcribe in the original spoken language.
10. The output MUST be a complete, well-formed JSON object — no truncation."""


# Stash for the most-recent qwen-omni call's metadata (read by cli.py /
# format.py). Module-level for simplicity; rewrite if we ever need to
# parallelize qwen-omni calls within one process.
LAST_TITLE: str = ""
LAST_FORM: str = ""
LAST_ROLES: dict[str, str] = {}


def transcribe_qwen_omni(
    audio: Path,
    language: Optional[str] = None,
    group_slug: str = "unknown",
    content_dir: Optional[Path] = None,
    raw_title: str = "",
) -> tuple[list[Word], str, list]:
    """Returns (words, detected_lang, turns) where turns has speaker == member
    name. Caller should bypass pyannote when turns is non-empty.

    Side-effect: sets module-level LAST_TITLE to the model-generated short
    title; cli.py reads this and uses as title_override when present."""
    from pipeline.diarize.speakers import Turn

    client = _client()
    prompt = _build_prompt(
        group_slug, content_dir or Path("."), language, raw_title=raw_title
    )

    duration = _ffprobe_duration(audio)
    chunk_starts = [s for s in range(0, int(duration), _CHUNK_SEC)] or [0]

    chunk_titles: list[str] = []
    chunk_forms: list[str] = []
    chunk_roles: list[dict] = []
    all_segments: list[dict] = []

    for cstart in chunk_starts:
        cend = min(cstart + _CHUNK_SEC, duration)
        mp3 = _to_mp3(audio, start=cstart, end=cend)
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
                max_tokens=16384,
            )
        finally:
            mp3.unlink(missing_ok=True)

        raw = (completion.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()

        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            obj_match = re.search(r"\{[\s\S]*\}\s*$", raw)
            arr_match = re.search(r"\[[\s\S]*\]\s*$", raw)
            candidate = obj_match.group(0) if obj_match else (arr_match.group(0) if arr_match else None)
            if candidate:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
        if data is None:
            data = _salvage_partial(raw)
            if not data.get("segments"):
                raise RuntimeError(
                    f"qwen-omni chunk @{cstart}s gave no parseable segments: {raw[:300]}"
                )
            print(f"  chunk @{cstart}s: salvaged {len(data['segments'])} segments from truncated response")

        ctitle = ""
        cform = ""
        croles: dict = {}
        if isinstance(data, dict):
            ctitle = str(data.get("title") or "").strip()
            cform = str(data.get("form") or "").strip().lower()
            raw_roles = data.get("roles") or {}
            if isinstance(raw_roles, dict):
                croles = {str(k): str(v) for k, v in raw_roles.items()}
            csegs = data.get("segments") or []
        else:
            csegs = data
        if ctitle:
            chunk_titles.append(ctitle)
        if cform:
            chunk_forms.append(cform)
        if croles:
            chunk_roles.append(croles)

        # Offset segment timestamps by chunk start
        for s in csegs:
            try:
                s["start"] = float(s.get("start", 0)) + cstart
                s["end"] = float(s.get("end", s["start"])) + cstart
                all_segments.append(s)
            except (TypeError, ValueError):
                continue

    # Pick the first non-empty chunk title as the canonical bit name
    title = next((t for t in chunk_titles if t), "")
    # Form: pick the most-frequent value across chunks
    form = ""
    if chunk_forms:
        from collections import Counter
        form = Counter(chunk_forms).most_common(1)[0][0]
    # Roles: merge across chunks (later chunks override)
    roles: dict[str, str] = {}
    for r in chunk_roles:
        roles.update(r)

    words: list[Word] = []
    turns: list[Turn] = []
    for item in all_segments:
        speaker = str(item.get("speaker") or "unknown").strip()
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        words.append(Word(start=start, end=end, text=text))
        turns.append(Turn(start=start, end=end, speaker=speaker))

    global LAST_TITLE, LAST_FORM, LAST_ROLES
    LAST_TITLE = title
    LAST_FORM = form
    LAST_ROLES = roles
    return words, (language or "auto"), turns
