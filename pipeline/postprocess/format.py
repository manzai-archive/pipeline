from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional

import os

import yaml
from slugify import slugify

from pipeline.asr.transcribe import Word
from pipeline.diarize.speakers import Turn
from pipeline.sources.base import FetchResult

DEFAULT_CONTRIBUTOR = os.environ.get("MANZAI_CONTRIBUTOR", "wheatfox")


def _speaker_for(t: float, turns: list[Turn]) -> str:
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    return min(
        turns,
        key=lambda x: min(abs(x.start - t), abs(x.end - t)),
    ).speaker


def _hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _group_into_lines(
    words: list[Word], turns: list[Turn]
) -> list[tuple[str, float, str]]:
    if not words or not turns:
        return []
    lines: list[tuple[str, float, str]] = []
    cur_speaker = _speaker_for((words[0].start + words[0].end) / 2, turns)
    cur_start = words[0].start
    cur_text: list[str] = []
    for w in words:
        mid = (w.start + w.end) / 2
        sp = _speaker_for(mid, turns)
        if sp != cur_speaker:
            joined = "".join(cur_text).strip()
            if joined:
                lines.append((cur_speaker, cur_start, joined))
            cur_speaker, cur_start, cur_text = sp, w.start, []
        cur_text.append(w.text)
    joined = "".join(cur_text).strip()
    if joined:
        lines.append((cur_speaker, cur_start, joined))
    return lines


def _iso_date(yyyymmdd: str) -> str:
    if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return yyyymmdd


def write_script(
    *,
    out_dir: Path,
    fetched: FetchResult,
    words: list[Word],
    turns: list[Turn],
    group_slug: Optional[str],
    title_override: Optional[str],
    tags: list[str],
    sensitivity: str,
    language: str,
) -> Path:
    title = title_override or fetched.title
    group = group_slug or "unknown"
    year = (fetched.upload_date or "")[:4] or datetime.now().strftime("%Y")
    slug = slugify(title, allow_unicode=False, max_length=60) or fetched.raw_id
    fname = f"{group}-{year}-{slug}.md"
    out_path = out_dir / fname

    lines = _group_into_lines(words, turns)
    speakers = sorted({sp for sp, _, _ in lines})

    frontmatter = {
        "title": title,
        "performers": [{"name": group, "members": []}],
        "source": {
            "platform": fetched.platform,
            "url": fetched.source_url,
            "uploader": fetched.uploader,
            "uploaded_at": _iso_date(fetched.upload_date),
            "duration_sec": fetched.duration_sec,
        },
        "language": language,
        "tags": tags,
        "speakers": {s: s for s in speakers},
        "sensitivity": sensitivity,
        "status": "draft",
        "contributed_by": DEFAULT_CONTRIBUTOR,
    }

    body: list[str] = [
        "",
        "<!-- speaker keys are placeholders; map them in frontmatter `speakers` -->",
        "",
    ]
    for sp, t, text in lines:
        body.append(f"**{sp}** [{_hms(t)}] {text}")
        body.append("")

    fm_yaml = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    out_path.write_text(f"---\n{fm_yaml}\n---\n" + "\n".join(body))
    return out_path
