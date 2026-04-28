from __future__ import annotations
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import yaml
from slugify import slugify

from pipeline.asr.transcribe import Word
from pipeline.diarize.speakers import Turn
from pipeline.sources.base import FetchResult

DEFAULT_CONTRIBUTOR = os.environ.get("MANZAI_CONTRIBUTOR", "wheatfox")


def _pipeline_version() -> str:
    try:
        return version("manzai-archive-pipeline")
    except PackageNotFoundError:
        return "unknown"


def _yt_dlp_version() -> str:
    try:
        out = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


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


_SENTENCE_END = "。.！!?？…ですよねかな"  # JP/zh/en sentence-final markers
_LONG_GAP_SEC = 0.8  # silence between words → likely sentence boundary
_MIN_LINE_CHARS = 2


def _ends_sentence(text: str) -> bool:
    if not text:
        return False
    t = text.rstrip()
    if not t:
        return False
    return t[-1] in "。.！!?？…"


def _group_into_lines(
    words: list[Word], turns: list[Turn]
) -> list[tuple[str, float, str]]:
    """Group words into lines, breaking on:
       - speaker change
       - >0.8s silence between consecutive words
       - previous word ended with sentence-final punctuation
       This produces readable short turns even when pyannote merges multiple
       short exchanges into one big speaker segment."""
    if not words or not turns:
        return []
    lines: list[tuple[str, float, str]] = []
    cur_speaker = _speaker_for((words[0].start + words[0].end) / 2, turns)
    cur_start = words[0].start
    cur_text: list[str] = []
    prev_end = words[0].start

    def flush():
        joined = "".join(cur_text).strip()
        if len(joined) >= _MIN_LINE_CHARS:
            lines.append((cur_speaker, cur_start, joined))

    for w in words:
        mid = (w.start + w.end) / 2
        sp = _speaker_for(mid, turns)
        gap = w.start - prev_end
        prev_text = "".join(cur_text)
        speaker_change = sp != cur_speaker
        silence_break = gap > _LONG_GAP_SEC
        sentence_end = _ends_sentence(prev_text)
        if cur_text and (speaker_change or silence_break or sentence_end):
            flush()
            cur_speaker, cur_start, cur_text = sp, w.start, []
        cur_text.append(w.text)
        prev_end = w.end
    flush()
    return lines


def _iso_date(yyyymmdd: str) -> str:
    if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return yyyymmdd


def _resolve_speaker_names(
    audio_path: Path,
    turns: list,
    content_dir: Path,
    group_slug: str,
) -> dict[str, str]:
    """Try to map cluster ids (SPEAKER_00, ...) to real member names via
    enrolled voice embeddings. Returns mapping for frontmatter `speakers:`."""
    try:
        import yaml as _yaml
        from pipeline.diarize.enroll import (
            load_group_embeddings, assign_clusters_to_members, member_slug,
        )

        web_root = content_dir.parent
        embeddings_dir = web_root / "voice_embeddings"
        enrolled = load_group_embeddings(group_slug, embeddings_dir)
        if not enrolled:
            return {sp: sp for sp in sorted({t.speaker for t in turns})}

        # Read group YAML for member display names
        group_yaml = web_root / "performers" / f"{group_slug}.yaml"
        member_display: dict[str, str] = {}
        if group_yaml.exists():
            data = _yaml.safe_load(group_yaml.read_text()) or {}
            for m in data.get("members") or []:
                name = m.get("name") or ""
                member_display[member_slug(name)] = name

        return assign_clusters_to_members(
            audio_path, turns, enrolled, member_display
        )
    except Exception as e:
        # Fall back gracefully — never block ingest because of voice match issues.
        import sys
        print(f"  voice-match: {e}; falling back to cluster ids", file=sys.stderr)
        return {sp: sp for sp in sorted({t.speaker for t in turns})}


def _normalize_speaker_names(turns: list, content_dir: Path, group_slug: str) -> list:
    """Map turn speaker labels to the canonical member names from the
    performer registry (handles traditional/simplified glyph drift, etc.).
    Names already matching a member name are left as-is."""
    try:
        import yaml as _yaml
        from difflib import SequenceMatcher

        group_yaml = content_dir.parent / "performers" / f"{group_slug}.yaml"
        if not group_yaml.exists():
            return turns
        data = _yaml.safe_load(group_yaml.read_text()) or {}
        members = [m.get("name") for m in (data.get("members") or []) if m.get("name")]
        if not members:
            return turns
    except Exception:
        return turns

    cache: dict[str, str] = {}
    out = []
    for t in turns:
        sp = getattr(t, "speaker", None)
        if sp is None:
            out.append(t)
            continue
        if sp in members:
            out.append(t)
            continue
        if sp in cache:
            t.speaker = cache[sp]
            out.append(t)
            continue
        best = max(members, key=lambda m: SequenceMatcher(None, sp, m).ratio())
        ratio = SequenceMatcher(None, sp, best).ratio()
        if ratio >= 0.5:
            cache[sp] = best
            t.speaker = best
        else:
            cache[sp] = sp
        out.append(t)
    return out


def _ensure_performer(content_dir: Path, slug: str, language: str) -> None:
    """Auto-stub a performer file if it doesn't exist yet, so build doesn't
    fail on first ingest of a new group. The stub has TODO fields for the
    contributor to fill in via PR."""
    performers_dir = content_dir.parent / "performers"
    performers_dir.mkdir(parents=True, exist_ok=True)
    f = performers_dir / f"{slug}.yaml"
    if f.exists():
        return
    stub = {
        "display_name": f"TODO ({slug})",
        "language": language,
        "region": "TODO",
        "members": [],
        "description": (
            f"Auto-stub created by pipeline. Please fill in display_name, "
            f"region, and members."
        ),
    }
    f.write_text(yaml.safe_dump(stub, allow_unicode=True, sort_keys=False))


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
    asr_backend: str,
    asr_model: str,
    form: Optional[str] = None,
    roles: Optional[dict[str, str]] = None,
) -> Path:
    title = (title_override or fetched.title or "").strip()
    # Clean obvious YouTube-title noise
    import re
    title = re.sub(r"#\S+", "", title).strip()  # strip hashtags
    title = re.sub(r"\s+", " ", title)
    if len(title) > 80:
        title = title[:80].rstrip() + "…"
    group = group_slug or "unknown"
    year = (fetched.upload_date or "")[:4] or datetime.now().strftime("%Y")
    # allow_unicode=True keeps CJK characters in slug (much nicer than romanization)
    slug = slugify(title, allow_unicode=True, max_length=40) or fetched.raw_id
    # Append a short source id so two videos with identical model-generated
    # titles (qwen-omni occasionally hallucinates the same bit name across
    # different videos) don't collide and overwrite each other.
    short_id = (fetched.raw_id or "")[:6]
    fname = f"{group}-{year}-{slug}-{short_id}.md" if short_id else f"{group}-{year}-{slug}.md"
    out_path = out_dir / fname

    _ensure_performer(out_dir, group, language)

    # Normalize qwen-omni speaker names against canonical member list
    # (handles 礼 vs 禮 / drift between traditional & simplified glyphs).
    turns = _normalize_speaker_names(turns, out_dir, group)

    lines = _group_into_lines(words, turns)
    speakers = sorted({sp for sp, _, _ in lines})

    # If turns already have member names (qwen-omni), use as-is. Else try
    # to map via enrolled voice signatures.
    cluster_ids = sorted({t.speaker for t in turns})
    looks_like_real_names = bool(cluster_ids) and not any(
        c.startswith("SPEAKER_") for c in cluster_ids
    )
    if looks_like_real_names:
        speaker_field = {s: s for s in speakers}
        auto_status = "reviewed"
    else:
        speaker_map = _resolve_speaker_names(
            audio_path=fetched.audio_path,
            turns=turns,
            content_dir=out_dir,
            group_slug=group,
        )
        speaker_field = {s: speaker_map.get(s, s) for s in speakers}
        matched_count = sum(
            1 for s, name in speaker_field.items() if name != s
        )
        auto_status = (
            "reviewed"
            if matched_count == len(speaker_field) and matched_count > 0
            else "draft"
        )

    # Fallback: if ASR didn't supply roles/form, derive them from the
    # performer registry (members[].role + default_form). This is what
    # makes downstream text_classify usable for entries ingested by
    # local ASR backends that don't return speaker-role metadata.
    if not roles or not form:
        try:
            import yaml as _yaml
            group_yaml = out_dir.parent / "performers" / f"{group}.yaml"
            if group_yaml.exists():
                gdata = _yaml.safe_load(group_yaml.read_text()) or {}
                if not roles:
                    derived: dict[str, str] = {}
                    for m in gdata.get("members") or []:
                        name = m.get("name")
                        role = m.get("role")
                        if name and role:
                            derived[name] = role
                    if derived:
                        roles = derived
                if not form:
                    form = gdata.get("default_form")
        except Exception as e:
            import sys
            print(f"  registry fallback: {e}", file=sys.stderr)

    fm_form = form or "manzai"
    frontmatter = {
        "title": title,
        "performers": [group],
        "form": fm_form,
        "source": {
            "platform": fetched.platform,
            "url": fetched.source_url,
            "uploader": fetched.uploader,
            "uploaded_at": _iso_date(fetched.upload_date),
            "duration_sec": fetched.duration_sec,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fetched_with": (
                f"yt-dlp/{_yt_dlp_version()}"
                if fetched.platform in ("youtube", "bilibili")
                else "ffmpeg"
            ),
        },
        "language": language,
        "tags": tags,
        "speakers": speaker_field,
        **({"roles": roles} if roles else {}),
        "sensitivity": sensitivity,
        "status": auto_status,
        "contributed_by": DEFAULT_CONTRIBUTOR,
        "ingestion": {
            "pipeline_version": _pipeline_version(),
            "asr": {
                "backend": asr_backend,
                "model": asr_model,
                "detected_language": language,
                "word_count": len(words),
            },
            "diarization": {
                "model": "pyannote/speaker-diarization-3.1",
                "num_speakers": len(speakers),
                "turn_count": len(turns),
            },
        },
    }

    # Translation: if source isn't zh, translate each line to Chinese.
    if not (language or "").lower().startswith("zh"):
        try:
            from pipeline.translate.qwen_text import translate_to_zh

            body_texts = [text for _sp, _t, text in lines]
            zh_lines = translate_to_zh(body_texts, language)
            if zh_lines:
                frontmatter["translations"] = {"zh": zh_lines}
        except Exception as e:
            import sys
            print(f"  translate: {e}; skipping zh translations", file=sys.stderr)

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
