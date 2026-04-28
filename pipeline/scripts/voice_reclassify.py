"""Re-classify speakers in an existing entry by voice-matching each line's
audio slice against reference voiceprints derived from the entry's own
self-introduction lines.

Bootstrap: for each performer named in `roles`, find the first segment
they speak (typically the self-intro). Use that audio slice as their
reference voiceprint. Then for every segment, embed its audio slice and
assign to the nearest reference by cosine similarity.

Run inside container:
  python -m pipeline.scripts.voice_reclassify <entry.md>

Audio is re-fetched from the source URL if not cached in work/.
"""
from __future__ import annotations
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from pipeline import config
from pipeline.sources import fetch as fetch_source


_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)$")


def _parse_md(path: Path):
    text = path.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("md missing frontmatter")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    lines = []  # (speaker, start_sec, text, raw_line)
    for raw in body.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        sp, hh, mm, ss, txt = m.groups()
        start = int(hh) * 3600 + int(mm) * 60 + int(ss)
        lines.append((sp, start, txt, raw))
    return fm, body, lines


def _slice(audio: Path, start: float, end: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start}", "-to", f"{end}",
            "-i", str(audio),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            str(dst),
        ],
        check=True,
    )


_emb_model = None


def _load_embedder():
    global _emb_model
    if _emb_model is None:
        from pyannote.audio import Model
        import torch

        _emb_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=config.HF_TOKEN,
        )
        if torch.cuda.is_available():
            _emb_model.to(torch.device("cuda"))
    return _emb_model


def _embed(wav: Path) -> np.ndarray:
    from pyannote.audio import Inference
    import torch

    model = _load_embedder()
    inference = Inference(model, window="whole")
    if torch.cuda.is_available():
        inference.to(torch.device("cuda"))
    arr = np.asarray(inference(str(wav)), dtype=np.float32).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-9)


def main(md_path: str) -> None:
    md = Path(md_path).resolve()
    fm, body, lines = _parse_md(md)
    if not lines:
        print("no lines in body; nothing to do")
        return

    # Source audio
    url = fm.get("source", {}).get("url")
    if not url:
        raise ValueError("entry has no source.url")
    work = config.WORK_DIR
    work.mkdir(exist_ok=True, parents=True)
    fetched = fetch_source(url, work)
    audio_path = fetched.audio_path
    print(f"audio: {audio_path}")

    # Performer names from the group YAML (canonical)
    web_root = config.WORKSPACE_ROOT / "web" / "src" / "content"
    perf_slug = (fm.get("performers") or [None])[0]
    members: list[str] = []
    if perf_slug:
        py = web_root / "performers" / f"{perf_slug}.yaml"
        if py.exists():
            data = yaml.safe_load(py.read_text()) or {}
            members = [m.get("name") for m in (data.get("members") or []) if m.get("name")]
    if len(members) < 2:
        print("performer group has fewer than 2 members; aborting")
        return
    print(f"members: {members}")

    # Self-intro anchors: line whose text contains "我是<name>" or
    # "私は<name>" or just <name> at the start. Trust the WORDS of the
    # self-intro to identify the speaker, regardless of qwen's tag.
    intro_pat = {
        m: re.compile(rf"(?:我是|私は|わたしは|僕は|名前は)\s*{re.escape(m)}|^{re.escape(m)}です")
        for m in members
    }
    intro_segments: dict[str, list] = {m: [] for m in members}
    for sp, start, txt, raw in lines:
        for m, pat in intro_pat.items():
            if pat.search(txt):
                intro_segments[m].append((start, txt))
                break

    refs: dict[str, np.ndarray] = {}
    tmp = Path(tempfile.mkdtemp(prefix="vr-ref-"))
    try:
        for m in members:
            anchors = intro_segments[m]
            if not anchors:
                # Fallback: trust qwen's first tag for this member
                fallback = [(s, t) for sp, s, t, _ in lines if sp == m][:2]
                anchors = fallback
                print(f"  no self-intro found for {m}; using qwen's first 2 lines as fallback")
            picked = []
            for start, txt in anchors[:3]:
                end = start + 4.0
                clip = tmp / f"ref_{m}_{start}.wav"
                try:
                    _slice(audio_path, start, end, clip)
                    picked.append(_embed(clip))
                except Exception as e:
                    print(f"    skip ref clip @{start}: {e}")
            if not picked:
                print(f"  ERROR: no usable reference for {m}; aborting")
                return
            ref = np.mean(np.stack(picked, axis=0), axis=0)
            refs[m] = ref / (np.linalg.norm(ref) + 1e-9)
            print(f"  ref voiceprint for {m}: {len(picked)} clips ({[round(s,1) for s, _ in anchors[:3]]})")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    # Now re-classify every line.
    print(f"re-classifying {len(lines)} lines …")
    new_speakers: list[str] = []
    flips = 0
    tmp = Path(tempfile.mkdtemp(prefix="vr-line-"))
    try:
        # Sort lines by start time and use end = next start (or +6s for last).
        sorted_lines = sorted(lines, key=lambda x: x[1])
        for i, (cur_sp, start, _txt, _raw) in enumerate(sorted_lines):
            end = sorted_lines[i + 1][1] if i + 1 < len(sorted_lines) else start + 6.0
            if end - start < 0.5:
                end = start + 0.5
            if end - start > 8.0:
                end = start + 8.0
            clip = tmp / f"line_{i}.wav"
            try:
                _slice(audio_path, start, end, clip)
                emb = _embed(clip)
            except Exception as e:
                print(f"  line {i} embed failed: {e}; keep {cur_sp}")
                new_speakers.append(cur_sp)
                continue
            best_sp, best_sim = cur_sp, -2.0
            for sp, ref in refs.items():
                sim = float(np.dot(emb, ref))
                if sim > best_sim:
                    best_sim, best_sp = sim, sp
            if best_sp != cur_sp:
                flips += 1
            new_speakers.append(best_sp)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"  flipped {flips} / {len(lines)} speaker assignments")

    # Rewrite body
    new_body_lines: list[str] = []
    line_idx_by_start = {start: i for i, (_, start, _, _) in enumerate(sorted(lines, key=lambda x: x[1]))}
    for raw in body.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            new_body_lines.append(raw)
            continue
        old_sp, hh, mm, ss, txt = m.groups()
        start = int(hh) * 3600 + int(mm) * 60 + int(ss)
        new_sp = new_speakers[line_idx_by_start[start]]
        new_body_lines.append(f"**{new_sp}** [{hh}:{mm}:{ss}] {txt}")
    new_body = "\n".join(new_body_lines)

    # Update frontmatter speakers map
    new_speaker_set = sorted(set(new_speakers))
    fm["speakers"] = {s: s for s in new_speaker_set}

    fm_yaml = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    md.write_text(f"---\n{fm_yaml}\n---{new_body}")
    print(f"wrote {md}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.scripts.voice_reclassify <entry.md>")
        sys.exit(1)
    main(sys.argv[1])
