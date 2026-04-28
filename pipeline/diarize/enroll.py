"""Speaker enrollment: pre-compute voice embeddings for individual
performers (members of a group) so diarization clusters can be auto-mapped
to real names instead of SPEAKER_00 / SPEAKER_01.

Schema add to performer (group) YAML at web/src/content/performers/<group>.yaml:

  voice_samples:
    - member: 中川剛               # must match a name in `members[].name`
      source: youtube:bW9QvvXUaFI  # or path:/abs/path/to/file.wav
      start: 30
      end:   60

Run enrollment:
  python -m pipeline enroll-group <group-slug>

Writes:
  web/src/content/voice_embeddings/<group-slug>__<member-slug>.npy
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from slugify import slugify

from pipeline import config


_pipeline_emb = None


def _load_embedding_model():
    global _pipeline_emb
    if _pipeline_emb is None:
        from pyannote.audio import Model
        import torch

        _pipeline_emb = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=config.HF_TOKEN,
        )
        if torch.cuda.is_available():
            _pipeline_emb.to(torch.device("cuda"))
    return _pipeline_emb


def _proxy_args() -> list[str]:
    p = (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or ""
    )
    return ["--proxy", p] if p else []


def _yt_audio(yt_id: str, dst_stem: Path) -> Path:
    args = ["yt-dlp", *_proxy_args()]
    args += [
        "-x", "--audio-format", "wav", "--no-playlist",
        "-o", str(dst_stem.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={yt_id}",
    ]
    subprocess.run(args, check=True)
    return dst_stem.with_suffix(".wav")


def _slice_wav(src: Path, start: float, end: float, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start}", "-to", f"{end}",
            "-i", str(src),
            "-ac", "1", "-ar", "16000",
            "-loglevel", "error",
            str(dst),
        ],
        check=True,
    )


def _embed(audio_path: Path) -> np.ndarray:
    """Mean ECAPA embedding for the entire wav, normalized."""
    from pyannote.audio import Inference
    import torch

    model = _load_embedding_model()
    inference = Inference(model, window="whole")
    if torch.cuda.is_available():
        inference.to(torch.device("cuda"))
    emb = inference(str(audio_path))
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-9)


def member_slug(member_name: str) -> str:
    """Slugify a member name for use in filenames."""
    s = slugify(member_name, allow_unicode=True, max_length=40)
    return s or member_name


def enroll_group(group_slug: str, performers_dir: Path, embeddings_dir: Path) -> dict[str, Path]:
    """Enroll every member of a group that has voice_samples in their YAML.
    Returns {member_slug: written .npy path}."""
    yaml_path = performers_dir / f"{group_slug}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)
    data = yaml.safe_load(yaml_path.read_text()) or {}
    samples = data.get("voice_samples") or []
    if not samples:
        raise ValueError(f"{group_slug}: no voice_samples in {yaml_path.name}")

    # Group samples by member name
    by_member: dict[str, list[dict]] = {}
    for s in samples:
        by_member.setdefault(s["member"], []).append(s)

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"enroll-{group_slug}-"))
    written: dict[str, Path] = {}
    try:
        for member_name, member_samples in by_member.items():
            mslug = member_slug(member_name)
            embs: list[np.ndarray] = []
            for i, s in enumerate(member_samples):
                src_spec = s["source"]
                start = float(s.get("start", 0))
                end = float(s.get("end", start + 10))
                if src_spec.startswith("youtube:"):
                    yt_id = src_spec.split(":", 1)[1]
                    full = _yt_audio(yt_id, work / f"{yt_id}_{mslug}_{i}")
                elif src_spec.startswith("path:"):
                    full = Path(src_spec[5:]).expanduser().resolve()
                else:
                    raise ValueError(f"unknown source spec: {src_spec}")
                slice_path = work / f"{mslug}_{i}_slice.wav"
                _slice_wav(full, start, end, slice_path)
                embs.append(_embed(slice_path))
            mean = np.mean(np.stack(embs, axis=0), axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            out = embeddings_dir / f"{group_slug}__{mslug}.npy"
            np.save(out, mean.astype(np.float32))
            written[mslug] = out
        return written
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def load_group_embeddings(group_slug: str, embeddings_dir: Path) -> dict[str, np.ndarray]:
    """Return {member_name: embedding} for the given group, or empty dict if
    no enrollments exist yet."""
    out: dict[str, np.ndarray] = {}
    if not embeddings_dir.exists():
        return out
    for f in embeddings_dir.glob(f"{group_slug}__*.npy"):
        member_slug_part = f.stem.split("__", 1)[1]
        try:
            v = np.load(f).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            out[member_slug_part] = v
        except Exception:
            continue
    return out


def cluster_embedding(audio_path: Path, turns: list, speaker_id: str) -> Optional[np.ndarray]:
    """Mean embedding of all segments where this cluster speaks."""
    segs = [t for t in turns if t.speaker == speaker_id and t.end - t.start >= 1.0]
    if not segs:
        return None
    work = Path(tempfile.mkdtemp(prefix=f"cluster-{speaker_id}-"))
    embs = []
    try:
        for i, s in enumerate(segs[:8]):
            slice_path = work / f"{i}.wav"
            _slice_wav(audio_path, s.start, s.end, slice_path)
            embs.append(_embed(slice_path))
        if not embs:
            return None
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        return mean / (np.linalg.norm(mean) + 1e-9)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def assign_clusters_to_members(
    audio_path: Path,
    turns: list,
    enrolled: dict[str, np.ndarray],   # {member_slug: emb}
    member_display: dict[str, str],     # {member_slug: display_name}
    threshold: float = 0.45,
) -> dict[str, str]:
    """For each cluster id in turns, return the matched member display name,
    or the original cluster id if no good match."""
    cluster_ids = sorted({t.speaker for t in turns})
    if not enrolled:
        return {c: c for c in cluster_ids}

    cluster_embs: dict[str, np.ndarray] = {}
    for c in cluster_ids:
        e = cluster_embedding(audio_path, turns, c)
        if e is not None:
            cluster_embs[c] = e

    mapping: dict[str, str] = {}
    used: set[str] = set()
    # Sort clusters by best-match similarity, assign greedily
    candidates: list[tuple[float, str, str]] = []
    for c, ce in cluster_embs.items():
        for mslug, me in enrolled.items():
            sim = float(np.dot(ce, me))
            candidates.append((sim, c, mslug))
    candidates.sort(reverse=True)
    for sim, c, mslug in candidates:
        if c in mapping or mslug in used:
            continue
        if sim >= threshold:
            mapping[c] = member_display.get(mslug, mslug)
            used.add(mslug)

    # Inference: if all members but one are matched, last cluster = unmatched member
    unmatched_clusters = [c for c in cluster_ids if c not in mapping]
    unmatched_members = [
        m for m in enrolled if m not in used
    ]
    if len(unmatched_clusters) == 1 and len(unmatched_members) == 1:
        m = unmatched_members[0]
        mapping[unmatched_clusters[0]] = member_display.get(m, m)

    for c in cluster_ids:
        mapping.setdefault(c, c)
    return mapping
