"""Per-word speaker classification by voiceprint matching.

For each ASR word with start/end timestamps, slice a small audio window
around the word, embed it, and assign it to the reference voiceprint
with highest cosine similarity. This bypasses pyannote's unsupervised
clustering — useful when two performers have similar voices and
clustering tends to merge them.

The references are built per-entry by finding self-introduction lines
(e.g. "大家好，我是徐浩伦") in the ASR output and using those audio
ranges as per-member voice samples. No external enrollment required.
"""
from __future__ import annotations
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline import config
from pipeline.diarize.speakers import Turn


@dataclass
class WordLabel:
    start: float
    end: float
    text: str
    speaker: str


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


def _slice(audio: Path, start: float, end: float, dst: Path) -> None:
    start = max(0.0, start)
    if end <= start:
        end = start + 0.5
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


def _embed(wav_path: Path) -> np.ndarray:
    from pyannote.audio import Inference
    import torch

    model = _load_embedder()
    inference = Inference(model, window="whole")
    if torch.cuda.is_available():
        inference.to(torch.device("cuda"))
    arr = np.asarray(inference(str(wav_path)), dtype=np.float32).reshape(-1)
    return arr / (np.linalg.norm(arr) + 1e-9)


# ---------- self-intro detection ----------

# Capture the name candidate after a self-intro phrase. We extract whatever
# 2-4 CJK/kana characters follow, then fuzzy-match against the canonical
# member names (so ASR errors like 徐浩伦 → 徐昊伦 still match).
_INTRO_RE = re.compile(
    r"(?:我[是叫名]|私は|わたしは|僕は|名前は)\s*"
    r"([぀-ヿ一-鿿]{2,5})"
)
# Also: "X です" / "X だ" pattern and the standalone leading-name pattern.
_DESU_RE = re.compile(r"^([぀-ヿ一-鿿]{2,5})\s*(?:です|だ)\b")


def _match_member(candidate: str, members: list[str], threshold: float = 0.4) -> str | None:
    """Fuzzy-match a captured name candidate to a canonical member name.
    Returns the matched member name, or None if no member exceeds threshold."""
    from difflib import SequenceMatcher

    best_name: str | None = None
    best_ratio = 0.0
    for m in members:
        r = SequenceMatcher(None, candidate, m).ratio()
        if r > best_ratio:
            best_ratio = r
            best_name = m
    if best_ratio >= threshold:
        return best_name
    return None


def find_intro_ranges(
    words, member_names: list[str]
) -> dict[str, list[tuple[float, float]]]:
    """For each member, return EXACT audio time ranges (start, end) of
    segments whose text contains a self-introduction of that member.

    No padding is applied here — duos often say their self-intros back-
    to-back, so padding would cross-contaminate references. The caller
    is responsible for clamping any extension against neighbouring
    intros."""
    found: dict[str, list[tuple[float, float]]] = {m: [] for m in member_names}
    for w in words:
        text = getattr(w, "text", "") or ""
        candidate = None
        m = _INTRO_RE.search(text)
        if m:
            candidate = m.group(1)
        else:
            m2 = _DESU_RE.match(text.strip())
            if m2:
                candidate = m2.group(1)
        if not candidate:
            continue
        member = _match_member(candidate, member_names)
        if member is None:
            continue
        start = float(getattr(w, "start", 0.0))
        end = float(getattr(w, "end", start + 1.5))
        found[member].append((max(0.0, start), end))
    return found


def _all_intros(ranges: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for rs in ranges.values():
        out.extend(rs)
    return sorted(out)


def _safe_extend(
    s: float, e: float, all_intros: list[tuple[float, float]],
    desired_min: float = 1.5,
) -> tuple[float, float]:
    """Extend [s, e] to at least desired_min seconds long, but never cross
    into a neighbouring self-intro segment owned by anyone else."""
    if e - s >= desired_min:
        return s, e
    # Find the previous and next OTHER intro boundaries
    prev_end = 0.0
    next_start = float("inf")
    for ps, pe in all_intros:
        if pe <= s and pe > prev_end:
            prev_end = pe
        if ps >= e and ps < next_start:
            next_start = ps
    # First try forward extension
    need = desired_min - (e - s)
    forward = min(need, max(0.0, next_start - e))
    e += forward
    need -= forward
    # Then backward extension
    if need > 0:
        backward = min(need, max(0.0, s - prev_end))
        s -= backward
    return max(0.0, s), e


def build_refs(
    audio: Path,
    words,
    member_names: list[str],
    min_clip_sec: float = 2.5,
    max_clips_per_member: int = 4,
) -> dict[str, np.ndarray]:
    """Build per-member reference voiceprint from self-intro audio ranges.
    Returns {member_name: embedding}; missing members are absent from dict.

    `words` is a list of Word/WordLabel-like objects with .start/.end/.text.

    Fallback: if exactly one member's intro is missing for a 2-member group,
    pick the longest non-overlapping early segment as the missing member's
    reference (since duos almost always alternate self-intros at the start).
    """
    ranges = find_intro_ranges(words, member_names)
    all_intros = _all_intros(ranges)
    refs: dict[str, np.ndarray] = {}
    work = Path(tempfile.mkdtemp(prefix="vw-ref-"))
    try:
        for m, rs in ranges.items():
            embs: list[np.ndarray] = []
            for i, (s, e) in enumerate(rs[:max_clips_per_member]):
                # Extend to min_clip_sec, but never cross into another
                # member's intro range.
                s2, e2 = _safe_extend(s, e, all_intros, min_clip_sec)
                clip = work / f"ref_{i}_{m[:8]}.wav"
                try:
                    _slice(audio, s2, e2, clip)
                    embs.append(_embed(clip))
                except Exception as exc:
                    print(f"  vw: ref clip {m} @{s2:.1f}: {exc}")
            if embs:
                v = np.mean(np.stack(embs, axis=0), axis=0)
                refs[m] = v / (np.linalg.norm(v) + 1e-9)
                print(
                    f"  vw: ref for {m}: {len(embs)} clip(s) "
                    f"({[f'{s:.1f}-{e:.1f}' for s, e in rs[:max_clips_per_member]]})"
                )

        # Two-member fallback: if exactly one member is missing, find an
        # early segment that doesn't overlap any matched intro and use it
        # as the missing member's reference.
        missing = [m for m in member_names if m not in refs]
        if len(member_names) == 2 and len(missing) == 1 and refs:
            other_name = next(iter(refs))
            other_ranges = ranges[other_name]
            other_intervals = [(s, e) for s, e in other_ranges]

            def _overlaps(s: float, e: float) -> bool:
                for ls, le in other_intervals:
                    if not (e <= ls or s >= le):
                        return True
                return False

            candidate: tuple[float, float] | None = None
            for w in words:
                start = float(getattr(w, "start", 0.0))
                end = float(getattr(w, "end", start))
                if start > 60.0:
                    break  # only look at first minute for self-intro fallback
                if end - start < min_clip_sec:
                    continue
                if _overlaps(start, end):
                    continue
                candidate = (start, end)
                break
            if candidate:
                clip = work / f"ref_fb_{missing[0][:8]}.wav"
                try:
                    s, e = candidate
                    _slice(audio, max(0.0, s - 0.3), e + 0.5, clip)
                    refs[missing[0]] = _embed(clip)
                    print(
                        f"  vw: ref for {missing[0]} (fallback @{s:.1f}-{e:.1f}): "
                        f"1 clip(s)"
                    )
                except Exception as exc:
                    print(f"  vw: fallback ref {missing[0]}: {exc}")
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return refs


# ---------- per-word classification ----------


def classify_words(
    audio: Path,
    words,
    refs: dict[str, np.ndarray],
    window_sec: float = 3.0,
    min_clip_sec: float = 1.5,
    anchors: dict[str, list[tuple[float, float]]] | None = None,
) -> list[str]:
    """For each ASR segment (each `word` is sentence-level for SenseVoice),
    slice the audio at the segment's actual [start, end], extending to at
    least `min_clip_sec` symmetrically if shorter. Embed the slice and
    assign to the nearest reference voiceprint by cosine similarity.

    Using the segment's natural boundaries (rather than a fixed window
    around the midpoint) gives the embedder the cleanest single-speaker
    audio available — short segments get a small symmetric pad."""
    if not refs:
        return [""] * len(words)
    members = list(refs.keys())
    out: list[str] = []
    last = members[0]
    work = Path(tempfile.mkdtemp(prefix="vw-word-"))
    try:
        for i, w in enumerate(words):
            s_orig = float(getattr(w, "start", 0.0))
            e_orig = float(getattr(w, "end", s_orig))
            # Anchor short-circuit: if this segment overlaps a known
            # self-intro range, force its label to that member.
            if anchors:
                wmid = (s_orig + e_orig) / 2
                forced: str | None = None
                for member, rs in anchors.items():
                    for as_, ae in rs:
                        if as_ <= wmid <= ae:
                            forced = member
                            break
                    if forced:
                        break
                if forced is not None:
                    out.append(forced)
                    last = forced
                    continue
            length = e_orig - s_orig
            if length < min_clip_sec:
                # Pad symmetrically up to min_clip_sec
                pad = (min_clip_sec - length) / 2
                s = max(0.0, s_orig - pad)
                e = e_orig + pad
            else:
                # If naturally longer than window_sec, take the central
                # window_sec to avoid speaker bleed at boundaries.
                if length > window_sec:
                    mid = (s_orig + e_orig) / 2
                    s = max(0.0, mid - window_sec / 2)
                    e = mid + window_sec / 2
                else:
                    s, e = s_orig, e_orig
            clip = work / f"w_{i}.wav"
            try:
                _slice(audio, s, e, clip)
                emb = _embed(clip)
                best_sim = -2.0
                best = last
                for m, ref in refs.items():
                    sim = float(np.dot(emb, ref))
                    if sim > best_sim:
                        best_sim, best = sim, m
                last = best
                out.append(best)
            except Exception:
                out.append(last)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return out


def smooth_labels(labels: list[str], window: int = 3) -> list[str]:
    """Majority-vote smoothing over a sliding window of `window` labels.
    Removes single-word speaker flips that come from noisy short embeddings."""
    if window <= 1 or not labels:
        return labels
    half = window // 2
    out = list(labels)
    n = len(labels)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        votes: dict[str, int] = {}
        for j in range(lo, hi):
            votes[labels[j]] = votes.get(labels[j], 0) + 1
        out[i] = max(votes.items(), key=lambda kv: kv[1])[0]
    return out


def words_to_turns(words, labels: list[str]) -> list[Turn]:
    """Group consecutive same-speaker words into Turn segments."""
    turns: list[Turn] = []
    if not words or not labels:
        return turns
    cur_sp = labels[0]
    cur_start = float(getattr(words[0], "start", 0.0))
    cur_end = float(getattr(words[0], "end", cur_start))
    for w, sp in zip(words[1:], labels[1:]):
        if sp != cur_sp:
            turns.append(Turn(start=cur_start, end=cur_end, speaker=cur_sp))
            cur_sp = sp
            cur_start = float(getattr(w, "start", cur_end))
        cur_end = float(getattr(w, "end", cur_end))
    turns.append(Turn(start=cur_start, end=cur_end, speaker=cur_sp))
    return turns


# ---------- entry point ----------


def _embed_segments(audio: Path, words, min_clip_sec: float = 1.5,
                    max_clip_sec: float = 6.0) -> list[np.ndarray | None]:
    """Embed every ASR segment using its natural [start, end] (extended
    or trimmed to fall inside [min_clip_sec, max_clip_sec])."""
    out: list[np.ndarray | None] = []
    work = Path(tempfile.mkdtemp(prefix="vw-seg-"))
    try:
        for i, w in enumerate(words):
            s = float(getattr(w, "start", 0.0))
            e = float(getattr(w, "end", s))
            length = e - s
            if length < min_clip_sec:
                pad = (min_clip_sec - length) / 2
                s = max(0.0, s - pad)
                e = e + pad
            elif length > max_clip_sec:
                mid = (s + e) / 2
                s = max(0.0, mid - max_clip_sec / 2)
                e = mid + max_clip_sec / 2
            clip = work / f"s_{i}.wav"
            try:
                _slice(audio, s, e, clip)
                out.append(_embed(clip))
            except Exception:
                out.append(None)
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return out


def diarize_by_voiceprint(
    audio: Path,
    words,
    member_names: list[str],
    smoothing_window: int = 3,
    embed_window_sec: float = 3.0,  # kept for API compat; unused in new path
) -> tuple[list[Turn], list[str]]:
    """End-to-end pipeline:
      1. Embed every ASR segment (using its natural span).
      2. KMeans-cluster the segment embeddings into N (= num members)
         clusters. KMeans is initialized from self-intro embeddings
         when available, falling back to k-means++ otherwise.
      3. Map each cluster to a member name by finding which cluster the
         self-intro segment of that member falls into. If both intros
         land in the same cluster (similar voices), use the second-
         largest cluster for the missing member as a fallback.

    Returns (turns, per_word_labels). Returns ([], []) if embedding fails
    too often."""
    if not words:
        return [], []

    print(f"  vw: embedding {len(words)} segments …")
    seg_embs = _embed_segments(audio, words)
    valid_idx = [i for i, e in enumerate(seg_embs) if e is not None]
    if len(valid_idx) < len(words) * 0.5:
        print(f"  vw: only {len(valid_idx)} / {len(words)} embeddings ok; skipping")
        return [], []

    # Find self-intro anchors (member -> [(start, end), ...])
    anchors = find_intro_ranges(words, member_names)
    print(f"  vw: anchors={ {m: rs for m, rs in anchors.items() if rs} }")

    # Build initial centroids from self-intro segment embeddings (one per
    # member). For members without a self-intro, omit and let kmeans++ do
    # its thing.
    init_centroids: list[np.ndarray] = []
    init_members: list[str] = []
    for m in member_names:
        for i, w in enumerate(words):
            wmid = (float(getattr(w, "start", 0.0)) + float(getattr(w, "end", 0.0))) / 2
            if any(s <= wmid <= e for s, e in anchors.get(m, [])):
                if seg_embs[i] is not None:
                    init_centroids.append(seg_embs[i])
                    init_members.append(m)
                    break

    # Prepare the matrix
    X = np.stack([seg_embs[i] for i in valid_idx], axis=0)
    n_clusters = len(member_names)
    init = "k-means++"
    if len(init_centroids) == n_clusters:
        init = np.stack(init_centroids, axis=0)

    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        print(f"  vw: sklearn missing ({e}); skipping")
        return [], []

    km = KMeans(n_clusters=n_clusters, init=init, n_init=10, random_state=0)
    cluster_ids = km.fit_predict(X)

    # Map cluster id -> member by which member's self-intro segment falls
    # into that cluster. Build cluster_to_member.
    cluster_to_member: dict[int, str] = {}
    for m in member_names:
        for i, w in enumerate(words):
            if i not in valid_idx:
                continue
            wmid = (float(getattr(w, "start", 0.0)) + float(getattr(w, "end", 0.0))) / 2
            if any(s <= wmid <= e for s, e in anchors.get(m, [])):
                pos = valid_idx.index(i)
                cluster_to_member[int(cluster_ids[pos])] = m
                break

    # If both intros mapped to the same cluster (similar voices fooled
    # KMeans), assign the unused cluster id to the missing member.
    used_clusters = set(cluster_to_member.keys())
    used_members = set(cluster_to_member.values())
    if len(used_clusters) < n_clusters:
        for c in range(n_clusters):
            if c not in used_clusters:
                missing = [m for m in member_names if m not in used_members]
                if missing:
                    cluster_to_member[c] = missing[0]
                    used_members.add(missing[0])
    # If still missing (no self-intro found), fall back: assign by cluster size
    for c in range(n_clusters):
        if c not in cluster_to_member:
            remaining = [m for m in member_names if m not in used_members]
            if remaining:
                cluster_to_member[c] = remaining[0]
                used_members.add(remaining[0])

    print(f"  vw: cluster→member: {cluster_to_member}")

    # Project labels back into the full words order; for words whose
    # embedding failed, copy the previous label.
    raw: list[str] = []
    last = member_names[0]
    j = 0
    for i in range(len(words)):
        if i in valid_idx and j < len(cluster_ids):
            label = cluster_to_member.get(int(cluster_ids[j]), last)
            j += 1
        else:
            label = last
        raw.append(label)
        last = label

    # Light smoothing
    smooth = smooth_labels(raw, window=smoothing_window)
    # Re-apply anchors (self-intro segments must be labelled correctly)
    for i, w in enumerate(words):
        wmid = (float(getattr(w, "start", 0.0)) + float(getattr(w, "end", 0.0))) / 2
        for m, rs in anchors.items():
            if any(s <= wmid <= e for s, e in rs):
                smooth[i] = m
                break

    turns = words_to_turns(words, smooth)
    print(f"  vw: {len(turns)} turns")
    return turns, smooth
