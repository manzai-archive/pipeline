"""Text-only speaker classification: read an entry's lines and ask
qwen-max to assign each one to the right performer based on the role
(ボケ/装傻 vs ツッコミ/吐槽) and dialogue logic.

This is a fallback when voice matching is unreliable (similar voices).
The LLM uses linguistic cues only — self-intros, role-typical phrases,
multi-line consecutive speech — no audio.

Run inside container:
  python -m pipeline.scripts.text_classify <entry.md>
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

import yaml


_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$")
_BATCH = 80


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


def _classify_batch(
    client, model: str, lines: list[str], context: str, names: list[str], previous_tail: list[str]
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    prev_block = ""
    if previous_tail:
        prev_block = (
            "\n\nFor continuity, the last few lines from the previous batch "
            "were attributed as follows (do not output these, just for "
            "context):\n" + "\n".join(previous_tail) + "\n"
        )

    prompt = (
        f"{context}\n\n"
        f"Performers: {' / '.join(names)}.\n\n"
        "For each numbered line below, output the speaker's NAME (one of the "
        "performers above). Use:\n"
        "- self-introductions (\"我是 X\" / 「○○です」 reveal speaker)\n"
        "- role tendencies: ボケ/装傻 proposes absurd setups, "
        "introduces premises, plays the fool; ツッコミ/吐槽 reacts, "
        "corrects (\"なんで？\" / \"是这样吗\" / \"你疯了\"), brings sanity\n"
        "- adjacency: consecutive short responses are often the OTHER speaker; "
        "consecutive long lines are often the SAME speaker\n"
        "- proper-noun mentions: if line N+1 mentions a name from line N, "
        "speaker likely flipped\n\n"
        "Output a single JSON array of strings, length exactly equal to the "
        "input. Each element MUST be one of the performer names. No "
        "commentary, no markdown.\n"
        f"{prev_block}\nLines to classify:\n{numbered}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output ONLY a JSON array of speaker names."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8192,
    )
    raw = (completion.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            raise RuntimeError(f"classifier did not return JSON: {raw[:200]}")
        out = json.loads(m.group(0))
    if not isinstance(out, list):
        raise RuntimeError(f"classifier returned non-list: {type(out)}")
    if len(out) < len(lines):
        out = out + [None] * (len(lines) - len(out))
    elif len(out) > len(lines):
        out = out[: len(lines)]
    # Normalize to one of the known names
    norm: list[str] = []
    for i, x in enumerate(out):
        s = str(x or "").strip()
        if s in names:
            norm.append(s)
        else:
            # Best-effort fuzzy match
            best = max(names, key=lambda n: sum(1 for c in n if c in s))
            norm.append(best)
    return norm


def main(md_path: str) -> None:
    md = Path(md_path).resolve()
    text = md.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("md missing frontmatter")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]

    title = fm.get("title", "")
    perf = (fm.get("performers") or ["?"])[0]
    form = fm.get("form", "manzai")
    language = fm.get("language", "zh")
    roles = fm.get("roles") or {}
    if not roles:
        raise ValueError("entry has no roles; cannot classify")

    role_lines = "\n".join(f"  - {name}: {role}" for name, role in roles.items())
    context = (
        f"Transcript of: {title}\n"
        f"Group: {perf}, Form: {form}, Language: {language}\n"
        f"Roles:\n{role_lines}"
    )
    names = list(roles.keys())

    file_lines = body.splitlines()
    body_idx: list[tuple[int, str]] = []  # (file_line_idx, raw_text_after_prefix)
    prefixes: list[tuple[str, str]] = []  # (timestamp_part_only, text)
    body_texts: list[str] = []
    for i, raw in enumerate(file_lines):
        m = _LINE_RE.match(raw)
        if not m:
            continue
        sp, ts, txt = m.groups()
        body_idx.append((i, raw))
        body_texts.append(txt)
        prefixes.append((ts, txt))

    if not body_texts:
        print("no body lines; nothing to do")
        return

    model = os.environ.get("QWEN_CLASSIFY_MODEL") or "qwen-max"
    print(f"classifying {len(body_texts)} lines with {model} …")

    client = _client()
    all_speakers: list[str] = []
    for i in range(0, len(body_texts), _BATCH):
        batch = body_texts[i : i + _BATCH]
        # Pass last 5 of previous batch as continuity context
        prev_tail = []
        if i > 0:
            tail_start = max(0, i - 5)
            for j in range(tail_start, i):
                prev_tail.append(f"{body_texts[j][:30]} → {all_speakers[j]}")
        out = _classify_batch(client, model, batch, context, names, prev_tail)
        all_speakers.extend(out)
        print(f"  {min(i + _BATCH, len(body_texts))}/{len(body_texts)} done")

    diff = sum(
        1
        for raw, new_sp in zip([raw for _, raw in body_idx], all_speakers)
        if not raw.startswith(f"**{new_sp}**")
    )
    print(f"flipped {diff} / {len(body_texts)} speaker assignments")

    # Rewrite body
    for j, (file_idx, raw) in enumerate(body_idx):
        ts, txt = prefixes[j]
        file_lines[file_idx] = f"**{all_speakers[j]}** [{ts}] {txt}"
    new_body = "\n".join(file_lines)

    # Update frontmatter speakers map
    fm["speakers"] = {s: s for s in sorted(set(all_speakers))}
    fm_yaml = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    md.write_text(f"---\n{fm_yaml}\n---{new_body}")
    print(f"wrote {md}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.scripts.text_classify <entry.md>")
        sys.exit(1)
    main(sys.argv[1])
