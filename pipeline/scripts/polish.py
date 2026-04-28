"""LLM polish: send entry transcript to qwen-max for ASR error / typo
correction. Preserves speaker tags, timestamps, line count.

Run:
  python -m pipeline.scripts.polish <entry.md>
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

import yaml


_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$")
_BATCH = 50


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


def _polish_batch(client, model: str, lines: list[str], context: str) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    prompt = (
        f"{context}\n\n"
        "Below are transcript lines from automatic speech recognition. "
        "Some lines may have ASR errors (homophones, missing characters, "
        "wrong word boundaries). Review each line and output a corrected "
        "version in the SAME ORDER, ONE per input line.\n\n"
        "Rules:\n"
        "- Fix obvious ASR errors only; do NOT change meaning, do NOT translate, "
        "do NOT add explanation, do NOT remove interjections / fillers.\n"
        "- Preserve filler words (嗯, 哎呀, ええ, あー).\n"
        "- Preserve punctuation style; add punctuation only if missing.\n"
        "- If a line is already correct, output it as-is.\n\n"
        "Output a single JSON array of strings, length exactly equal to "
        "the input. No commentary, no markdown.\n\nInput:\n" + numbered
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output ONLY a JSON array of strings."},
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
            raise RuntimeError(f"polish did not return JSON: {raw[:200]}")
        out = json.loads(m.group(0))
    if not isinstance(out, list) or len(out) != len(lines):
        # Pad/truncate to align
        if len(out) < len(lines):
            out = out + [None] * (len(lines) - len(out))
        else:
            out = out[: len(lines)]
    # Replace None with original
    return [str(o).strip() if o else lines[i] for i, o in enumerate(out)]


def main(md_path: str) -> None:
    md = Path(md_path).resolve()
    text = md.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("md missing frontmatter")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]

    title = fm.get("title", "")
    perf = fm.get("performers", ["?"])[0]
    form = fm.get("form", "manzai")
    language = fm.get("language", "zh")
    context = (
        f"Transcript of: {title}\n"
        f"Performers: {perf}\nForm: {form}\nLanguage: {language}"
    )

    body_lines: list[tuple[int, str, str]] = []  # (file_line_idx, text, prefix)
    file_lines = body.splitlines()
    for i, raw in enumerate(file_lines):
        m = _LINE_RE.match(raw)
        if not m:
            continue
        sp, ts, txt = m.groups()
        body_lines.append((i, txt, f"**{sp}** [{ts}] "))

    if not body_lines:
        print("no body lines; nothing to do")
        return

    model = os.environ.get("QWEN_POLISH_MODEL") or "qwen-max"
    print(f"polishing {len(body_lines)} lines with {model} …")

    client = _client()
    texts = [b[1] for b in body_lines]
    polished: list[str] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        out = _polish_batch(client, model, batch, context)
        polished.extend(out)
        print(f"  {min(i + _BATCH, len(texts))}/{len(texts)} done")

    diff = sum(1 for a, b in zip(texts, polished) if a != b)
    print(f"changed {diff} / {len(texts)} lines")

    # Write back
    for j, (file_idx, _orig, prefix) in enumerate(body_lines):
        file_lines[file_idx] = prefix + polished[j]
    new_body = "\n".join(file_lines)
    md.write_text(f"---\n{parts[1].strip()}\n---{new_body}")
    print(f"wrote {md}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.scripts.polish <entry.md>")
        sys.exit(1)
    main(sys.argv[1])
