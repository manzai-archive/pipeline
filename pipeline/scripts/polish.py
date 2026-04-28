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
_BATCH = 25


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
        "Below are transcript lines from automatic speech recognition.\n\n"
        "VERY STRICT RULES — only fix CLEAR ASR errors:\n"
        "- Homophone mistakes (e.g. 川贝 mis-transcribed as 穿倍, "
        "阿莫西林 → 阿木西林, 谭湘文 → 谭山文).\n"
        "- Missing or extra characters in obvious words.\n"
        "- Wrong word boundaries (e.g. 「下次再来」 → 「下，次再来」).\n"
        "- Add punctuation only if completely missing at sentence end.\n\n"
        "DO NOT, under any circumstance:\n"
        "- Rephrase or rewrite for fluency.\n"
        "- Change meaning, even subtly.\n"
        "- Translate, summarize, expand, or shorten.\n"
        "- Remove or modify interjections / fillers (嗯, 哎呀, ええ, あー).\n"
        "- Change formal/informal register.\n"
        "- Combine or split lines.\n\n"
        "If you are not >95% sure something is an ASR error, leave it as-is. "
        "When in doubt, DO NOTHING.\n\n"
        "Output a single JSON array of strings, length exactly equal to "
        "the input, in the same order. No commentary, no markdown.\n\n"
        "Input:\n" + numbered
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
    roles = fm.get("roles") or {}

    # Pull canonical member names from the performer registry so the LLM
    # can fix homophone misrecognitions of names (徐昊伦 → 徐浩伦, etc.).
    member_names: list[str] = list(roles.keys())
    if not member_names:
        try:
            registry = (
                md.parent.parent.parent / "performers" / f"{perf}.yaml"
            )
            if registry.exists():
                data = yaml.safe_load(registry.read_text()) or {}
                member_names = [
                    m.get("name") for m in (data.get("members") or [])
                    if m.get("name")
                ]
        except Exception:
            pass

    role_block = ""
    if roles:
        role_lines = "\n".join(f"  - {n}: {r}" for n, r in roles.items())
        role_block = f"\nRoles:\n{role_lines}"
    name_block = ""
    if member_names:
        name_block = (
            "\nCanonical performer names (fix homophone ASR errors of these "
            "EXACTLY — e.g. 徐昊伦 → 徐浩伦, 中川礼二 vs 中川禮二): "
            + " / ".join(member_names)
        )

    context = (
        f"Transcript of: {title}\n"
        f"Performers: {perf}\nForm: {form}\nLanguage: {language}"
        f"{role_block}{name_block}"
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
