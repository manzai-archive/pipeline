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


def _resolve_dialogue_path(md: Path) -> Path:
    """Sibling structured-dialogue file: web/src/content/dialogues/<slug>.yaml.
    md is web/src/content/manzai/<slug>.md."""
    return md.parent.parent / "dialogues" / f"{md.stem}.yaml"


def main(md_path: str) -> None:
    md = Path(md_path).resolve()
    text = md.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("md missing frontmatter")
    fm = yaml.safe_load(parts[1]) or {}

    title = fm.get("title", "")
    perf = fm.get("performers", ["?"])[0]
    form = fm.get("form", "manzai")
    language = fm.get("language", "zh")
    roles = fm.get("roles") or {}

    member_names: list[str] = list(roles.keys())
    if not member_names:
        try:
            registry = (
                md.parent.parent / "performers" / f"{perf}.yaml"
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

    # Prefer the structured dialogue file; fall back to in-body parsing
    # for legacy entries.
    dialogue_path = _resolve_dialogue_path(md)
    dialogue: dict | None = None
    body_fallback = False
    texts: list[str]
    if dialogue_path.exists():
        dialogue = yaml.safe_load(dialogue_path.read_text()) or {}
        utterances = dialogue.get("utterances") or []
        texts = [str(u.get("text") or "") for u in utterances]
    else:
        body_fallback = True
        body = parts[2]
        body_lines: list[tuple[int, str, str]] = []
        file_lines = body.splitlines()
        for i, raw in enumerate(file_lines):
            m = _LINE_RE.match(raw)
            if not m:
                continue
            sp, ts, txt = m.groups()
            body_lines.append((i, txt, f"**{sp}** [{ts}] "))
        texts = [b[1] for b in body_lines]

    if not texts:
        print("no utterances; nothing to do")
        return

    model = os.environ.get("QWEN_POLISH_MODEL") or "qwen-max"
    print(f"polishing {len(texts)} lines with {model} …")

    client = _client()
    polished: list[str] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        out = _polish_batch(client, model, batch, context)
        polished.extend(out)
        print(f"  {min(i + _BATCH, len(texts))}/{len(texts)} done")

    diff = sum(1 for a, b in zip(texts, polished) if a != b)
    print(f"changed {diff} / {len(texts)} lines")

    if not body_fallback and dialogue is not None:
        utterances = dialogue.get("utterances") or []
        for u, new_text in zip(utterances, polished):
            u["text"] = new_text
        dialogue["utterances"] = utterances
        dialogue_path.write_text(
            yaml.safe_dump(dialogue, allow_unicode=True, sort_keys=False)
        )
        print(f"wrote {dialogue_path}")
    else:
        # Legacy path: rewrite md body in place
        body = parts[2]
        file_lines = body.splitlines()
        # Rebuild body_lines for the rewrite step
        idx = 0
        for i, raw in enumerate(file_lines):
            m = _LINE_RE.match(raw)
            if not m:
                continue
            sp, ts, _ = m.groups()
            file_lines[i] = f"**{sp}** [{ts}] {polished[idx]}"
            idx += 1
        new_body = "\n".join(file_lines)
        md.write_text(f"---\n{parts[1].strip()}\n---{new_body}")
        print(f"wrote {md} (legacy in-body path)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.scripts.polish <entry.md>")
        sys.exit(1)
    main(sys.argv[1])
