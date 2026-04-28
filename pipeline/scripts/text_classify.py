"""Text-only speaker classification with optional line splitting.

For each ASR line, the LLM either assigns a single speaker, or — when it
detects a multi-speaker back-and-forth merged into one VAD segment —
splits the line into multiple parts each with its own speaker.

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
_BATCH = 40


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


def _normalize_part(part: dict, names: list[str]) -> dict:
    sp = str(part.get("speaker") or "").strip()
    txt = str(part.get("text") or "").strip()
    if sp not in names and names:
        # Best-effort fuzzy match: pick the name with the most overlapping chars
        sp = max(names, key=lambda n: sum(1 for c in n if c in sp))
    return {"speaker": sp, "text": txt}


def _classify_batch(
    client, model: str, lines: list[str], context: str,
    names: list[str], previous_tail: list[str],
) -> list[list[dict]]:
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
        "For each numbered input line, assign a speaker. Use:\n"
        "- self-introductions (\"我是 X\" / 「○○です」 reveal speaker)\n"
        "- role tendencies: ボケ/装傻 proposes absurd setups, "
        "introduces premises, plays the fool; ツッコミ/吐槽 reacts, "
        "corrects (\"なんで？\" / \"是这样吗\" / \"你疯了\"), brings sanity\n"
        "- adjacency: consecutive short responses are often the OTHER speaker\n"
        "- proper-noun mentions: if line N+1 mentions a name from line N, "
        "speaker likely flipped\n\n"
        "*** LINE SPLITTING IS MANDATORY ***\n"
        "ASR (SenseVoice) often bundles 2–3 speakers' speech into ONE "
        "line. Whenever you see ANY of these patterns inside a single "
        "input line, you MUST split it:\n"
        "  P1. \"你/你是 ... 啊？\" or \"你是 ... 吗？\" or any question "
        "directed at the other speaker → SPLIT before the question (the "
        "question is the OTHER speaker)\n"
        "  P2. A question or accusation immediately followed by a denial "
        "(\"不是 ...\", \"没有 ...\", \"哪里 ...\") in the same line → "
        "SPLIT between them (different speakers)\n"
        "  P3. \"对对对\" / \"嗯嗯嗯\" / \"哎呀\" mid-line answering "
        "something said before → that interjection is often the OTHER "
        "speaker reacting → SPLIT\n"
        "  P4. Vocatives (calling someone by name) followed by a reply\n"
        "  P5. Stage directions / narrator-like asides mixed with "
        "in-character speech\n"
        "Rules when splitting:\n"
        "  1. Each part is an EXACT contiguous substring of the input. "
        "Concatenating all parts of a line MUST reproduce the original "
        "line character-for-character (including punctuation).\n"
        "  2. Do NOT rewrite, paraphrase, translate, add, or remove a "
        "single character — only choose where to cut.\n"
        "  3. You may have 2, 3, 4, or more parts in a single line.\n"
        "  4. Only skip splitting when the line is genuinely one speaker.\n\n"
        "CONCRETE EXAMPLE — this exact line MUST be split:\n"
        '  Input: "对对对啊，我准备改头换面了，你是犯了什么事啊，不是犯了什么事啊，就是准备换一个新发型啊，从头开始。"\n'
        "  Speaker A is buying a haircut, B is the listener.\n"
        "  Correct split (A→B→A):\n"
        '    [{"speaker":"A","text":"对对对啊，我准备改头换面了，"},\n'
        '     {"speaker":"B","text":"你是犯了什么事啊，"},\n'
        '     {"speaker":"A","text":"不是犯了什么事啊，就是准备换一个新发型啊，从头开始。"}]\n'
        "  Why: B asks an accusatory question \"你是犯了什么事\" (P1+P2), "
        "A then explicitly denies \"不是犯了什么事\" (P2). This is THREE "
        "speaker turns, not one.\n\n"
        "*** OUTPUT FORMAT — STRICT ***\n"
        f"Output a JSON array of length EXACTLY {len(lines)} (one element "
        "per input line, in input order). Each element is itself a JSON "
        "array of one-or-more {\"speaker\": NAME, \"text\": SUBSTRING} "
        "objects. The outer array structure MUST be preserved even when "
        "no line is split — wrap each single-speaker line in its own "
        "1-element array.\n\n"
        "WORKED EXAMPLE — given 3 input lines:\n"
        "  1. \"大家好，我是张三\"\n"
        "  2. \"对对对，我准备改头换面了，你是犯了什么事啊，不是犯了什么事啊\"\n"
        "  3. \"是这样吗\"\n"
        "Correct output (assume performers 张三 / 李四):\n"
        "[\n"
        '  [{"speaker":"张三","text":"大家好，我是张三"}],\n'
        '  [{"speaker":"张三","text":"对对对，我准备改头换面了，"},'
        '{"speaker":"李四","text":"你是犯了什么事啊，"},'
        '{"speaker":"张三","text":"不是犯了什么事啊"}],\n'
        '  [{"speaker":"李四","text":"是这样吗"}]\n'
        "]\n\n"
        "WRONG (do NOT do this — flat, missing inner arrays):\n"
        '  [{"speaker":"张三","text":"大家好，我是张三"},'
        '{"speaker":"张三","text":"对对对…"}, …]\n\n'
        f"{prev_block}\nLines to classify:\n{numbered}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output ONLY a JSON array of arrays of {speaker,text} objects."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=12000,
    )
    raw = (completion.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    if os.environ.get("CLASSIFY_DEBUG"):
        print(f"[debug raw first 600]: {raw[:600]}")
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            raise RuntimeError(f"classifier did not return JSON: {raw[:300]}")
        out = json.loads(m.group(0))
    if not isinstance(out, list):
        raise RuntimeError(f"classifier returned non-list: {type(out)}")
    if os.environ.get("CLASSIFY_DEBUG"):
        types = [type(x).__name__ for x in out[:5]]
        print(
            f"[debug] out len={len(out)} expected={len(lines)} "
            f"first5_types={types} finish={completion.choices[0].finish_reason}"
        )

    # Graceful fallback: if model returned a flat list of dicts (one per
    # input line) instead of the nested format, wrap each in a 1-element
    # list so downstream parsing treats them as unsplit lines.
    if (
        len(out) == len(lines)
        and all(isinstance(x, dict) for x in out)
    ):
        out = [[x] for x in out]

    # Pad / truncate to align with input
    if len(out) < len(lines):
        out = out + [None] * (len(lines) - len(out))
    elif len(out) > len(lines):
        out = out[: len(lines)]

    norm_all: list[list[dict]] = []
    for i, item in enumerate(out):
        original = lines[i]
        # Backwards compatibility: accept a bare string as single-speaker line
        if isinstance(item, str):
            norm_all.append([{"speaker": item, "text": original}])
            continue
        if not isinstance(item, list) or not item:
            # Failed — fall back to one part with first known name
            norm_all.append([{"speaker": names[0], "text": original}])
            continue
        parts = [_normalize_part(p, names) for p in item if isinstance(p, dict)]
        # If concatenated text drifted from original (model paraphrased),
        # fall back to single-speaker with original text. Use the first
        # part's speaker as best guess.
        joined = "".join(p["text"] for p in parts)
        if not parts or _strip_punct(joined) != _strip_punct(original):
            sp = parts[0]["speaker"] if parts else names[0]
            norm_all.append([{"speaker": sp, "text": original}])
        else:
            norm_all.append(parts)
    return norm_all


_PUNCT_RE = re.compile(r"[\s，。！？、,.!?…\-]+")


def _strip_punct(s: str) -> str:
    return _PUNCT_RE.sub("", s)


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
    body_idx: list[int] = []  # file_line_idx for each body line
    body_ts: list[str] = []
    body_texts: list[str] = []
    for i, raw in enumerate(file_lines):
        m = _LINE_RE.match(raw)
        if not m:
            continue
        _sp, ts, txt = m.groups()
        body_idx.append(i)
        body_ts.append(ts)
        body_texts.append(txt)

    if not body_texts:
        print("no body lines; nothing to do")
        return

    model = os.environ.get("QWEN_CLASSIFY_MODEL") or "qwen-max"
    print(f"classifying {len(body_texts)} lines with {model} …")

    client = _client()
    per_line: list[list[dict]] = []
    for i in range(0, len(body_texts), _BATCH):
        batch = body_texts[i : i + _BATCH]
        prev_tail: list[str] = []
        if i > 0:
            tail_start = max(0, i - 5)
            for j in range(tail_start, i):
                last_sp = per_line[j][-1]["speaker"] if per_line[j] else "?"
                prev_tail.append(f"{body_texts[j][:30]} → {last_sp}")
        out = _classify_batch(client, model, batch, context, names, prev_tail)
        per_line.extend(out)
        print(f"  {min(i + _BATCH, len(body_texts))}/{len(body_texts)} done")

    splits = sum(1 for parts in per_line if len(parts) > 1)
    total_parts = sum(len(parts) for parts in per_line)
    print(
        f"split {splits} / {len(body_texts)} ASR lines "
        f"into {total_parts} speaker turns"
    )

    # Rewrite body: for each original body line, emit one or more
    # **speaker** [ts] text lines. Non-body lines stay put.
    new_lines: list[str] = []
    body_pos = 0
    for i, raw in enumerate(file_lines):
        if body_pos < len(body_idx) and body_idx[body_pos] == i:
            ts = body_ts[body_pos]
            for k, part in enumerate(per_line[body_pos]):
                if k > 0:
                    new_lines.append("")  # blank between parts of same ASR line
                new_lines.append(f"**{part['speaker']}** [{ts}] {part['text']}")
            body_pos += 1
            continue
        new_lines.append(raw)
    new_body = "\n".join(new_lines)

    all_speakers = sorted({p["speaker"] for parts in per_line for p in parts})
    fm["speakers"] = {s: s for s in all_speakers}
    fm_yaml = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    md.write_text(f"---\n{fm_yaml}\n---{new_body}")
    print(f"wrote {md}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.scripts.text_classify <entry.md>")
        sys.exit(1)
    main(sys.argv[1])
