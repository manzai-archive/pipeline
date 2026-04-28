"""Qwen text-model translation: ja/en/etc → zh, line-by-line.

Used after ASR to populate frontmatter `translations.zh`. The site has
a language toggle that flips between original and translated text per
entry. Translation is line-aligned with the script body so 1:1 mapping
is preserved.

Default model: `qwen-plus` (good quality, cheap). Override via
QWEN_TRANSLATE_MODEL env.
"""
from __future__ import annotations
import json
import os
import re
from typing import Optional


_BATCH_SIZE = 60
_LANG_NAME = {
    "ja": "Japanese",
    "en": "English",
    "ko": "Korean",
    "yue": "Cantonese",
    "zh": "Simplified Chinese",
}


def _client():
    from openai import OpenAI

    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("VLM_API_KEY")
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError("QWEN_API_KEY / VLM_API_KEY not set for translation")
    return OpenAI(api_key=api_key, base_url=base_url)


def _model_name() -> str:
    return os.environ.get("QWEN_TRANSLATE_MODEL") or "qwen-plus"


def _translate_batch(
    client, model: str, lines: list[str], source_lang: str
) -> list[str]:
    src_label = _LANG_NAME.get(source_lang, source_lang)
    numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    prompt = (
        f"Translate the following {src_label} sentences from a comedy "
        "performance into Simplified Chinese (zh-CN). Match the comedic "
        "tone, preserve interjections and fillers (ええ→嗯, あー→啊, "
        "なんでやねん→搞什么啊, etc.). Use natural conversational Chinese.\n\n"
        "Output a single JSON array of translated strings, ONE PER INPUT "
        "LINE, in the same order. No commentary, no markdown, just the "
        "JSON array.\n\nInput:\n" + numbered
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise translator. Output ONLY JSON."},
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
            raise RuntimeError(f"translator did not return JSON: {raw[:200]}")
        out = json.loads(m.group(0))
    if not isinstance(out, list):
        raise RuntimeError(f"translator returned non-list: {type(out)}")
    if len(out) != len(lines):
        # Truncate or pad to keep alignment with the script body
        if len(out) < len(lines):
            out = out + [""] * (len(lines) - len(out))
        else:
            out = out[: len(lines)]
    return [str(x).strip() for x in out]


def translate_to_zh(lines: list[str], source_lang: str) -> Optional[list[str]]:
    """Translate `lines` (length N) into zh (returns length N).

    Returns None if no API key set or source_lang already starts with 'zh'."""
    if not lines:
        return None
    if (source_lang or "").lower().startswith("zh"):
        return None
    if not (os.environ.get("QWEN_API_KEY") or os.environ.get("VLM_API_KEY")):
        return None

    client = _client()
    model = _model_name()
    all_out: list[str] = []
    for i in range(0, len(lines), _BATCH_SIZE):
        batch = lines[i : i + _BATCH_SIZE]
        try:
            translated = _translate_batch(client, model, batch, source_lang)
        except Exception as e:
            print(f"  translate batch {i}-{i + len(batch)} failed: {e}")
            translated = ["" for _ in batch]
        all_out.extend(translated)
    return all_out
