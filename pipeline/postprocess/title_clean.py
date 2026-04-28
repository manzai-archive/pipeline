"""Clean a noisy YouTube/Bilibili title down to the bit's actual name.

Examples:
  "【春节必看小品】漫才兄弟碰上听不懂话的理发师 还冒出个"剪阑尾"的神剧情！|
   2025湖南卫视芒果TV春节联欢晚会 | MangoTV"
  → "听不懂话的理发师"

  "中川家の寄席2024「保険の契約」"
  → "保険の契約"
"""
from __future__ import annotations
import os
import re


def _client():
    from openai import OpenAI

    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("VLM_API_KEY")
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def clean_title(raw: str, group_display: str = "", language: str = "") -> str:
    """Return a cleaned title. Falls back to a regex-stripped version of
    raw if no LLM is available."""
    if not raw:
        return raw
    fallback = _regex_clean(raw)

    client = _client()
    if not client:
        return fallback

    model = (
        os.environ.get("QWEN_TITLE_MODEL")
        or os.environ.get("QWEN_POLISH_MODEL")
        or "qwen-max"
    )
    prompt = (
        "下面是一个漫才/相声/喜剧表演视频的原始标题（来自YouTube或B站）。"
        "请提炼出表演作品本身的名字，去掉所有宣传语、平台名、综艺名、"
        "演员名、季度年份、emoji、井号标签、感叹号、引号等装饰。"
        "如果原标题里有书名号《》或日文「」括起来的作品名，直接用那个名字。"
        "只输出最终标题，不要多余解释，不要加引号。\n\n"
        f"演员/团体: {group_display or '未知'}\n"
        f"语言: {language or '未知'}\n"
        f"原始标题: {raw}\n\n"
        "干净的作品标题:"
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only the cleaned title — no quotes, no commentary."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
        )
        out = (completion.choices[0].message.content or "").strip()
        # Strip outer quotes if model added them
        out = out.strip("「」『』\"'《》【】 ")
        # Single-line, no trailing punctuation noise
        out = out.splitlines()[0].strip() if out else fallback
        if not out or len(out) > 60:
            return fallback
        return out
    except Exception as e:
        import sys
        print(f"  title-clean: {e}; using regex fallback", file=sys.stderr)
        return fallback


_NOISE_TOKENS = [
    r"芒果TV[^|]*", r"MangoTV", r"CCTV[^|]*", r"央视[^|]*", r"湖南卫视[^|]*",
    r"春节联欢晚会", r"春晚", r"芒果", r"#\S+",
    r"【[^】]*】", r"\[[^\]]*\]",
    r"\(\d{4}\)", r"\d{4}年", r"\d{4}",
]


def _regex_clean(s: str) -> str:
    s = re.sub(r"\|.+$", "", s)  # drop after last pipe (often platform tail)
    for pat in _NOISE_TOKENS:
        s = re.sub(pat, "", s)
    s = re.sub(r"\s+", " ", s).strip(" |，,。.!！~-")
    return s.strip()
