from __future__ import annotations
from pathlib import Path

import yaml


def load_prompt(path: Path) -> str:
    if not path.exists():
        return ""
    data = yaml.safe_load(path.read_text()) or {}
    parts: list[str] = []
    for section in ("groups", "people", "shows"):
        for item in data.get(section, []) or []:
            parts.append(str(item))
    return "、".join(parts)
