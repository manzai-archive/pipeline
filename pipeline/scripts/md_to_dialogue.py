"""One-shot migration: convert each entry's in-body `**Speaker** [ts] text`
lines into a structured dialogue YAML at sibling
`web/src/content/dialogues/<slug>.yaml`, then rewrite the entry .md so it
contains only frontmatter (body emptied).

Run inside container:
  python -m pipeline.scripts.md_to_dialogue <entry.md>
  # or batch:
  for f in /workspace/web/src/content/manzai/*.md; do
    python -m pipeline.scripts.md_to_dialogue "$f"
  done
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import yaml


_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$")


def migrate(md_path: Path) -> None:
    text = md_path.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{md_path}: missing frontmatter")
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]

    # Move per-utterance translations out of frontmatter
    legacy_translations = fm.pop("translations", None) or {}

    utterances: list[dict] = []
    for raw in body.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        sp, ts, txt = m.groups()
        idx = len(utterances)
        u: dict = {
            "id": f"u{idx + 1}",
            "t": ts,
            "speaker": sp.strip(),
            "text": txt.strip(),
        }
        # Per-utterance translations: pull from legacy fm.translations[lang][i]
        per_u_translations: dict = {}
        for lang, arr in legacy_translations.items():
            if isinstance(arr, list) and idx < len(arr) and isinstance(arr[idx], str):
                per_u_translations[lang] = arr[idx]
        if per_u_translations:
            u["translations"] = per_u_translations
        utterances.append(u)

    if not utterances:
        print(f"  no utterances in body of {md_path.name}; skipping")
        return

    # Determine target dialogues directory: web/src/content/dialogues/
    content_root = md_path.parent.parent  # .../content/
    dialogues_dir = content_root / "dialogues"
    dialogues_dir.mkdir(parents=True, exist_ok=True)
    slug = md_path.stem
    out_yaml = dialogues_dir / f"{slug}.yaml"
    out_yaml.write_text(
        yaml.safe_dump(
            {"utterances": utterances},
            allow_unicode=True,
            sort_keys=False,
        )
    )

    # Rewrite entry .md with empty body (frontmatter only).
    fm_yaml = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()
    md_path.write_text(f"---\n{fm_yaml}\n---\n")
    print(f"  {md_path.name}: {len(utterances)} utterances → {out_yaml.name}")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m pipeline.scripts.md_to_dialogue <entry.md> [entry2.md ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        migrate(Path(arg).resolve())


if __name__ == "__main__":
    main()
