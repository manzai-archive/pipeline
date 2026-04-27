from __future__ import annotations
from pathlib import Path

from .base import FetchResult
from .youtube import _ytdlp_fetch


class BilibiliSource:
    def handles(self, url: str) -> bool:
        u = url.lower()
        return "bilibili.com" in u or u.startswith("bv") or u.startswith("av")

    def fetch(self, url: str, work_dir: Path) -> FetchResult:
        return _ytdlp_fetch(url, work_dir, platform="bilibili")
