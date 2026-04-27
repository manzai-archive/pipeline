from pathlib import Path

from .base import FetchResult, Source
from .bilibili import BilibiliSource
from .youtube import YouTubeSource

REGISTRY: list[Source] = [YouTubeSource(), BilibiliSource()]


def fetch(url: str, work_dir: Path) -> FetchResult:
    for src in REGISTRY:
        if src.handles(url):
            return src.fetch(url, work_dir)
    raise ValueError(f"No source handles URL: {url}")


__all__ = ["FetchResult", "Source", "fetch", "REGISTRY"]
