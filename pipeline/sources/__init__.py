from pathlib import Path

from .base import FetchResult, Source
from .bilibili import BilibiliSource
from .local import LocalFileSource
from .youtube import YouTubeSource

# Order matters: file paths checked first so they short-circuit before
# URL handlers see them.
REGISTRY: list[Source] = [LocalFileSource(), YouTubeSource(), BilibiliSource()]


def fetch(src: str, work_dir: Path) -> FetchResult:
    for source in REGISTRY:
        if source.handles(src):
            return source.fetch(src, work_dir)
    raise ValueError(f"No source handles: {src}")


__all__ = ["FetchResult", "Source", "fetch", "REGISTRY"]
