from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class FetchResult:
    audio_path: Path
    title: str
    uploader: str
    upload_date: str
    duration_sec: int
    source_url: str
    platform: str
    raw_id: str


class Source(Protocol):
    def handles(self, url: str) -> bool: ...
    def fetch(self, url: str, work_dir: Path) -> FetchResult: ...
