from __future__ import annotations
import json
import subprocess
from pathlib import Path

from .base import FetchResult


class YouTubeSource:
    def handles(self, url: str) -> bool:
        return "youtube.com" in url or "youtu.be" in url

    def fetch(self, url: str, work_dir: Path) -> FetchResult:
        return _ytdlp_fetch(url, work_dir, platform="youtube")


def _ytdlp_fetch(url: str, work_dir: Path, platform: str) -> FetchResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(work_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--write-info-json",
            "--no-playlist",
            "-o", out_template,
            url,
        ],
        check=True,
    )
    info_files = sorted(work_dir.glob("*.info.json"), key=lambda p: p.stat().st_mtime)
    if not info_files:
        raise RuntimeError("yt-dlp succeeded but no .info.json was written")
    info = json.loads(info_files[-1].read_text())
    audio = work_dir / f"{info['id']}.wav"
    if not audio.exists():
        raise RuntimeError(f"Expected audio {audio} not found")
    return FetchResult(
        audio_path=audio,
        title=info.get("title", ""),
        uploader=info.get("uploader") or info.get("channel") or "",
        upload_date=info.get("upload_date", ""),
        duration_sec=int(info.get("duration") or 0),
        source_url=info.get("webpage_url", url),
        platform=platform,
        raw_id=info["id"],
    )
