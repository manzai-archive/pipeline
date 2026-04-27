from __future__ import annotations
import subprocess
from pathlib import Path

from .base import FetchResult


class LocalFileSource:
    """Accepts a path to any audio/video file on disk; transcodes to wav."""

    def handles(self, src: str) -> bool:
        try:
            p = Path(src).expanduser()
            return p.exists() and p.is_file()
        except (OSError, ValueError):
            return False

    def fetch(self, src: str, work_dir: Path) -> FetchResult:
        path = Path(src).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        out_audio = work_dir / f"{path.stem}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-ac", "1", "-ar", "16000", "-vn",
                "-loglevel", "error",
                str(out_audio),
            ],
            check=True,
        )
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True, text=True, check=True,
            )
            duration = int(float(result.stdout.strip()))
        except (subprocess.CalledProcessError, ValueError):
            duration = 0

        return FetchResult(
            audio_path=out_audio,
            title=path.stem,
            uploader="local",
            upload_date="",
            duration_sec=duration,
            source_url=f"file://{path}",
            platform="local",
            raw_id=path.stem,
        )
