# Canvas: ffmpeg_video_adapter_v1
# Path: app/features/clip_extractor/data/ffmpeg_video_adapter.py

import os
import subprocess
import tempfile
from pathlib import Path

from app.features.clip_extractor.domain.interfaces import IFFmpegVideoAdapter
from app.features.clip_extractor.utils.error_mapper import map_clip_extractor_error


class FFmpegVideoAdapter(IFFmpegVideoAdapter):
    def __init__(self, timeout_seconds: int = 300):
        # Timeouts are critical in enterprise software to prevent hung CPU threads
        self._timeout = timeout_seconds

    def extract_clip(self, source_path: str | Path, start_sec: float, end_sec: float) -> Path:
        src = Path(source_path).resolve()

        if not src.exists():
            raw_error = FileNotFoundError(f"Source file not found at path: {src}")
            raise map_clip_extractor_error(raw_error)

        # 1. Secure Temporary File Management
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=".mp4", prefix="onyx_clip_")
        os.close(temp_fd)
        temp_path = Path(temp_path_str)

        # 2. FFmpeg Execution Command
        # Placing -ss before -i enables fast seeking to the nearest keyframe.
        # -c copy avoids expensive CPU re-encoding by just copying the stream.
        command: list[str] = [
            "ffmpeg",
            "-ss", str(start_sec),
            "-i", os.fspath(src),
            "-to", str(end_sec),
            "-c", "copy",
            "-y",  # Overwrite output file if it exists
            os.fspath(temp_path)
        ]

        try:
            # 3. Execute
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self._timeout
            )
            return temp_path

        except Exception as e:
            # Fallback Cleanup: Erase the temporary file if FFmpeg crashes mid-write
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise map_clip_extractor_error(e, context=f"Extracting {start_sec} to {end_sec} from {src.name}") from e