# Canvas: ffmpeg_extraction_adapter_v3
# Path: app/features/frame_extraction/data/ffmpeg_adapter.py

import subprocess
from pathlib import Path
from typing import List, Tuple

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.interfaces import IFFmpegExtractionAdapter
from app.features.frame_extraction.utils.error_mapper import map_extraction_error


class FFmpegExtractionAdapter(IFFmpegExtractionAdapter):
    """
    Adapter for executing OS-level FFmpeg commands to slice out physical frames
    based on a chunk's exact time boundaries.
    Now automatically downscales frames to a max dimension of 1024px to save Vault
    storage space and accelerate downstream VLM prefill times.
    """

    def __init__(self, timeout_seconds: int = 300, max_dimension: int = 1024):
        self._timeout = timeout_seconds
        self._max_dimension = max_dimension

    def extract_raw_frames(
            self,
            video_path: str | Path,
            output_dir: str | Path,
            chunk: ContextualizedChunk,
            fps: int = 1
    ) -> List[Tuple[float, Path]]:

        src_path = Path(video_path).resolve()
        out_dir = Path(output_dir).resolve()

        try:
            if not src_path.exists():
                raise FileNotFoundError(f"Source video missing: {src_path}")

            out_dir.mkdir(parents=True, exist_ok=True)

            start_sec = float(chunk.start_sec)
            end_sec = float(chunk.end_sec)
            duration = max(0.1, end_sec - start_sec)

            # Build the video filter string:
            # 1. Extract at the specified FPS
            # 2. Scale down to max_dimension (preserving aspect ratio, preventing upscaling)
            vf_string = (
                f"fps={fps},"
                f"scale='min({self._max_dimension},iw)':'min({self._max_dimension},ih)':"
                f"force_original_aspect_ratio=decrease"
            )

            # Build the fast-seeking FFmpeg command
            command: list[str] = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-ss", str(start_sec),
                "-t", str(duration),
                "-i", str(src_path),
                "-vf", vf_string,
                "-q:v", "2",
                str(out_dir / "frame_%04d.jpg")
            ]

            # Execute the extraction
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout
            )

            extracted_frames: List[Tuple[float, Path]] = []
            frame_files = sorted(out_dir.glob("frame_*.jpg"))

            for index, frame_path in enumerate(frame_files):
                absolute_timestamp = start_sec + (index * (1.0 / fps))
                extracted_frames.append((absolute_timestamp, frame_path))

            return extracted_frames

        except Exception as e:
            raise map_extraction_error(e, context=f"FFmpeg extraction for chunk {chunk.chunk_id}") from e