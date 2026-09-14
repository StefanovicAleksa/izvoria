# Canvas: frame_ffmpeg_adapter_v8
# Path: app/features/frame_extraction/data/ffmpeg_adapter.py

import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings
from app.features.frame_extraction.domain.interfaces import IFFmpegExtractionAdapter
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk

logger = logging.getLogger(__name__)

# Corrupted-video resilience: keep decoding past broken packets instead of aborting
_RESILIENCE_FLAGS = ["-err_detect", "ignore_err", "-fflags", "+discardcorrupt"]


class FFmpegExtractionAdapter(IFFmpegExtractionAdapter):

    def extract_raw_frames(
            self,
            video_path: str | Path,
            output_dir: str | Path,
            chunk: ContextualizedChunk,
            fps: int = 1
    ) -> List[Tuple[float, Path]]:
        """
        Decodes the chunk's interval once and samples it at `fps`. One FFmpeg process per frame re-opened
        and re-seeked the video for every second of flagged footage (~3,600 processes per hour).
        Frame i of the output is stamped chunk.start_sec + i / fps.
        """
        start_sec = max(0.0, float(chunk.start_sec))
        duration_sec = float(chunk.end_sec) - start_sec
        if duration_sec <= 0:
            return []

        output_dir = Path(output_dir)
        cmd = [
            "ffmpeg", "-y",
            *_RESILIENCE_FLAGS,
            "-ss", f"{start_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{duration_sec:.3f}",
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality JPEG
            "-threads", str(settings.FRAME_FFMPEG_THREADS),
            "-start_number", "0",
            str(output_dir / "frame_%05d.jpg")
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=max(60.0, duration_sec * 4)
            )
        except subprocess.CalledProcessError as e:
            # Frames written before the failure are still valid evidence
            logger.error(f"FFmpeg frame extraction failed for {start_sec:.2f}s-{chunk.end_sec:.2f}s. Error: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg frame extraction timed out for {start_sec:.2f}s-{chunk.end_sec:.2f}s.")

        raw_frames: List[Tuple[float, Path]] = []
        for frame_path in sorted(output_dir.glob("frame_[0-9]*.jpg")):
            timestamp_sec = start_sec + int(frame_path.stem.split("_")[1]) / fps
            if timestamp_sec < chunk.end_sec:
                raw_frames.append((timestamp_sec, frame_path))

        if not raw_frames:
            # Sub-second or badly indexed intervals can yield nothing from the fps filter
            midpoint_sec = start_sec + duration_sec / 2.0
            midpoint_path = output_dir / "frame_midpoint.jpg"
            if self.extract_frame(video_path, midpoint_sec, str(midpoint_path)) and midpoint_path.exists():
                raw_frames.append((midpoint_sec, midpoint_path))

        return raw_frames

    def extract_frame(self, video_path: str | Path, timestamp_sec: float, output_image_path: str) -> bool:
        """
        Extracts a single high-quality frame from the video at the given timestamp.
        Returns True if successful, False otherwise.
        """
        cmd = [
            "ffmpeg", "-y",
            *_RESILIENCE_FLAGS,
            "-ss", str(timestamp_sec),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",  # High quality JPEG
            str(output_image_path)
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=30
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed at {timestamp_sec}s. Error: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg extraction timed out at {timestamp_sec}s.")
            return False
