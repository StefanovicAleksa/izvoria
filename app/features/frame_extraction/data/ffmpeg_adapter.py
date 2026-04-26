# Canvas: frame_ffmpeg_adapter_v7
# Path: app/features/frame_extraction/data/ffmpeg_adapter.py

import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

# noinspection PyUnresolvedReferences
from app.features.frame_extraction.domain.interfaces import IFFmpegExtractionAdapter
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk

logger = logging.getLogger(__name__)


class FFmpegExtractionAdapter:

    def extract_raw_frames(
            self,
            video_path: str,
            output_dir: Path,
            chunk: ContextualizedChunk,
            fps: int = 1
    ) -> List[Tuple[float, Path]]:
        """
        Batch extracts frames for a given chunk interval.
        Leverages the underlying resilient extract_frame method.
        """
        raw_frames = []
        current_sec = float(chunk.start_sec)

        while current_sec < chunk.end_sec:
            output_image_path = output_dir / f"frame_{current_sec:.2f}.jpg"

            # Use our hardened extraction logic for each frame
            success = self.extract_frame(
                video_path=video_path,
                timestamp_sec=current_sec,
                output_image_path=str(output_image_path)
            )

            if success and output_image_path.exists():
                raw_frames.append((current_sec, output_image_path))

            current_sec += (1.0 / fps)

        return raw_frames

    def extract_frame(self, video_path: str, timestamp_sec: float, output_image_path: str) -> bool:
        """
        Extracts a single high-quality frame from the video at the given timestamp.
        Returns True if successful, False otherwise.
        """
        cmd = [
            "ffmpeg", "-y",
            # --- FIX: Corrupted Video Resilience Flags ---
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            # ---------------------------------------------
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