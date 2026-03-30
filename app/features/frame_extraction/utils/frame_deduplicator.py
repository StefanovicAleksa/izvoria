# Canvas: frame_deduplicator_v1
# Path: app/features/frame_extraction/utils/frame_deduplicator.py

import logging
import imagehash
from PIL import Image
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class FrameDeduplicator:
    """
    CPU-bound utility to process sequential, high-quality image frames, calculating their
    perceptual hashes, and safely deleting visually identical frames from disk.
    """

    def __init__(self, hamming_threshold: int = 5):
        # A threshold of 5 allows for minor video compression artifacts
        # while confidently catching static slides or locked-off camera shots.
        self.hamming_threshold = hamming_threshold

    def deduplicate(
            self,
            quality_filtered_frames: List[Tuple[float, Path, float]]
    ) -> List[Tuple[float, Path, float, str]]:
        """
        Scans chronologically sorted image paths.
        Returns a list of tuples: (timestamp, path, quality_score, phash).
        """
        unique_frames: List[Tuple[float, Path, float, str]] = []
        previous_hash = None

        for timestamp, file_path, quality_score in quality_filtered_frames:
            if not file_path.exists():
                continue

            try:
                # 1. Open image safely with context manager to release OS file locks
                with Image.open(file_path) as img:
                    current_hash = imagehash.phash(img)

                # 2. Compare against the last unique frame
                is_duplicate = False
                if previous_hash is not None:
                    distance = current_hash - previous_hash
                    if distance <= self.hamming_threshold:
                        is_duplicate = True

                # 3. Route the frame
                if is_duplicate:
                    self._safe_delete(file_path)
                else:
                    unique_frames.append((timestamp, file_path, quality_score, str(current_hash)))
                    previous_hash = current_hash

            except Exception as e:
                logger.warning(f"Failed to generate pHash for {file_path.name}: {e}")
                self._safe_delete(file_path)

        return unique_frames

    def _safe_delete(self, file_path: Path) -> None:
        """Helper to ensure we never crash while cleaning up duplicate files."""
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass