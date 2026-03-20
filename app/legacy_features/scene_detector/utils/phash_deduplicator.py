# Canvas: scene_detector_phash_v2
# Path: app/features/scene_detector/utils/phash_deduplicator.py

from pathlib import Path
from typing import List, Tuple
from PIL import Image
import imagehash


class PhashDeduplicator:
    """
    CPU-bound utility to mathematically drop near-duplicate video frames.
    Built for Graceful Degradation: skips corrupted frames without crashing the batch.
    """

    def __init__(self, hamming_distance_threshold: int = 5):
        self.threshold = hamming_distance_threshold

    def deduplicate(self, extracted_scenes: List[Tuple[float, Path]]) -> List[Tuple[float, Path, str]]:
        if not extracted_scenes:
            return []

        # Ensure strict chronological order
        sorted_scenes = sorted(extracted_scenes, key=lambda x: x[0])

        unique_scenes: List[Tuple[float, Path, str]] = []
        previous_hash = None

        for timestamp, file_path in sorted_scenes:
            try:
                # 1. Open the image safely (using a context manager to release file locks)
                with Image.open(file_path) as img:
                    current_hash = imagehash.phash(img)

                # 2. Compare against the last unique frame
                is_duplicate = False
                if previous_hash is not None:
                    distance = current_hash - previous_hash
                    if distance <= self.threshold:
                        is_duplicate = True

                # 3. Route the frame
                if is_duplicate:
                    self._safe_delete(file_path)
                else:
                    unique_scenes.append((timestamp, file_path, str(current_hash)))
                    previous_hash = current_hash

            except Exception as e:
                # GRACEFUL DEGRADATION:
                # If a specific image is corrupted, delete it and skip to the next one.
                # We DO NOT crash the entire batch over one bad frame.
                print(f"[WARNING] Dropping corrupted frame at {timestamp}s: {e}")
                self._safe_delete(file_path)
                continue

        return unique_scenes

    def _safe_delete(self, file_path: Path) -> None:
        """Helper to ensure we never crash while trying to clean up temp files."""
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass