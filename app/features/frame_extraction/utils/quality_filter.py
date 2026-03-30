# Canvas: quality_filter_v2
# Path: app/features/frame_extraction/utils/quality_filter.py

import logging
from pathlib import Path
from typing import List, Tuple

# noinspection PyPackageRequirements
import cv2

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    CPU-bound utility to filter out blurry or motion-heavy frames before they
    reach the database or GPU. Uses OpenCV Laplacian variance.
    """

    def __init__(self, blur_threshold: float = 100.0):
        self.blur_threshold = blur_threshold

    def filter_frames(
        self,
        extracted_frames: List[Tuple[float, Path]]
    ) -> List[Tuple[float, Path, float]]:
        """
        Scans frames for sharpness.
        Returns a list of tuples containing (timestamp, path, quality_score).
        Deletes failed frames from disk immediately.
        """
        valid_frames: List[Tuple[float, Path, float]] = []

        for timestamp, file_path in extracted_frames:
            if not file_path.exists():
                continue

            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    self._safe_delete(file_path)
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                if laplacian_var >= self.blur_threshold:
                    valid_frames.append((timestamp, file_path, float(laplacian_var)))
                else:
                    self._safe_delete(file_path)

            except Exception as e:
                logger.warning(f"Failed to process quality check for {file_path.name}: {e}")
                self._safe_delete(file_path)

        return valid_frames

    def _safe_delete(self, file_path: Path) -> None:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass