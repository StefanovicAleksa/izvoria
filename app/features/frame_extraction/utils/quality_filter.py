# Canvas: quality_filter_v3
# Path: app/features/frame_extraction/utils/quality_filter.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple

# noinspection PyPackageRequirements
import cv2

from app.core.config import settings

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    CPU-bound utility to filter out blurry or motion-heavy frames before they
    reach the database or GPU. Uses OpenCV Laplacian variance.
    """

    def __init__(self, blur_threshold: Optional[float] = None):
        self.blur_threshold = settings.FRAME_BLUR_THRESHOLD if blur_threshold is None else blur_threshold

    def filter_frames(
        self,
        extracted_frames: List[Tuple[float, Path]]
    ) -> List[Tuple[float, Path, float]]:
        """
        Scans frames for sharpness.
        Returns a list of tuples containing (timestamp, path, quality_score).

        If no frame reaches the threshold, the single sharpest one is kept anyway: low-texture evidence
        (a scanned page, a mostly white slide) scores low on Laplacian variance, and dropping every frame
        would leave a chunk that the cue detector flagged with no visual at all. A perfectly uniform frame
        (score 0, e.g. a black transition) carries nothing and is never kept.
        Rejected frames are deleted from disk.
        """
        scored_frames: List[Tuple[float, Path, float]] = []

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
                scored_frames.append((timestamp, file_path, float(laplacian_var)))

            except Exception as e:
                logger.warning(f"Failed to process quality check for {file_path.name}: {e}")
                self._safe_delete(file_path)

        valid_frames = [frame for frame in scored_frames if frame[2] >= self.blur_threshold]

        if not valid_frames and scored_frames:
            sharpest = max(scored_frames, key=lambda frame: frame[2])
            if sharpest[2] > 0.0:
                valid_frames = [sharpest]

        for frame in scored_frames:
            if frame not in valid_frames:
                self._safe_delete(frame[1])

        return valid_frames

    def _safe_delete(self, file_path: Path) -> None:
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            pass
