# Canvas: scene_detector_interfaces_v2
# Path: app/features/scene_detector/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from pathlib import Path

from app.legacy_features.scene_detector.domain.entities import TimeWindow


class IFFmpegSceneAdapter(ABC):
    """
    Contract for the external FFmpeg tool.
    The implementation in the Data layer will handle the complex OS subprocess commands.
    """

    @abstractmethod
    def extract_scenes(
            self,
            video_path: str | Path,
            output_dir: str | Path,
            threshold: float = 0.4,
            windows: Optional[List[TimeWindow]] = None
    ) -> List[Tuple[float, Path]]:
        """
        Scans a video and extracts JPEG frames where pixel variance exceeds the threshold.

        Args:
            video_path: Physical path to the source video.
            output_dir: Temporary directory to dump the extracted JPEG frames.
            threshold: Pixel variance sensitivity (0.0 to 1.0).
            windows: If provided, FFmpeg must ONLY scan within these specific time boundaries.

        Returns:
            A list of tuples containing the exact timestamp and the physical Path to the extracted JPEG.
        """
        pass