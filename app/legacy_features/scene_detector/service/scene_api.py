# Canvas: scene_api_v2
# Path: app/features/scene_detector/service/scene_api.py

import uuid
import tempfile
from pathlib import Path
from typing import List, Optional

from app.legacy_features.scene_detector.domain.entities import SceneMap, DetectedScene, TimeWindow
from app.legacy_features.scene_detector.domain.interfaces import IFFmpegSceneAdapter
from app.legacy_features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.legacy_features.scene_detector.utils.error_mapper import map_scene_detector_error

from app.features.storage.service.storage_api import StorageAPI


class SceneDetectorAPI:
    """
    The orchestrating brain of the Scene Detector feature.
    Coordinates CPU-bound frame extraction, mathematical deduplication,
    and vault storage. Returns an in-memory SceneMap for the Intelligence Bridge.
    """

    def __init__(
            self,
            ffmpeg_adapter: IFFmpegSceneAdapter,
            storage_api: StorageAPI,
            deduplicator: PhashDeduplicator
    ):
        self._ffmpeg = ffmpeg_adapter
        self._storage = storage_api
        self._deduplicator = deduplicator

    def process_video(
            self,
            source_id: uuid.UUID,
            video_path: str,
            threshold: float = 0.4,
            windows: Optional[List[TimeWindow]] = None
    ) -> SceneMap:
        """
        Executes the end-to-end scene detection pipeline for a given video.

        Args:
            source_id: The UUID of the original Video Source.
            video_path: The physical OS path to the video file.
            threshold: FFmpeg pixel variance threshold (0.0 to 1.0).
            windows: Optional list of specific time boundaries to constrain FFmpeg scanning.

        Returns:
            An in-memory SceneMap containing the timeline of unique visual frames.
        """
        try:
            # 1. Secure Temporary Directory Context
            with tempfile.TemporaryDirectory(prefix="izvoria_scenes_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 2. CPU-Bound Extraction (Fast-Seeking via FFmpeg Adapter)
                raw_scenes = self._ffmpeg.extract_scenes(
                    video_path=video_path,
                    output_dir=tmp_path,
                    threshold=threshold,
                    windows=windows
                )

                if not raw_scenes:
                    return SceneMap(source_id=source_id, scenes=[])

                # 3. CPU-Bound Deduplication (Perceptual Hashing)
                unique_scenes = self._deduplicator.deduplicate(raw_scenes)

                # 4. Vault Ingestion & Domain Mapping
                detected_scenes: List[DetectedScene] = []

                for timestamp_sec, img_path, phash_val in unique_scenes:
                    vaulted_file = self._storage.ingest_raw_file(
                        file_path=str(img_path),
                        delete_original=True
                    )

                    detected_scene = DetectedScene(
                        timestamp_sec=timestamp_sec,
                        extracted_file_id=vaulted_file.id,
                        phash_value=phash_val
                    )
                    detected_scenes.append(detected_scene)

                # 5. Build and return the in-memory Map
                return SceneMap(
                    source_id=source_id,
                    scenes=detected_scenes
                )

        except Exception as e:
            raise map_scene_detector_error(e, context=f"Source ID {source_id}") from e