# Canvas: scene_api_v1
# Path: app/features/scene_detector/service/scene_api.py

import uuid
import tempfile
from pathlib import Path
from typing import List, Optional

from app.features.scene_detector.domain.entities import SceneMap, DetectedScene, TimeWindow
from app.features.scene_detector.domain.interfaces import ISceneRepository, IFFmpegSceneAdapter
from app.features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.features.scene_detector.utils.error_mapper import map_scene_detector_error

from app.features.storage.service.storage_api import StorageAPI


class SceneDetectorAPI:
    """
    The orchestrating brain of the Scene Detector feature.
    Coordinates CPU-bound frame extraction, mathematical deduplication,
    vault storage, and database persistence.
    """

    def __init__(
            self,
            repository: ISceneRepository,
            ffmpeg_adapter: IFFmpegSceneAdapter,
            storage_api: StorageAPI,
            deduplicator: PhashDeduplicator
    ):
        self._repo = repository
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
            A persisted SceneMap containing the timeline of unique visual frames.
        """
        # 1. Idempotency Check
        existing_map = self._repo.get_map_by_source_id(source_id)
        if existing_map:
            return existing_map

        try:
            # 2. Secure Temporary Directory Context
            # This guarantees that even if Python crashes, the OS cleans up the extracted JPEGs
            with tempfile.TemporaryDirectory(prefix="onyx_scenes_") as tmpdir:
                tmp_path = Path(tmpdir)

                # 3. CPU-Bound Extraction (Fast-Seeking via FFmpeg Adapter)
                raw_scenes = self._ffmpeg.extract_scenes(
                    video_path=video_path,
                    output_dir=tmp_path,
                    threshold=threshold,
                    windows=windows
                )

                if not raw_scenes:
                    # Return an empty map if no scenes met the threshold
                    empty_map = SceneMap(source_id=source_id, scenes=[])
                    return self._repo.save_map(empty_map)

                # 4. CPU-Bound Deduplication (Perceptual Hashing)
                unique_scenes = self._deduplicator.deduplicate(raw_scenes)

                # 5. Vault Ingestion & Domain Mapping
                detected_scenes: List[DetectedScene] = []

                for timestamp_sec, img_path, phash_val in unique_scenes:
                    # Vault the physical JPEG into our highly secure Storage feature
                    # delete_original=True removes it from the temp folder to save space instantly
                    vaulted_file = self._storage.ingest_raw_file(
                        file_path=str(img_path),
                        delete_original=True
                    )

                    # Create the immutable Domain Entity
                    detected_scene = DetectedScene(
                        timestamp_sec=timestamp_sec,
                        extracted_file_id=vaulted_file.id,
                        phash_value=phash_val
                    )
                    detected_scenes.append(detected_scene)

                # 6. Build and Persist the Aggregate Map
                scene_map = SceneMap(
                    source_id=source_id,
                    scenes=detected_scenes
                )

                return self._repo.save_map(scene_map)

        except Exception as e:
            # Catch everything, map it to our strict SceneDetectorError hierarchy, and bubble it up
            raise map_scene_detector_error(e, context=f"Source ID {source_id}") from e