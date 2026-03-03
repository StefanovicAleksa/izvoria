# Canvas: test_scene_detector_integration_v3
# Path: integration_tests/features/scene_detector/test_scene_detector_integration.py

import os
import pytest
import subprocess
from pathlib import Path

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager as global_db_manager

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.features.scene_detector.service.scene_api import SceneDetectorAPI
from app.features.scene_detector.utils.phash_deduplicator import PhashDeduplicator
from app.features.scene_detector.domain.entities import SceneMap, TimeWindow


class TestSceneDetectorIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL and FFmpeg installed on the host.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = global_db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(autouse=True)
    def check_ffmpeg_installed(self):
        """Skip these integration tests if FFmpeg is not installed on the host machine."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system. Skipping real integration test.")

    @pytest.fixture
    def api_setup(self, db_manager, tmp_path):
        """Wires up the REAL Storage Vault, REAL FFmpeg Adapter, REAL Deduplicator, and REAL Database."""
        # 1. Setup Storage
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 2. Setup Scene Detector (Now strictly stateless and functional)
        ffmpeg_adapter = FFmpegSceneAdapter(timeout_seconds=60)
        deduplicator = PhashDeduplicator(hamming_distance_threshold=5)

        scene_api = SceneDetectorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            storage_api=storage_api,
            deduplicator=deduplicator
        )

        return scene_api, storage_api

    @staticmethod
    def generate_hard_cut_video(output_path: Path) -> None:
        """
        Generates a 4-second video: 2 seconds of pure black, then 2 seconds of pure white.
        This guarantees a 100% mathematical scene change at exactly 2.0 seconds.
        """
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=white:s=320x240:d=2",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out_video]",
            "-map", "[out_video]",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_global_scene_extraction(self, api_setup, tmp_path):
        """
        Scenario: Global Pass (No Windows)
        1. A synthetic video with a hard cut at 2.0s is generated.
        2. The Scene API scans the whole video.
        3. Asserts exactly 1 scene was extracted at ~2.0s.
        4. Asserts the extracted frame is physically in the Storage Vault.
        """
        scene_api, storage_api = api_setup

        # PREP: Generate and Ingest the synthetic video
        source_dir = tmp_path / "test_media"
        source_dir.mkdir()
        video_file = source_dir / "hard_cut.mp4"
        self.generate_hard_cut_video(video_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(video_file),
            source_type=SourceType.VIDEO,
            name="Global Scene Test"
        )

        # ACTION: Run the Scene Detector with a strict threshold
        # We use threshold=0.8 (80% pixel change) to ignore any minor compression artifacts
        scene_map = scene_api.process_video(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            threshold=0.8
        )

        # ASSERTIONS
        assert isinstance(scene_map, SceneMap)
        assert scene_map.source_id == video_source.id

        # We expect exactly 1 major scene change at the transition from black to white
        assert len(scene_map.scenes) == 1

        # FFmpeg timestamps aren't perfectly precise down to the millisecond due to frame rates.
        # We allow a small tolerance window around the 2.0 second mark.
        extracted_time = scene_map.scenes[0].timestamp_sec
        assert 1.9 <= extracted_time <= 2.1, f"Expected scene at ~2.0s, got {extracted_time}s"

        # Verify the physical JPEG was vaulted
        vault_jpgs = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs) == 1

        # Verify cross-reference integrity
        vaulted_file_id = scene_map.scenes[0].extracted_file_id
        db_file = storage_api.get_file(vaulted_file_id)
        assert db_file is not None
        assert db_file.mime_type in ["image/jpeg", "image/jpg"]

    def test_windowed_scene_extraction(self, api_setup, tmp_path):
        """
        Scenario: Targeted Pass (With Time Windows)
        We provide a TimeWindow of [3.0s to 4.0s].
        Since the hard cut happens at 2.0s, the detector should find ZERO scenes in this window.
        """
        scene_api, storage_api = api_setup

        source_dir = tmp_path / "test_media_2"
        source_dir.mkdir()
        video_file = source_dir / "window_test.mp4"
        self.generate_hard_cut_video(video_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(video_file),
            source_type=SourceType.VIDEO
        )

        # ACTION: Pass a window that intentionally misses the scene change
        windows = [TimeWindow(start_sec=3.0, end_sec=4.0)]

        scene_map = scene_api.process_video(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            threshold=0.8,
            windows=windows
        )

        # ASSERTIONS
        assert isinstance(scene_map, SceneMap)
        assert len(scene_map.scenes) == 0  # Should be totally empty

        # Verify NO JPEGs were vaulted
        vault_jpgs = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs) == 0