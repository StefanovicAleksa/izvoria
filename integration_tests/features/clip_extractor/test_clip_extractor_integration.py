# Canvas: test_clip_extractor_integration_v2
# Path: integration_tests/features/clip_extractor/test_clip_extractor_integration.py

import pytest
import os
import uuid
import subprocess
from pathlib import Path

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.clip_extractor.data.repositories import VideoClipRepository
from app.features.clip_extractor.data.ffmpeg_video_adapter import FFmpegVideoAdapter
from app.features.clip_extractor.service.extractor_api import ClipExtractorAPI

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk


class TestClipExtractorIntegration:
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
        """Wires up the REAL Storage Vault, REAL FFmpeg Adapter, and REAL Database."""
        # 1. Setup Storage
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 2. Setup Clip Extractor
        clip_repo = VideoClipRepository(db_manager)
        ffmpeg_adapter = FFmpegVideoAdapter(timeout_seconds=60)

        extractor_api = ClipExtractorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            repository=clip_repo,
            storage_api=storage_api
        )

        return extractor_api, storage_api

    @staticmethod
    def generate_real_720p_video(output_path: Path) -> None:
        """
        Forces FFmpeg to render a real, physical 720p video at 30fps.
        This will take actual CPU time and generate a file roughly 100KB - 500KB in size.
        """
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=5",
            "-c:v", "libx264",  # Force actual H.264 encoding
            "-preset", "ultrafast",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_clip_extraction_integration(self, api_setup, tmp_path):
        """
        Scenario:
        1. A REAL 720p dummy video is encoded and ingested into the Storage Vault.
        2. We construct a UnifiedTranscript with two distinct chunks.
        3. Extractor API invokes REAL FFmpeg to slice the two clips.
        4. Extractor API vaults the new clips and saves REAL database lineage.
        5. Physical file checks prove the clips have actual byte mass.
        """
        extractor_api, storage_api = api_setup

        # ---------------------------------------------------------
        # PHASE 1: PREPARATION
        # ---------------------------------------------------------
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        source_file = source_dir / "real_test_video.mp4"

        # This will noticeably pause the test while your CPU encodes the video
        self.generate_real_720p_video(source_file)

        assert source_file.exists()
        assert source_file.stat().st_size > 50000  # Ensure the file is at least 50KB

        # Ingest the video to the Vault
        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Test Clip Extraction Webinar"
        )

        dummy_transcript = UnifiedTranscript(
            source_id=video_source.id,
            chunks=[
                UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0.5, end_sec=1.5, text_payload="First point."),
                UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=3.0, end_sec=4.0, text_payload="Second point.")
            ]
        )

        # ---------------------------------------------------------
        # PHASE 2: FIRST EXTRACTION
        # ---------------------------------------------------------
        extracted_clips = extractor_api.extract_clips_from_transcript(dummy_transcript)

        assert len(extracted_clips) == 2

        # Verify physical files in the vault.
        vault_mp4_files_1 = list(storage_api.file_manager.base_path.rglob("*.mp4"))
        assert len(vault_mp4_files_1) == 3

        # Prove the extracted clips are real by checking their mass
        for file_path in vault_mp4_files_1:
            assert file_path.stat().st_size > 1000  # Extracted chunks shouldn't be empty 0-byte files

        # ---------------------------------------------------------
        # PHASE 3: SECOND EXTRACTION (Idempotency Check)
        # ---------------------------------------------------------
        extracted_clips_2 = extractor_api.extract_clips_from_transcript(dummy_transcript)

        assert len(extracted_clips_2) == 2

        # Assert FFmpeg wasn't run again (no duplicate video files in vault)
        vault_mp4_files_2 = list(storage_api.file_manager.base_path.rglob("*.mp4"))
        assert len(vault_mp4_files_2) == 3