# Canvas: test_extractor_integration_v2
# Path: integration_tests/features/audio_extractor/test_extractor_integration.py

import pytest
import os
import subprocess
from pathlib import Path

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.domain.entities import AudioExtraction


class TestAudioExtractorIntegration:
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
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        audio_repo = AudioExtractionRepository(db_manager)
        ffmpeg_adapter = FFmpegAdapter(timeout_seconds=30)

        extractor_api = AudioExtractorAPI(
            ffmpeg_adapter=ffmpeg_adapter,
            repository=audio_repo,
            storage_api=storage_api
        )

        return extractor_api, storage_api

    @staticmethod
    def generate_real_dummy_video(output_path: Path) -> None:
        """Uses FFmpeg to generate a real 1-second blank video with silent audio."""
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=10x10:d=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_extraction_integration(self, api_setup, tmp_path):
        """
        Scenario:
        1. A REAL dummy video is ingested into the REAL Storage Vault.
        2. Extractor API invokes REAL FFmpeg to extract audio.
        3. Extractor API vaults the new audio and saves REAL database lineage.
        4. (IDEMPOTENCY) Extractor API is called again, bypassing FFmpeg entirely.
        """
        extractor_api, storage_api = api_setup

        # ---------------------------------------------------------
        # PHASE 1: PREPARATION
        # ---------------------------------------------------------
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        source_file = source_dir / "real_test_video.mp4"
        self.generate_real_dummy_video(source_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Test Video Extraction"
        )

        # ---------------------------------------------------------
        # PHASE 2: FIRST EXTRACTION (Heavy Lifting)
        # ---------------------------------------------------------
        extraction_record_1 = extractor_api.extract_audio_from_video(video_source.id)

        assert isinstance(extraction_record_1, AudioExtraction)
        assert extraction_record_1.source_id == video_source.id

        vault_wav_files_1 = list(storage_api.file_manager.base_path.rglob("*.wav"))
        assert len(vault_wav_files_1) == 1

        # ---------------------------------------------------------
        # PHASE 3: SECOND EXTRACTION (Idempotency Check)
        # ---------------------------------------------------------
        extraction_record_2 = extractor_api.extract_audio_from_video(video_source.id)

        # Assert we got the exact same database record back
        assert extraction_record_1.id == extraction_record_2.id

        # Assert FFmpeg wasn't run again (no duplicate audio files in vault)
        vault_wav_files_2 = list(storage_api.file_manager.base_path.rglob("*.wav"))
        assert len(vault_wav_files_2) == 1