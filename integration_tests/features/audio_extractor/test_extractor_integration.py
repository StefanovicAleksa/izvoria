# Canvas: test_extractor_api_integration_v2
# Path: integration_tests/features/audio_extractor/test_extractor_api_integration.py

import os
import subprocess
import pytest
from pathlib import Path

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import SourceType

from app.features.audio_extractor.data.repositories import AudioExtractionRepository
from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.domain.entities import AudioExtraction


class TestAudioExtractorAPIIntegration:
    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def storage_api(self, tmp_path):
        repo = StorageRepository(db_manager)
        api = StorageAPI(repo)
        api.file_manager.base_path = tmp_path / "test_vault"
        api.file_manager.base_path.mkdir(parents=True, exist_ok=True)
        return api

    @pytest.fixture
    def extractor_api(self, storage_api):
        ffmpeg_adapter = FFmpegAdapter(timeout_seconds=30)
        repo = AudioExtractionRepository(db_manager)
        return AudioExtractorAPI(ffmpeg_adapter, repo, storage_api)

    def generate_real_dummy_video(self, output_path: Path, include_audio: bool = True):
        """Helper to generate a real, readable MP4 file using FFmpeg for integration testing."""
        if include_audio:
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=128x128:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
                "-c:v", "libx264", "-c:a", "aac", str(output_path)
            ]
        else:
            # Generates a video track ONLY, no audio stream
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=128x128:rate=10",
                "-c:v", "libx264", str(output_path)
            ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_extract_audio_idempotency_and_success(self, storage_api, extractor_api, tmp_path):
        # [Keep your existing test_extract_audio_happy_path / idempotency test logic exactly as it was]
        source_dir = tmp_path / "inputs"
        source_dir.mkdir()
        source_file = source_dir / "real_test_video.mp4"
        self.generate_real_dummy_video(source_file, include_audio=True)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Test Video Extraction"
        )

        # First Extraction
        extraction_record_1 = extractor_api.extract_audio_from_video(video_source.id)
        assert isinstance(extraction_record_1, AudioExtraction)

        # Idempotency Check
        extraction_record_2 = extractor_api.extract_audio_from_video(video_source.id)
        assert extraction_record_1.id == extraction_record_2.id

    def test_extract_audio_silent_video_fallback(self, storage_api, extractor_api, tmp_path):
        """Ensures the API successfully processes a video completely lacking an audio track."""
        source_dir = tmp_path / "inputs_silent"
        source_dir.mkdir()
        source_file = source_dir / "silent_test_video.mp4"

        # Create a video with NO audio stream
        self.generate_real_dummy_video(source_file, include_audio=False)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Silent Video Extraction"
        )

        # This should NOT crash, it should generate a silent WAV and vault it
        extraction_record = extractor_api.extract_audio_from_video(video_source.id)

        assert isinstance(extraction_record, AudioExtraction)
        assert extraction_record.source_id == video_source.id

        # Verify the vaulted file exists and is accessible
        vaulted_audio = storage_api.get_file(extraction_record.extracted_file_id)
        assert vaulted_audio is not None
        assert Path(vaulted_audio.file_path).exists()
        assert Path(vaulted_audio.file_path).stat().st_size > 0  # Not a zero-byte file