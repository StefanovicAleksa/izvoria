# Canvas: test_extraction_api_integration_v8
# Path: integration_tests/features/frame_extraction/test_extraction_api_integration.py

import os
import pytest
import subprocess
import uuid
from pathlib import Path

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager as global_db_manager

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter
from app.features.frame_extraction.data.repositories import FrameExtractionRepository
from app.features.frame_extraction.service.extraction_api import FrameExtractionAPI
from app.features.frame_extraction.utils.quality_filter import QualityFilter
from app.features.frame_extraction.utils.frame_deduplicator import FrameDeduplicator
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames


class TestExtractionAPIIntegration:
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
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except FileNotFoundError:
            pytest.skip("FFmpeg is not installed on this system.")

    @pytest.fixture
    def storage_api(self, tmp_path):
        repo = StorageRepository(global_db_manager)
        api = StorageAPI(repo)
        api.file_manager.base_path = tmp_path / "test_vault"
        api.file_manager.base_path.mkdir(parents=True, exist_ok=True)
        return api

    @pytest.fixture
    def extraction_api(self, storage_api):
        ffmpeg_adapter = FFmpegExtractionAdapter()
        repo = FrameExtractionRepository(global_db_manager)
        quality_filter = QualityFilter()
        deduplicator = FrameDeduplicator()

        # noinspection PyTypeChecker
        return FrameExtractionAPI(
            repository=repo,
            ffmpeg_adapter=ffmpeg_adapter,
            storage_api=storage_api,
            quality_filter=quality_filter,
            deduplicator=deduplicator
        )

    @staticmethod
    def generate_real_dummy_video(output_path: Path):
        """Generates a real 5-second video file using FFmpeg for testing."""
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc=duration=5:size=320x240:rate=1",
            "-c:v", "libx264", str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_chunk_extraction_batch(self, storage_api, extraction_api, tmp_path):
        """
        Tests the entire extraction flow:
        1. Ingesting a real video.
        2. Extracting frames for a specific chunk.
        3. Saving frames to the vault and PostgreSQL.
        4. Verifying idempotency on repeat runs.
        """
        # ---------------------------------------------------------
        # PHASE 1: SETUP REAL VIDEO AND MOCK CHUNK
        # ---------------------------------------------------------
        source_dir = tmp_path / "inputs"
        source_dir.mkdir()
        source_file = source_dir / "real_test_video.mp4"
        self.generate_real_dummy_video(source_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Frame Extraction Test Video"
        )

        test_chunk = ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            start_sec=1.0,
            end_sec=4.0,
            original_text="This is a test chunk.",
            needs_visual=True
        )

        # ---------------------------------------------------------
        # PHASE 2: FIRST EXTRACTION RUN
        # ---------------------------------------------------------
        assert video_source.file is not None  # Explicit assert for strict typing

        extraction_records_1 = extraction_api.process_chunks_batch(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            chunks=[test_chunk]
        )

        assert len(extraction_records_1) == 1
        extraction_record_1 = extraction_records_1[0]

        assert isinstance(extraction_record_1, ExtractedChunkFrames)
        assert extraction_record_1.source_id == video_source.id
        assert extraction_record_1.chunk_id == test_chunk.chunk_id

        assert len(extraction_record_1.frames) == 3

        assert extraction_record_1.frames[0].timestamp_sec == 1.0
        assert extraction_record_1.frames[1].timestamp_sec == 2.0
        assert extraction_record_1.frames[2].timestamp_sec == 3.0

        vault_jpgs_1 = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs_1) == 3

        # ---------------------------------------------------------
        # PHASE 3: IDEMPOTENCY CHECK
        # ---------------------------------------------------------
        assert video_source.file is not None

        extraction_records_2 = extraction_api.process_chunks_batch(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            chunks=[test_chunk]
        )
        extraction_record_2 = extraction_records_2[0]

        assert extraction_record_1.id == extraction_record_2.id

        vault_jpgs_2 = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs_2) == 3