# Canvas: test_extraction_api_integration_v1
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
        """Wires up the REAL Storage Vault, REAL Extractors, REAL CPU Filters, and REAL Database."""
        # 1. Setup Storage Vault
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir(parents=True, exist_ok=True)

        # 2. Setup Frame Extraction Dependencies
        extraction_repo = FrameExtractionRepository(db_manager)
        ffmpeg_adapter = FFmpegExtractionAdapter(timeout_seconds=60)

        # Using a safer blur threshold for compressed synthetic video
        quality_filter = QualityFilter(blur_threshold=50.0)
        deduplicator = FrameDeduplicator(hamming_threshold=5)

        # 3. Build Orchestrating Service
        extraction_api = FrameExtractionAPI(
            repository=extraction_repo,
            ffmpeg_adapter=ffmpeg_adapter,
            storage_api=storage_api,
            quality_filter=quality_filter,
            deduplicator=deduplicator
        )

        return extraction_api, storage_api

    @staticmethod
    def generate_synthetic_video(output_path: Path) -> None:
        """
        Generates a 5-second video with a ticking timestamp and scrolling color bars.
        This guarantees sharp edges (passes Quality Filter) and changing pixels (passes Deduplicator).
        """
        command: list[str] = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-c:v", "libx264",
            os.fspath(output_path)
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def test_end_to_end_chunk_extraction(self, api_setup, tmp_path):
        """
        Scenario:
        1. A REAL dummy video is generated and ingested into the REAL Storage Vault.
        2. A ContextualizedChunk is simulated for [1.0s -> 4.0s].
        3. Extractor API invokes REAL FFmpeg to fast-seek and extract frames.
        4. Frames pass through REAL OpenCV Blur Detection and pHash Deduplication.
        5. Surviving frames are vaulted, and JSONB mapping is saved to Postgres.
        6. Idempotency is verified by running it again.
        """
        extraction_api, storage_api = api_setup

        # ---------------------------------------------------------
        # PHASE 1: PREPARATION & INGESTION
        # ---------------------------------------------------------
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        source_file = source_dir / "synthetic_test.mp4"
        self.generate_synthetic_video(source_file)

        video_source = storage_api.ingest_local_file(
            file_path=os.fspath(source_file),
            source_type=SourceType.VIDEO,
            name="Extraction Integration Test"
        )

        # Create the simulated logical text chunk targeting the middle 3 seconds
        test_chunk = ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            start_sec=1.0,
            end_sec=4.0,
            original_text="The speaker points to the ticking clock.",
            context_string="A demonstration of time passing."
        )

        # ---------------------------------------------------------
        # PHASE 2: FIRST EXTRACTION (Heavy Lifting)
        # ---------------------------------------------------------
        extraction_record_1 = extraction_api.process_chunk(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            chunk=test_chunk
        )

        # Assertions
        assert isinstance(extraction_record_1, ExtractedChunkFrames)
        assert extraction_record_1.source_id == video_source.id
        assert extraction_record_1.chunk_id == test_chunk.chunk_id

        # At 1 fps for 3 seconds (1.0 to 4.0), we expect exactly 3 frames
        assert len(extraction_record_1.frames) == 3

        # Verify timestamps map accurately to the chunk bounds
        assert extraction_record_1.frames[0].timestamp_sec == 1.0
        assert extraction_record_1.frames[1].timestamp_sec == 2.0
        assert extraction_record_1.frames[2].timestamp_sec == 3.0

        # Verify the physical JPEGs are securely in the Vault
        vault_jpgs_1 = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs_1) == 3

        # ---------------------------------------------------------
        # PHASE 3: IDEMPOTENCY CHECK
        # ---------------------------------------------------------
        extraction_record_2 = extraction_api.process_chunk(
            source_id=video_source.id,
            video_path=video_source.file.file_path,
            chunk=test_chunk
        )

        # Verify the exact same DB record was returned instantly
        assert extraction_record_1.id == extraction_record_2.id

        # Verify FFmpeg was bypassed (no duplicate junk frames in the vault)
        vault_jpgs_2 = list(storage_api.file_manager.base_path.rglob("*.jpg"))
        assert len(vault_jpgs_2) == 3