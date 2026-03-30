# Canvas: test_extraction_api_v1
# Path: tests/features/frame_extraction/service/test_extraction_api.py

import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.service.extraction_api import FrameExtractionAPI
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames


class TestFrameExtractionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repository": Mock(),
            "ffmpeg": Mock(),
            "storage": Mock(),
            "quality_filter": Mock(),
            "deduplicator": Mock()
        }

    @pytest.fixture
    def api(self, mocks):
        return FrameExtractionAPI(
            repository=mocks["repository"],
            ffmpeg_adapter=mocks["ffmpeg"],
            storage_api=mocks["storage"],
            quality_filter=mocks["quality_filter"],
            deduplicator=mocks["deduplicator"]
        )

    @pytest.fixture
    def sample_chunk(self):
        return ContextualizedChunk(
            chunk_id=uuid.uuid4(),
            start_sec=10.0,
            end_sec=15.0
        )

    def test_process_chunk_idempotency(self, api, mocks, sample_chunk):
        """Verifies that if a chunk is already processed, we skip extraction."""
        existing_record = MagicMock(spec=ExtractedChunkFrames)
        mocks["repository"].get_by_chunk_id.return_value = existing_record

        result = api.process_chunk(uuid.uuid4(), "vid.mp4", sample_chunk)

        assert result == existing_record
        mocks["ffmpeg"].extract_raw_frames.assert_not_called()

    @patch("app.features.frame_extraction.service.extraction_api.tempfile.TemporaryDirectory")
    def test_process_chunk_happy_path(self, mock_tempdir, api, mocks, sample_chunk):
        """Verifies the full extraction, filtering, and deduplication orchestration."""
        mocks["repository"].get_by_chunk_id.return_value = None
        mock_tempdir.return_value.__enter__.return_value = "/tmp/onyx"

        # Mock FFmpeg (Raw)
        raw_frames = [(10.0, Path("/tmp/onyx/frame_0001.jpg"))]
        mocks["ffmpeg"].extract_raw_frames.return_value = raw_frames

        # Mock Quality Filter (Pass)
        high_quality = [(10.0, Path("/tmp/onyx/frame_0001.jpg"), 150.0)]
        mocks["quality_filter"].filter_frames.return_value = high_quality

        # Mock Deduplicator (Pass)
        unique_frames = [(10.0, Path("/tmp/onyx/frame_0001.jpg"), 150.0, "hash123")]
        mocks["deduplicator"].deduplicate.return_value = unique_frames

        # Mock Vault Storage
        vaulted_file = MagicMock()
        vaulted_file.id = uuid.uuid4()
        mocks["storage"].ingest_raw_file.return_value = vaulted_file

        # Mock Repository Save
        mocks["repository"].save.side_effect = lambda x: x

        result = api.process_chunk(uuid.uuid4(), "vid.mp4", sample_chunk)

        assert isinstance(result, ExtractedChunkFrames)
        assert len(result.frames) == 1
        assert result.frames[0].phash_value == "hash123"
        assert result.frames[0].file_id == vaulted_file.id

        mocks["ffmpeg"].extract_raw_frames.assert_called_once()
        mocks["quality_filter"].filter_frames.assert_called_once_with(raw_frames)
        mocks["deduplicator"].deduplicate.assert_called_once_with(high_quality)
        mocks["storage"].ingest_raw_file.assert_called_once()
        mocks["repository"].save.assert_called_once()

    @patch("app.features.frame_extraction.service.extraction_api.tempfile.TemporaryDirectory")
    def test_process_chunk_empty_after_quality_filter(self, mock_tempdir, api, mocks, sample_chunk):
        """Verifies that if all frames are blurry, it skips storage and saves an empty map."""
        mocks["repository"].get_by_chunk_id.return_value = None
        mock_tempdir.return_value.__enter__.return_value = "/tmp/onyx"

        # FFmpeg extracts 1 frame
        raw_frames = [(10.0, Path("/tmp/onyx/frame_0001.jpg"))]
        mocks["ffmpeg"].extract_raw_frames.return_value = raw_frames

        # Quality Filter drops the frame (returns empty list)
        mocks["quality_filter"].filter_frames.return_value = []

        mocks["repository"].save.side_effect = lambda x: x

        result = api.process_chunk(uuid.uuid4(), "vid.mp4", sample_chunk)

        assert len(result.frames) == 0
        mocks["deduplicator"].deduplicate.assert_not_called()
        mocks["storage"].ingest_raw_file.assert_not_called()
        mocks["repository"].save.assert_called_once()