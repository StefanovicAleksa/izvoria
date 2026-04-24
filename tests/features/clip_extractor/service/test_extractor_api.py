# Canvas: test_clip_extractor_api_v2
# Path: tests/features/clip_extractor/service/test_extractor_api.py

import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock, MagicMock

from app.features.clip_extractor.service.extractor_api import ClipExtractorAPI
from app.features.clip_extractor.domain.entities import VideoClip
from app.features.clip_extractor.exceptions.base import ClipExtractorError
from app.features.storage.domain.entities import SourceType
from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk


class TestClipExtractorAPI:
    @pytest.fixture
    def mock_ffmpeg(self):
        return Mock()

    @pytest.fixture
    def mock_repo(self):
        return Mock()

    @pytest.fixture
    def mock_storage(self):
        return Mock()

    @pytest.fixture
    def api(self, mock_ffmpeg, mock_repo, mock_storage):
        return ClipExtractorAPI(mock_ffmpeg, mock_repo, mock_storage)

    @pytest.fixture
    def dummy_transcript(self):
        return UnifiedTranscript(
            source_id=uuid.uuid4(),
            chunks=[
                UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=10.0, end_sec=20.0, text_payload="Chunk 1"),
                UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=30.0, end_sec=40.0, text_payload="Chunk 2")
            ]
        )

    def test_extract_clips_idempotency_skip(self, api, mock_ffmpeg, mock_repo, dummy_transcript):
        """If clips for this source ID exist, skip extraction and return them."""
        existing_clips = [MagicMock(spec=VideoClip)]
        mock_repo.get_by_source_id.return_value = existing_clips

        result = api.extract_clips_from_transcript(dummy_transcript)

        assert result == existing_clips
        mock_repo.get_by_source_id.assert_called_once_with(dummy_transcript.source_id)

        # FIX: Call assertion directly on the mock fixture to satisfy IDE type hinting
        mock_ffmpeg.extract_clip.assert_not_called()

    def test_extract_clips_invalid_source_type(self, api, mock_repo, mock_storage, dummy_transcript):
        """Verifies it fails fast if we pass an AUDIO or TEXT source ID."""
        mock_repo.get_by_source_id.return_value = []
        mock_source = MagicMock()
        mock_source.source_type = SourceType.TEXT
        mock_storage.get_source.return_value = mock_source

        # FIX: Utilize the imported base exception to verify strict validation
        with pytest.raises(ClipExtractorError, match="is not a VIDEO"):
            api.extract_clips_from_transcript(dummy_transcript)

    def test_extract_clips_happy_path(self, api, mock_ffmpeg, mock_repo, mock_storage, dummy_transcript):
        """Verifies loops and storage logic work as expected."""
        mock_repo.get_by_source_id.return_value = []

        mock_source = MagicMock()
        mock_source.source_type = SourceType.VIDEO
        mock_source.file.file_path = "/vault/test.mp4"
        mock_storage.get_source.return_value = mock_source

        # FFmpeg returns a fake path
        mock_ffmpeg.extract_clip.return_value = Path("/tmp/onyx_clip.mp4")

        # Storage returns a vaulted file
        vaulted_file = MagicMock()
        vaulted_file.id = uuid.uuid4()
        mock_storage.ingest_raw_file.return_value = vaulted_file

        result = api.extract_clips_from_transcript(dummy_transcript)

        assert len(result) == 2
        assert mock_ffmpeg.extract_clip.call_count == 2
        assert mock_storage.ingest_raw_file.call_count == 2
        mock_repo.create_bulk.assert_called_once()

        # Verify buffers were applied (start - 0.5, end + 0.5)
        first_ffmpeg_call = mock_ffmpeg.extract_clip.call_args_list[0][1]
        assert first_ffmpeg_call['start_sec'] == 9.5
        assert first_ffmpeg_call['end_sec'] == 20.5

    def test_extract_clips_cleans_up_on_storage_failure(self, api, mock_ffmpeg, mock_repo, mock_storage,
                                                        dummy_transcript):
        """Verifies the specific temp file is deleted if Storage API crashes during loop."""
        mock_repo.get_by_source_id.return_value = []

        mock_source = MagicMock()
        mock_source.source_type = SourceType.VIDEO
        mock_storage.get_source.return_value = mock_source

        mock_temp_path = MagicMock()
        mock_temp_path.exists.return_value = True
        mock_ffmpeg.extract_clip.return_value = mock_temp_path

        mock_storage.ingest_raw_file.side_effect = Exception("Storage Drive Full")

        with pytest.raises(Exception, match="Storage Drive Full"):
            api.extract_clips_from_transcript(dummy_transcript)

        mock_temp_path.unlink.assert_called_once()