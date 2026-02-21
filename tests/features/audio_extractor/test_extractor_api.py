# filename: tests/features/audio_extractor/test_extractor_api.py

import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.features.audio_extractor.service.extractor_api import AudioExtractorAPI
from app.features.audio_extractor.domain.entities import AudioExtraction
from app.features.audio_extractor.exceptions.base import AudioExtractorError
from app.features.storage.domain.entities import SourceType


class TestAudioExtractorAPI:
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
        return AudioExtractorAPI(mock_ffmpeg, mock_repo, mock_storage)

    def test_extract_audio_happy_path(self, api, mock_ffmpeg, mock_repo, mock_storage):
        """Verifies the complete pipeline runs successfully."""
        video_source_id = uuid.uuid4()

        # 1. Mock Storage get_source (returns a valid VIDEO source)
        mock_source = MagicMock()
        mock_source.id = video_source_id
        mock_source.source_type = SourceType.VIDEO
        mock_source.file.file_path = "/vault/vid.mp4"
        mock_storage.get_source.return_value = mock_source

        # 2. Mock FFmpeg Adapter
        extracted_temp_path = Path("/tmp/onyx_audio_123.wav")
        mock_ffmpeg.extract_audio.return_value = extracted_temp_path

        # 3. Mock Storage ingest_raw_file
        mock_audio_file = MagicMock()
        mock_audio_file.id = uuid.uuid4()
        mock_storage.ingest_raw_file.return_value = mock_audio_file

        # 4. Mock Repository Create
        mock_repo.create.side_effect = lambda x: x

        # Execute
        result = api.extract_audio_from_video(video_source_id)

        # Assertions
        assert isinstance(result, AudioExtraction)
        assert result.source_id == video_source_id
        assert result.extracted_file_id == mock_audio_file.id

        mock_storage.get_source.assert_called_once_with(video_source_id)
        mock_ffmpeg.extract_audio.assert_called_once_with("/vault/vid.mp4")
        mock_storage.ingest_raw_file.assert_called_once_with(
            file_path=str(extracted_temp_path),
            delete_original=True
        )
        mock_repo.create.assert_called_once()

    def test_extract_audio_invalid_source_type(self, api, mock_storage):
        """Verifies it fails fast if we pass an AUDIO or TEXT source ID."""
        mock_source = MagicMock()
        mock_source.source_type = SourceType.TEXT
        mock_storage.get_source.return_value = mock_source

        with pytest.raises(AudioExtractorError, match="is not a VIDEO"):
            api.extract_audio_from_video(uuid.uuid4())

    @patch("app.features.audio_extractor.service.extractor_api.Path.exists")
    @patch("app.features.audio_extractor.service.extractor_api.Path.unlink")
    def test_extract_audio_cleans_up_on_storage_failure(
            self, mock_unlink, mock_exists, api, mock_ffmpeg, mock_storage
    ):
        """Verifies temp files are deleted if Storage API crashes."""
        # Mock source validation
        mock_source = MagicMock()
        mock_source.source_type = SourceType.VIDEO
        mock_storage.get_source.return_value = mock_source

        # FFmpeg succeeds
        mock_ffmpeg.extract_audio.return_value = Path("/tmp/onyx_audio_123.wav")
        mock_exists.return_value = True  # Pretend the file exists for cleanup logic

        # Storage fails to ingest the raw file
        mock_storage.ingest_raw_file.side_effect = Exception("Storage Drive Full")

        with pytest.raises(Exception, match="Storage Drive Full"):
            api.extract_audio_from_video(uuid.uuid4())

        # CRITICAL VERIFICATION: Ensure the fallback cleanup mechanism ran!
        mock_unlink.assert_called_once()