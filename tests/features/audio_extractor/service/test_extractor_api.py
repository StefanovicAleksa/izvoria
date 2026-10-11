# Canvas: test_extractor_api_v3
# Path: tests/features/audio_extractor/test_extractor_api.py

import pytest
import uuid
from pathlib import Path
from unittest.mock import Mock, MagicMock

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

    def test_extract_audio_idempotency_skip(self, api, mock_ffmpeg, mock_repo, mock_storage):
        """Verifies that if an extraction already exists, it is returned immediately without processing."""
        video_source_id = uuid.uuid4()
        existing_record = AudioExtraction(id=uuid.uuid4(), source_id=video_source_id, extracted_file_id=uuid.uuid4())

        # Simulate that the DB already has an extraction record
        mock_repo.get_latest_by_source_id.return_value = existing_record

        result = api.extract_audio_from_video(video_source_id)

        # Assert we returned the cached record and completely bypassed FFmpeg & Storage
        assert result == existing_record
        mock_repo.get_latest_by_source_id.assert_called_once_with(video_source_id)
        mock_storage.get_file_path.assert_called_once_with(existing_record.extracted_file_id)
        mock_storage.get_source.assert_not_called()
        mock_ffmpeg.extract_audio.assert_not_called()

    def test_extract_audio_happy_path(self, api, mock_ffmpeg, mock_repo, mock_storage):
        """Verifies the complete pipeline runs successfully on a fresh video."""
        video_source_id = uuid.uuid4()

        # 0. Mock Idempotency (no previous extraction)
        mock_repo.get_latest_by_source_id.return_value = None

        # 1. Mock Storage get_source (returns a valid VIDEO source)
        mock_source = MagicMock()
        mock_source.id = video_source_id
        mock_source.source_type = SourceType.VIDEO
        mock_source.file.file_path = "/vault/vid.mp4"
        mock_storage.get_source.return_value = mock_source

        # 2. Mock FFmpeg Adapter
        extracted_temp_path = Path("/tmp/izvoria_audio_123.wav")
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

    def test_extract_audio_invalid_source_type(self, api, mock_repo, mock_storage):
        """Verifies it fails fast if we pass an AUDIO or TEXT source ID."""
        mock_repo.get_latest_by_source_id.return_value = None
        mock_source = MagicMock()
        mock_source.source_type = SourceType.TEXT
        mock_storage.get_source.return_value = mock_source

        with pytest.raises(AudioExtractorError, match="is not a VIDEO"):
            api.extract_audio_from_video(uuid.uuid4())

    def test_extract_audio_cleans_up_on_storage_failure(
            self, api, mock_ffmpeg, mock_repo, mock_storage
    ):
        """Verifies temp files are deleted if Storage API crashes."""
        mock_repo.get_latest_by_source_id.return_value = None

        # Mock source validation
        mock_source = MagicMock()
        mock_source.source_type = SourceType.VIDEO
        mock_storage.get_source.return_value = mock_source

        # FFmpeg succeeds and returns a mock path object instead of a real Path
        mock_temp_path = MagicMock()
        mock_temp_path.exists.return_value = True
        mock_ffmpeg.extract_audio.return_value = mock_temp_path

        # Storage fails to ingest the raw file
        mock_storage.ingest_raw_file.side_effect = Exception("Storage Drive Full")

        with pytest.raises(Exception, match="Storage Drive Full"):
            api.extract_audio_from_video(uuid.uuid4())

        # CRITICAL VERIFICATION: Ensure the fallback cleanup mechanism ran directly on our mock!
        mock_temp_path.unlink.assert_called_once()

    def test_extract_audio_normalizes_audio_sources_too(self, api, mock_ffmpeg, mock_repo, mock_storage):
        """AUDIO sources get the same 16kHz mono WAV as videos so every phase reads identical samples."""
        source_id = uuid.uuid4()
        mock_repo.get_latest_by_source_id.return_value = None
        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.source_type = SourceType.AUDIO
        mock_source.file.file_path = "/vault/interview.m4a"
        mock_storage.get_source.return_value = mock_source
        mock_ffmpeg.extract_audio.return_value = Path("/tmp/izvoria_audio_456.wav")
        mock_storage.ingest_raw_file.return_value = MagicMock(id=uuid.uuid4())
        mock_repo.create.side_effect = lambda x: x

        result = api.extract_audio_from_video(source_id)

        mock_ffmpeg.extract_audio.assert_called_once_with("/vault/interview.m4a")
        assert result.source_id == source_id

    def test_get_extracted_audio_path_returns_vaulted_wav(self, api, mock_repo, mock_storage):
        source_id = uuid.uuid4()
        extraction = AudioExtraction(id=uuid.uuid4(), source_id=source_id, extracted_file_id=uuid.uuid4())
        mock_repo.get_latest_by_source_id.return_value = extraction
        mock_storage.get_file_path.return_value = Path("/vault/2026/09/14/audio.wav")

        assert api.get_extracted_audio_path(source_id) == "/vault/2026/09/14/audio.wav"
        mock_repo.get_latest_by_source_id.assert_called_once_with(source_id)
        mock_storage.get_file_path.assert_called_once_with(extraction.extracted_file_id)

    def test_get_extracted_audio_path_requires_prior_extraction(self, api, mock_repo, mock_storage):
        mock_repo.get_latest_by_source_id.return_value = None

        with pytest.raises(AudioExtractorError, match="No extracted audio"):
            api.get_extracted_audio_path(uuid.uuid4())

        mock_storage.get_file_path.assert_not_called()


class TestDiscardedAudio:
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

    def test_a_cached_extraction_whose_wav_is_gone_is_extracted_again(self, api, mock_ffmpeg, mock_repo,
                                                                       mock_storage):
        source_id = uuid.uuid4()
        stale = AudioExtraction(id=uuid.uuid4(), source_id=source_id, extracted_file_id=uuid.uuid4())
        mock_repo.get_latest_by_source_id.return_value = stale
        mock_storage.get_file_path.side_effect = FileNotFoundError("Physical file missing")
        source = MagicMock(id=source_id, source_type=SourceType.VIDEO)
        source.file.file_path = "/vault/vid.mp4"
        mock_storage.get_source.return_value = source
        mock_ffmpeg.extract_audio.return_value = Path("/tmp/izvoria_audio_1.wav")
        mock_storage.ingest_raw_file.return_value = MagicMock(id=uuid.uuid4())
        mock_repo.create.side_effect = lambda x: x

        result = api.extract_audio_from_video(source_id)

        mock_storage.delete_raw_file.assert_called_once_with(stale.extracted_file_id)
        mock_ffmpeg.extract_audio.assert_called_once_with("/vault/vid.mp4")
        assert result.extracted_file_id == mock_storage.ingest_raw_file.return_value.id

    def test_discard_deletes_the_wav_and_reports_its_size(self, api, mock_repo, mock_storage):
        source_id = uuid.uuid4()
        extraction = AudioExtraction(id=uuid.uuid4(), source_id=source_id, extracted_file_id=uuid.uuid4())
        mock_repo.get_latest_by_source_id.return_value = extraction
        mock_storage.get_file.return_value = MagicMock(file_size_bytes=115_200_000)

        assert api.discard_extracted_audio(source_id) == 115_200_000
        mock_storage.get_file.assert_called_once_with(extraction.extracted_file_id)
        mock_storage.delete_raw_file.assert_called_once_with(extraction.extracted_file_id)

    def test_a_dry_run_only_measures(self, api, mock_repo, mock_storage):
        mock_repo.get_latest_by_source_id.return_value = AudioExtraction(
            id=uuid.uuid4(), source_id=uuid.uuid4(), extracted_file_id=uuid.uuid4())
        mock_storage.get_file.return_value = MagicMock(file_size_bytes=10)

        assert api.discard_extracted_audio(uuid.uuid4(), dry_run=True) == 10
        mock_storage.delete_raw_file.assert_not_called()

    def test_a_source_without_audio_frees_nothing(self, api, mock_repo, mock_storage):
        mock_repo.get_latest_by_source_id.return_value = None

        assert api.discard_extracted_audio(uuid.uuid4()) == 0
        mock_storage.delete_raw_file.assert_not_called()
