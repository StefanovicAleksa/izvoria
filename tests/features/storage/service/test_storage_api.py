# Canvas: test_storage_api_v3
# Path: tests/features/storage/test_storage_api.py

import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import cast

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import Source, SourceType, File
from app.features.storage.exceptions.base import StorageError
from app.features.storage.exceptions.not_found import FileNotFound


class TestStorageAPI:
    @pytest.fixture
    def mock_repo(self):
        return Mock()

    @pytest.fixture
    def api(self, mock_repo):
        with patch("app.features.storage.service.storage_api.FileManager") as MockFM:
            mock_fm_instance = MockFM.return_value
            mock_fm_instance.generate_path.return_value = Path("/mock/vault/video.mp4")
            mock_fm_instance.store_file.return_value = 1024

            service = StorageAPI(mock_repo)
            service.file_manager = mock_fm_instance
            yield service

    def test_get_file_happy_path(self, api):
        """Ensures the Storage API correctly proxies get_file directly to the repo."""
        repo_mock = cast(Mock, api.repository)
        mock_file = MagicMock(spec=File)
        file_id = uuid.uuid4()

        repo_mock.get_file_by_id.return_value = mock_file

        result = api.get_file(file_id)

        assert result == mock_file
        repo_mock.get_file_by_id.assert_called_once_with(file_id)

    @patch("app.features.storage.service.storage_api.Path")
    def test_get_file_path_happy_path(self, mock_path_cls, api):
        """Happy Path: Retrieves the physical Path object when DB and disk agree."""
        repo_mock = cast(Mock, api.repository)
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "/mock/vault/frame.jpg"
        file_id = uuid.uuid4()
        repo_mock.get_file_by_id.return_value = mock_file

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_cls.return_value = mock_path_obj

        result = api.get_file_path(file_id)

        assert result == mock_path_obj
        repo_mock.get_file_by_id.assert_called_once_with(file_id)
        mock_path_cls.assert_called_once_with("/mock/vault/frame.jpg")

    def test_get_file_path_db_record_missing(self, api):
        """Error Path: Fails immediately if the File UUID is not in the DB."""
        repo_mock = cast(Mock, api.repository)
        file_id = uuid.uuid4()
        repo_mock.get_file_by_id.return_value = None

        with pytest.raises(FileNotFound, match="Database record missing"):
            api.get_file_path(file_id)

    @patch("app.features.storage.service.storage_api.Path")
    def test_get_file_path_physical_file_missing(self, mock_path_cls, api):
        """Error Path: DB record exists, but file is missing from hard drive."""
        repo_mock = cast(Mock, api.repository)
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "/mock/vault/ghost_frame.jpg"
        file_id = uuid.uuid4()
        repo_mock.get_file_by_id.return_value = mock_file

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path_cls.return_value = mock_path_obj

        with pytest.raises(FileNotFoundError, match="Physical file missing"):
            api.get_file_path(file_id)

    @patch("app.features.storage.service.storage_api.mimetypes.guess_type")
    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_raw_file_happy_path(self, mock_path_cls, mock_guess, api):
        """Tests vaulting a standalone file without Source creation."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_obj.suffix = ".wav"
        mock_path_obj.name = "audio.wav"
        mock_path_cls.return_value = mock_path_obj

        mock_guess.return_value = ("audio/wav", None)

        repo_mock = cast(Mock, api.repository)
        fm_mock = cast(Mock, api.file_manager)

        repo_mock.create_file.side_effect = lambda f: f

        result = api.ingest_raw_file("/tmp/audio.wav")

        assert isinstance(result, File)
        assert result.file_name == "audio.wav"
        assert result.mime_type == "audio/wav"

        fm_mock.store_file.assert_called_once()
        repo_mock.create_file.assert_called_once()

    @patch("app.features.storage.service.storage_api.calculate_file_hash")
    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_new_file_happy_path(self, mock_path_cls, mock_hash_func, api):
        """Happy Path: Ingesting a NEW file as a Source."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_obj.suffix = ".mp4"
        mock_path_obj.name = "video.mp4"
        mock_path_cls.return_value = mock_path_obj

        mock_hash_func.return_value = "new_unique_hash"

        repo_mock = cast(Mock, api.repository)
        fm_mock = cast(Mock, api.file_manager)

        repo_mock.get_by_hash.return_value = None
        repo_mock.create.side_effect = lambda source: source

        result = api.ingest_local_file("/tmp/video.mp4", SourceType.VIDEO)

        assert result.content_hash == "new_unique_hash"
        assert result.name == "video.mp4"
        assert result.file_id is not None

        mock_hash_func.assert_called_once()
        repo_mock.get_by_hash.assert_called_with("new_unique_hash")
        fm_mock.store_file.assert_called_once()
        repo_mock.create.assert_called_once()

    @patch("app.features.storage.service.storage_api.calculate_file_hash")
    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_duplicate_file_returns_existing(self, mock_path_cls, mock_hash_func, api):
        """Deduplication Path: Ingesting an EXISTING file."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_cls.return_value = mock_path_obj

        mock_hash_func.return_value = "existing_hash"

        repo_mock = cast(Mock, api.repository)
        fm_mock = cast(Mock, api.file_manager)

        existing_source = Source(
            id=uuid.uuid4(),
            name="old_video.mp4",
            source_type=SourceType.VIDEO,
            content_hash="existing_hash",
            file_id=uuid.uuid4()
        )
        repo_mock.get_by_hash.return_value = existing_source

        result = api.ingest_local_file("/tmp/new_upload.mp4", SourceType.VIDEO)

        assert result.id == existing_source.id
        assert result.name == "old_video.mp4"

        fm_mock.store_file.assert_not_called()
        repo_mock.create.assert_not_called()

    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_file_not_found(self, mock_path_cls, api):
        """Error Path: File does not exist on disk."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_cls.return_value = mock_path_obj

        with pytest.raises(FileNotFoundError):
            api.ingest_local_file("ghost.mp4", SourceType.VIDEO)

    @patch("app.features.storage.service.storage_api.calculate_file_hash")
    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_repo_failure(self, mock_path_cls, mock_hash_func, api):
        """Error Path: Database failure during create."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_obj.name = "video.mp4"
        mock_path_cls.return_value = mock_path_obj

        mock_hash_func.return_value = "hash"

        repo_mock = cast(Mock, api.repository)
        repo_mock.get_by_hash.return_value = None
        repo_mock.create.side_effect = StorageError("DB Crashed")

        with pytest.raises(StorageError, match="DB Crashed"):
            api.ingest_local_file("video.mp4", SourceType.VIDEO)

    def test_delete_raw_file_removes_record_and_physical_file(self, api):
        repo_mock = cast(Mock, api.repository)
        fm_mock = cast(Mock, api.file_manager)
        file_id = uuid.uuid4()
        deleted = File(id=file_id, file_path="/mock/vault/clip.mp4", file_name="clip.mp4", file_size_bytes=10)
        repo_mock.delete_file.return_value = deleted

        assert api.delete_raw_file(file_id) is True

        repo_mock.delete_file.assert_called_once_with(file_id)
        fm_mock.delete.assert_called_once_with("/mock/vault/clip.mp4")

    def test_delete_raw_file_missing_record_is_noop(self, api):
        repo_mock = cast(Mock, api.repository)
        fm_mock = cast(Mock, api.file_manager)
        repo_mock.delete_file.return_value = None

        assert api.delete_raw_file(uuid.uuid4()) is False
        fm_mock.delete.assert_not_called()
