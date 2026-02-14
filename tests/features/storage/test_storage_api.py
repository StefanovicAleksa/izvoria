import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import cast

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.domain.entities import Source, SourceType
from app.features.storage.exceptions.base import StorageError


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

    @patch("app.features.storage.service.storage_api.calculate_file_hash")
    @patch("app.features.storage.service.storage_api.Path")
    def test_ingest_new_file_happy_path(self, mock_path_cls, mock_hash_func, api):
        """Happy Path: Ingesting a NEW file."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        # FIX: resolve() must return the same mock
        mock_path_obj.resolve.return_value = mock_path_obj
        mock_path_obj.suffix = ".mp4"
        # FIX: Pydantic needs a real string
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
            content_hash="existing_hash"
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
        # FIX: The code calls resolve() first, so resolve() must return the object checking for existence
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
        # FIX: Pydantic needs a real string
        mock_path_obj.name = "video.mp4"
        mock_path_cls.return_value = mock_path_obj

        mock_hash_func.return_value = "hash"

        repo_mock = cast(Mock, api.repository)
        repo_mock.get_by_hash.return_value = None

        repo_mock.create.side_effect = StorageError("DB Crashed")

        with pytest.raises(StorageError, match="DB Crashed"):
            api.ingest_local_file("video.mp4", SourceType.VIDEO)