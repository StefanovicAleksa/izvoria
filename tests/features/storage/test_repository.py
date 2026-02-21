# filename: tests/features/storage/test_repository.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import Source, SourceType, File
from app.features.storage.exceptions.base import StorageError
from app.features.storage.exceptions.not_found import SourceNotFound


class TestStorageRepository:
    @pytest.fixture
    def mock_db_manager(self):
        """Mock the Database Manager and the Session."""
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx

        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return StorageRepository(mock_db_manager)

    def test_create_file_success(self, repository, mock_db_manager):
        """Should save a standalone File record."""
        file_entity = File(
            id=uuid.uuid4(),
            file_path="/tmp/audio.wav",
            file_name="audio.wav",
            file_size_bytes=500
        )

        result = repository.create_file(file_entity)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]

        assert added_model.file_path == "/tmp/audio.wav"
        assert result == file_entity

    def test_create_happy_path(self, repository, mock_db_manager):
        """Should convert Domain Entity -> DB Model and add both File and Source to session."""
        file_id = uuid.uuid4()
        file_entity = File(
            id=file_id,
            file_path="/tmp/vid.mp4",
            file_name="vid.mp4",
            file_size_bytes=100
        )
        source_entity = Source(
            id=uuid.uuid4(),
            name="Test Video",
            source_type=SourceType.VIDEO,
            content_hash="abc-123",
            file_id=file_id,
            file=file_entity
        )

        result = repository.create(source_entity)

        session = mock_db_manager.session.return_value.__enter__.return_value

        # Verify session.add() was called twice (once for File, once for Source)
        assert session.add.call_count == 2

        # Verify the Source model contains the mapped file_id
        added_source_model = session.add.call_args_list[1][0][0]
        assert added_source_model.content_hash == "abc-123"
        assert added_source_model.file_id == file_id

        assert result == source_entity

    def test_create_db_error(self, repository, mock_db_manager):
        """Should catch db_exc.DatabaseError and raise StorageError."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.side_effect = db_exc.SystemError("Connection Dead")

        source = Source(
            id=uuid.uuid4(),
            name="T",
            source_type=SourceType.TEXT,
            content_hash="123",
            file_id=uuid.uuid4()
        )

        with pytest.raises(StorageError, match="storage error"):
            repository.create(source)

    def test_get_by_hash_found(self, repository, mock_db_manager):
        """Should execute a SELECT statement and return a mapped Domain Entity."""
        session = mock_db_manager.session.return_value.__enter__.return_value

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.name = "Test Video"
        mock_model.source_type = "video"
        mock_model.content_hash = "found-hash"
        mock_model.file_id = uuid.uuid4()
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.updated_at = datetime.now(timezone.utc)
        mock_model.file = None

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.get_by_hash("found-hash")

        assert result is not None
        assert result.content_hash == "found-hash"
        assert result.source_type == SourceType.VIDEO
        assert result.file_id == mock_model.file_id

    def test_get_by_hash_not_found(self, repository, mock_db_manager):
        """Should return None if DB returns None."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = repository.get_by_hash("ghost-hash")

        assert result is None

    def test_mark_as_processed_success(self, repository, mock_db_manager):
        """Should fetch model, update flag, and return entity."""
        session = mock_db_manager.session.return_value.__enter__.return_value

        mock_file = MagicMock()
        mock_file.id = uuid.uuid4()
        mock_file.file_path = "/tmp/test.mp4"
        mock_file.file_name = "test.mp4"
        mock_file.file_size_bytes = 1024
        mock_file.mime_type = "video/mp4"
        mock_file.file_metadata = {}
        mock_file.is_processed = False
        mock_file.created_at = datetime.now(timezone.utc)

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.name = "Processed Video"
        mock_model.source_type = "video"
        mock_model.content_hash = "hash_123"
        mock_model.file_id = mock_file.id
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.updated_at = datetime.now(timezone.utc)
        mock_model.file = mock_file

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.mark_as_processed(uuid.uuid4())

        assert mock_file.is_processed is True
        assert result.file.is_processed is True

    def test_mark_as_processed_not_found(self, repository, mock_db_manager):
        """Should raise SourceNotFound if ID doesn't exist."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        with pytest.raises(SourceNotFound):
            repository.mark_as_processed(uuid.uuid4())