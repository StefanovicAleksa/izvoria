# filename: test_repository_v2.py
# path: tests/features/storage/test_repository.py
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import Source, SourceType, File
from app.features.storage.exceptions.base import StorageError
from app.features.storage.exceptions.not_found import SourceNotFound, FileNotFound
from app.features.storage.data.models import FileModel


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

    def test_get_file_by_id_found(self, repository, mock_db_manager):
        """Should fetch a physical file directly by ID."""
        session = mock_db_manager.session.return_value.__enter__.return_value

        file_id = uuid.uuid4()
        mock_file_model = MagicMock()
        mock_file_model.id = file_id
        mock_file_model.file_path = "/tmp/vaulted_audio.wav"
        mock_file_model.file_name = "vaulted_audio.wav"
        mock_file_model.file_size_bytes = 2048
        mock_file_model.mime_type = "audio/wav"
        mock_file_model.file_metadata = {}
        mock_file_model.is_processed = False
        mock_file_model.created_at = datetime.now(timezone.utc)

        session.execute.return_value.scalar_one_or_none.return_value = mock_file_model

        result = repository.get_file_by_id(file_id)

        assert result is not None
        assert result.id == file_id
        assert result.file_path == "/tmp/vaulted_audio.wav"

    def test_get_file_by_id_not_found(self, repository, mock_db_manager):
        """Should return None if the file ID is not in the DB."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = None

        result = repository.get_file_by_id(uuid.uuid4())

        assert result is None

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
            language="sr",
            file_id=file_id,
            file=file_entity
        )

        result = repository.create(source_entity)

        session = mock_db_manager.session.return_value.__enter__.return_value

        assert session.add.call_count == 2
        added_source_model = session.add.call_args_list[1][0][0]
        assert added_source_model.content_hash == "abc-123"
        assert added_source_model.file_id == file_id
        assert added_source_model.language == "sr"

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
        mock_model.language = "sr"
        mock_model.file_id = uuid.uuid4()
        mock_model.created_at = datetime.now(timezone.utc)
        mock_model.updated_at = datetime.now(timezone.utc)
        mock_model.file = None

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.get_by_hash("found-hash")

        assert result is not None
        assert result.content_hash == "found-hash"
        assert result.language == "sr"
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
        mock_model.language = None
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

    @pytest.mark.parametrize("language", ["sl", None])
    def test_set_language_stamps_or_clears_the_source(self, repository, mock_db_manager, language):
        session = mock_db_manager.session.return_value.__enter__.return_value
        source_id = uuid.uuid4()
        model = MagicMock(id=source_id, source_type="video", content_hash="h", language="sr", file=None,
                          file_id=uuid.uuid4(), created_at=datetime.now(timezone.utc),
                          updated_at=datetime.now(timezone.utc))
        model.name = "clip.mp4"
        session.get.return_value = model

        result = repository.set_language(source_id, language)

        session.get.assert_called_once()
        assert session.get.call_args[0][1] == source_id
        assert model.language == language
        assert result.language == language

    def test_set_language_on_a_missing_source_raises(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.get.return_value = None

        with pytest.raises(SourceNotFound):
            repository.set_language(uuid.uuid4(), "sr")

    def test_delete_file_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        file_id = uuid.uuid4()
        mock_file_model = MagicMock()
        mock_file_model.id = file_id
        mock_file_model.file_path = "/vault/clip.mp4"
        mock_file_model.file_name = "clip.mp4"
        mock_file_model.file_size_bytes = 10
        mock_file_model.mime_type = "video/mp4"
        mock_file_model.file_metadata = {}
        mock_file_model.is_processed = False
        mock_file_model.created_at = datetime.now(timezone.utc)
        session.get.return_value = mock_file_model

        result = repository.delete_file(file_id)

        assert result.id == file_id
        assert result.file_path == "/vault/clip.mp4"
        session.delete.assert_called_once_with(mock_file_model)

    def test_delete_file_not_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.get.return_value = None

        assert repository.delete_file(uuid.uuid4()) is None
        session.delete.assert_not_called()

    def test_update_file_rewrites_location_size_and_type(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        file_id = uuid.uuid4()
        model = MagicMock()
        model.id = file_id
        model.file_name = "lecture.mkv"
        model.file_metadata = {}
        model.is_processed = True
        model.created_at = datetime.now(timezone.utc)
        session.get.return_value = model
        updated = File(id=file_id, file_path="/vault/new.mp4", file_name="lecture.mkv", file_size_bytes=7,
                       mime_type="video/mp4")

        result = repository.update_file(updated)

        session.get.assert_called_once_with(FileModel, file_id)
        assert (model.file_path, model.file_size_bytes, model.mime_type) == ("/vault/new.mp4", 7, "video/mp4")
        assert result.file_path == "/vault/new.mp4"
        assert result.file_name == "lecture.mkv"

    def test_update_file_on_a_missing_record_raises(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.get.return_value = None

        with pytest.raises(FileNotFound):
            repository.update_file(File(file_path="/vault/x.mp4", file_name="x.mp4", file_size_bytes=1))
