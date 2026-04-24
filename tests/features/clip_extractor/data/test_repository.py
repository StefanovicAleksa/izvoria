# Canvas: test_clip_extractor_repository_v1
# Path: tests/features/clip_extractor/data/test_repository.py

import pytest
import uuid
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.clip_extractor.data.repositories import VideoClipRepository
from app.features.clip_extractor.domain.entities import VideoClip
from app.features.clip_extractor.exceptions.database import ClipExtractorDatabaseError


class TestVideoClipRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        mock_session_ctx = MagicMock()
        db_manager.session.return_value = mock_session_ctx
        session = MagicMock()
        mock_session_ctx.__enter__.return_value = session
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return VideoClipRepository(mock_db_manager)

    def test_create_bulk_happy_path(self, repository, mock_db_manager):
        """Verifies bulk saving records passes to SQLAlchemy bulk_save_objects."""
        clips = [
            VideoClip(
                id=uuid.uuid4(),
                source_id=uuid.uuid4(),
                chunk_id=uuid.uuid4(),
                extracted_file_id=uuid.uuid4(),
                start_sec=10.0,
                end_sec=20.0
            ),
            VideoClip(
                id=uuid.uuid4(),
                source_id=uuid.uuid4(),
                chunk_id=uuid.uuid4(),
                extracted_file_id=uuid.uuid4(),
                start_sec=25.0,
                end_sec=35.0
            )
        ]

        result = repository.create_bulk(clips)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.bulk_save_objects.assert_called_once()

        saved_models = session.bulk_save_objects.call_args[0][0]
        assert len(saved_models) == 2
        assert saved_models[0].id == clips[0].id
        assert result == clips

    def test_create_bulk_db_error(self, repository, mock_db_manager):
        """Verifies DB errors during bulk save are wrapped."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.bulk_save_objects.side_effect = db_exc.DatabaseError("Bulk insert failed")

        clips = [
            VideoClip(id=uuid.uuid4(), source_id=uuid.uuid4(), chunk_id=uuid.uuid4(), extracted_file_id=uuid.uuid4(),
                      start_sec=0.0, end_sec=5.0)]

        with pytest.raises(ClipExtractorDatabaseError, match="Bulk insert failed"):
            repository.create_bulk(clips)