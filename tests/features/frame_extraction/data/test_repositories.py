# Canvas: test_extraction_repositories_v1
# Path: tests/features/frame_extraction/data/test_repositories.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as core_db_exc
from app.features.frame_extraction.data.repositories import FrameExtractionRepository
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame
from app.features.frame_extraction.exceptions.database import ExtractionPersistenceError


class TestFrameExtractionRepository:
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
        return FrameExtractionRepository(mock_db_manager)

    def test_save_happy_path(self, repository, mock_db_manager):
        """Verifies an extraction mapping is properly added to the DB session."""
        frame = ExtractedFrame(
            file_id=uuid.uuid4(),
            timestamp_sec=10.5,
            phash_value="abc123hash",
            quality_score=150.0
        )
        chunk_frames = ExtractedChunkFrames(
            source_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            frames=[frame]
        )

        result = repository.save(chunk_frames)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert added_model.id == chunk_frames.id
        assert added_model.chunk_id == chunk_frames.chunk_id
        # Verify serialization
        assert len(added_model.frames) == 1
        assert added_model.frames[0]["phash_value"] == "abc123hash"

        assert result == chunk_frames

    def test_save_db_error_is_mapped(self, repository, mock_db_manager):
        """Verifies DB errors are intercepted and wrapped."""
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.side_effect = core_db_exc.DatabaseError("DB connection failed")

        chunk_frames = ExtractedChunkFrames(source_id=uuid.uuid4(), chunk_id=uuid.uuid4(), frames=[])

        with pytest.raises(ExtractionPersistenceError, match="DB connection failed"):
            repository.save(chunk_frames)

    def test_get_by_chunk_id_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = uuid.uuid4()
        mock_model.chunk_id = uuid.uuid4()
        mock_model.frames = [{
            "frame_id": str(uuid.uuid4()),
            "file_id": str(uuid.uuid4()),
            "timestamp_sec": 12.0,
            "phash_value": "hashxyz",
            "quality_score": 120.0
        }]
        mock_model.created_at = datetime.now(timezone.utc)

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.get_by_chunk_id(mock_model.chunk_id)

        assert result is not None
        assert result.chunk_id == mock_model.chunk_id
        assert len(result.frames) == 1
        assert result.frames[0].phash_value == "hashxyz"