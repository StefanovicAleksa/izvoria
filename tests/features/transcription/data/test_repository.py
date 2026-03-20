# Canvas: test_transcription_repo_v3
# Path: tests/features/transcription/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.transcription.data.models import TranscriptModel
from app.features.transcription.data.repositories import TranscriptRepository
from app.features.transcription.exceptions.database import TranscriptionDatabaseError


class TestTranscriptRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_transcript(self):
        # NEW: Inject word-level data to ensure JSONB handles nested arrays correctly
        words = [
            TranscriptWord(start_sec=1.0, end_sec=1.5, word="Test", probability=0.98),
            TranscriptWord(start_sec=1.5, end_sec=2.0, word="Segment", probability=0.95)
        ]
        return Transcript(
            source_id=uuid.uuid4(),
            segments=[TranscriptSegment(start_sec=1.0, end_sec=2.0, text="Test Segment", words=words)]
        )

    def test_save_success(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        repo = TranscriptRepository(manager)

        result = repo.save(sample_transcript)

        assert result == sample_transcript
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, TranscriptModel)
        assert added_model.source_id == sample_transcript.source_id

    def test_get_by_source_id_found(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager

        mock_db_model = TranscriptModel(
            id=sample_transcript.id,
            source_id=sample_transcript.source_id,
            segments=[seg.model_dump() for seg in sample_transcript.segments],
            created_at=sample_transcript.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = TranscriptRepository(manager)
        result = repo.get_by_source_id(sample_transcript.source_id)

        assert result is not None
        assert result.source_id == sample_transcript.source_id
        assert len(result.segments) == 1
        assert result.segments[0].text == "Test Segment"

        # Ensure the nested JSONB word array was perfectly restored to Pydantic objects
        assert len(result.segments[0].words) == 2
        assert result.segments[0].words[0].word == "Test"
        assert result.segments[0].words[0].probability == 0.98

    def test_update_success(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager

        # Mock finding the existing record
        mock_db_model = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = TranscriptRepository(manager)
        result = repo.update(sample_transcript)

        assert result == sample_transcript
        # Verify it mutated the existing SQLAlchemy object's JSONB segments in-place
        assert mock_db_model.segments == [seg.model_dump() for seg in sample_transcript.segments]

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = TranscriptRepository(manager)

        with pytest.raises(TranscriptionDatabaseError):
            repo.save(sample_transcript)