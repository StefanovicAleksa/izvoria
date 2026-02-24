# filename: test_transcription_repository_v2.py
# path: tests/features/transcription/test_repository.py
import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.transcription.domain.entities import Transcript, TranscriptSegment
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
        return Transcript(
            source_id=uuid.uuid4(),
            segments=[TranscriptSegment(start_sec=1.5, end_sec=3.0, text="Hello world")]
        )

    def test_save_transcript_success(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        repo = TranscriptRepository(manager)

        result = repo.save(sample_transcript)

        assert result == sample_transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, TranscriptModel)
        assert added_model.source_id == sample_transcript.source_id
        assert added_model.segments[0]["text"] == "Hello world"

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
        assert result.segments[0].text == "Hello world"

    def test_get_by_source_id_not_found(self, mock_db_manager):
        manager, session = mock_db_manager
        session.execute.return_value.scalar_one_or_none.return_value = None

        repo = TranscriptRepository(manager)
        result = repo.get_by_source_id(uuid.uuid4())

        assert result is None

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = TranscriptRepository(manager)

        with pytest.raises(TranscriptionDatabaseError):
            repo.save(sample_transcript)