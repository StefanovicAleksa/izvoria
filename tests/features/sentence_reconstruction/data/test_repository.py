# Canvas: test_repository_v1
# Path: tests/features/sentence_reconstruction/data/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.sentence_reconstruction.data.models import ReconstructedTranscriptModel
from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
from app.features.sentence_reconstruction.exceptions.database import SentenceDatabaseError


class TestSentenceRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    @pytest.fixture
    def sample_transcript(self):
        source_id = uuid.uuid4()
        words = [
            TranscriptWord(start_sec=1.0, end_sec=1.5, word="Hello", probability=0.99),
            TranscriptWord(start_sec=1.5, end_sec=2.0, word="world", probability=0.98)
        ]
        sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(),
            text="Hello world.",
            speaker_id="Speaker_0",
            start_sec=1.0,
            end_sec=2.0,
            words=words
        )
        return ReconstructedTranscript(source_id=source_id, sentences=[sentence])

    def test_save_success(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        repo = PostgresSentenceRepository(manager)

        result = repo.save(sample_transcript)

        assert result == sample_transcript
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, ReconstructedTranscriptModel)
        assert added_model.source_id == sample_transcript.source_id

        # Verify nested JSON serialization
        assert len(added_model.sentences) == 1
        assert len(added_model.sentences[0]["words"]) == 2
        assert added_model.sentences[0]["words"][0]["word"] == "Hello"

    def test_get_by_source_id_found(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager

        # Simulate the SQLAlchemy Row returning a JSONB dictionary
        mock_db_model = ReconstructedTranscriptModel(
            id=sample_transcript.id,
            source_id=sample_transcript.source_id,
            sentences=[s.model_dump(mode='json') for s in sample_transcript.sentences],
            created_at=sample_transcript.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = PostgresSentenceRepository(manager)
        result = repo.get_by_source_id(sample_transcript.source_id)

        assert result is not None
        assert result.source_id == sample_transcript.source_id
        assert len(result.sentences) == 1

        # Prove the words array was successfully deserialized back into Pydantic Objects
        assert len(result.sentences[0].words) == 2
        assert isinstance(result.sentences[0].words[0], TranscriptWord)
        assert result.sentences[0].words[0].word == "Hello"

    def test_repository_maps_db_exceptions(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Connection Lost")

        repo = PostgresSentenceRepository(manager)

        with pytest.raises(SentenceDatabaseError):
            repo.save(sample_transcript)