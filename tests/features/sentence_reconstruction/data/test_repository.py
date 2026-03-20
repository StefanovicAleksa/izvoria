# Canvas: test_sentence_repository_v2
# Path: tests/features/sentence_reconstruction/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.features.sentence_reconstruction.data.models import ReconstructedTranscriptModel
from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository


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
        return ReconstructedTranscript(source_id=source_id, sentences=[])

    def test_save_success(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager
        repo = PostgresSentenceRepository(manager)

        result = repo.save(sample_transcript)

        assert result == sample_transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, ReconstructedTranscriptModel)
        assert added_model.source_id == sample_transcript.source_id

    def test_get_by_source_id_found(self, mock_db_manager, sample_transcript):
        manager, session = mock_db_manager

        mock_db_model = ReconstructedTranscriptModel(
            id=sample_transcript.id,
            source_id=sample_transcript.source_id,
            sentences=[],
            created_at=sample_transcript.created_at
        )
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        repo = PostgresSentenceRepository(manager)
        result = repo.get_by_source_id(sample_transcript.source_id)

        assert result is not None
        assert result.source_id == sample_transcript.source_id