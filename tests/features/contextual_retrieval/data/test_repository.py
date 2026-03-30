# Canvas: test_contextual_repository_v1
# Path: tests/features/contextual_retrieval/data/test_repository.py

import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy.exc import SQLAlchemyError

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.contextual_retrieval.data.models import ContextualizedTranscriptModel
from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.exceptions.database import ContextualPersistenceError


class TestContextualRepository:
    @pytest.fixture
    def mock_db_manager(self):
        manager = MagicMock()
        session_mock = MagicMock()
        manager.session.return_value.__enter__.return_value = session_mock
        return manager, session_mock

    def test_save_success(self, mock_db_manager):
        manager, session = mock_db_manager
        repo = ContextualRepository(manager)

        chunk = ContextualizedChunk(
            start_sec=0, end_sec=5, original_text="Hello", context_string="Context", sentences=[]
        )
        transcript = ContextualizedTranscript(source_id=uuid.uuid4(), chunks=[chunk])

        result = repo.save(transcript)

        assert result == transcript
        session.add.assert_called_once()
        added_model = session.add.call_args[0][0]
        assert isinstance(added_model, ContextualizedTranscriptModel)
        assert added_model.source_id == transcript.source_id
        assert len(added_model.chunks) == 1

    def test_db_exceptions_mapped(self, mock_db_manager):
        manager, session = mock_db_manager
        session.add.side_effect = SQLAlchemyError("DB Down")
        repo = ContextualRepository(manager)

        transcript = ContextualizedTranscript(source_id=uuid.uuid4(), chunks=[])

        with pytest.raises(ContextualPersistenceError):
            repo.save(transcript)