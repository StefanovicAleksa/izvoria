# Canvas: test_contextual_repositories_v2
# Path: tests/features/contextual_retrieval/data/test_repositories.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.contextual_retrieval.exceptions.database import ContextualDatabaseError


class TestContextualRepository:
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
        return ContextualRepository(mock_db_manager)

    # ----------------------------------------------------------------------
    # EXISTING TESTS (Save & Get)
    # ----------------------------------------------------------------------
    def test_save_happy_path(self, repository, mock_db_manager):
        chunk = ContextualizedChunk(original_text="Test", context_string="Context")
        transcript = ContextualizedTranscript(source_id=uuid.uuid4(), chunks=[chunk])

        result = repository.save(transcript)

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()

        added_model = session.add.call_args[0][0]
        assert added_model.id == transcript.id
        assert len(added_model.chunks) == 1
        assert added_model.chunks[0]["original_text"] == "Test"
        assert result == transcript

    def test_get_by_source_id_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value

        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.source_id = uuid.uuid4()
        mock_model.chunks = [{
            "chunk_id": str(uuid.uuid4()),
            "start_sec": 0.0,
            "end_sec": 10.0,
            "original_text": "Test",
            "context_string": "Context",
            "needs_visual": False,
            "visual_descriptions": [],
            "sentences": []
        }]
        mock_model.created_at = datetime.now(timezone.utc)

        session.execute.return_value.scalar_one_or_none.return_value = mock_model

        result = repository.get_by_source_id(mock_model.source_id)

        assert result is not None
        assert result.source_id == mock_model.source_id
        assert len(result.chunks) == 1
        assert result.chunks[0].needs_visual is False

    # ----------------------------------------------------------------------
    # NEW TESTS (Update)
    # ----------------------------------------------------------------------
    def test_update_happy_path_modifies_jsonb(self, repository, mock_db_manager):
        """Verifies that calling update correctly serializes the modified chunks array."""
        session = mock_db_manager.session.return_value.__enter__.return_value

        # 1. Mock the existing database record being found
        mock_db_model = MagicMock()
        mock_db_model.id = uuid.uuid4()
        mock_db_model.chunks = []  # Initially empty or different
        session.execute.return_value.scalar_one_or_none.return_value = mock_db_model

        # 2. Create the Domain Entity with the flipped NLP flag
        updated_chunk = ContextualizedChunk(
            needs_visual=True,
            original_text="Look at this X-Ray."
        )
        updated_transcript = ContextualizedTranscript(
            id=mock_db_model.id,
            chunks=[updated_chunk]
        )

        # 3. Perform the update
        result = repository.update(updated_transcript)

        # 4. Assertions
        assert result == updated_transcript

        # Verify the SQLAlchemy model's JSONB column was dynamically overwritten
        assert len(mock_db_model.chunks) == 1
        assert mock_db_model.chunks[0]["needs_visual"] is True
        assert mock_db_model.chunks[0]["original_text"] == "Look at this X-Ray."

    def test_update_not_found_raises_mapped_error(self, repository, mock_db_manager):
        """Verifies that if the transcript ID doesn't exist, it is safely caught and mapped."""
        session = mock_db_manager.session.return_value.__enter__.return_value

        # Mock the record NOT being found in the database
        session.execute.return_value.scalar_one_or_none.return_value = None

        transcript = ContextualizedTranscript()

        # The internal ValueError("Transcript not found") should be caught and mapped
        with pytest.raises(ContextualDatabaseError):
            repository.update(transcript)