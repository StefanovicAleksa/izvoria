# Path: tests/features/session_memory/service/test_session_memory_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.session_memory.service.session_memory_api import SessionMemoryAPI
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.exceptions.database import SessionMemorySystemError


class TestSessionMemoryAPI:
    @pytest.fixture
    def mock_repo(self): return MagicMock()

    @pytest.fixture
    def api(self, mock_repo):
        return SessionMemoryAPI(repository=mock_repo)

    def test_get_prior_context_none_when_no_last_turn(self, api, mock_repo):
        mock_repo.get_last_turn.return_value = None

        assert api.get_prior_context(uuid.uuid4()) is None

    def test_get_prior_context_formats_last_turn(self, api, mock_repo):
        mock_repo.get_last_turn.return_value = SessionTurn(
            id=uuid.uuid4(), session_id=uuid.uuid4(), question="What happened?",
            answer_text="The defendant.", created_at="2026-01-01T00:00:00Z"
        )

        context = api.get_prior_context(uuid.uuid4())

        assert "What happened?" in context
        assert "The defendant." in context

    def test_get_prior_context_wraps_repo_errors(self, api, mock_repo):
        mock_repo.get_last_turn.side_effect = RuntimeError("DB down")

        with pytest.raises(SessionMemorySystemError):
            api.get_prior_context(uuid.uuid4())

    def test_record_turn_delegates_to_repository(self, api, mock_repo):
        session_id = uuid.uuid4()
        expected = SessionTurn(id=uuid.uuid4(), session_id=session_id, question="Q", answer_text="A",
                               created_at="2026-01-01T00:00:00Z")
        mock_repo.record_turn.return_value = expected

        result = api.record_turn(session_id, "Q", "A")

        assert result == expected
        mock_repo.record_turn.assert_called_once_with(session_id, "Q", "A")

    def test_record_turn_strips_citation_tags(self, api, mock_repo):
        session_id = uuid.uuid4()

        api.record_turn(session_id, "Q", "It rose [Doc 1] and then fell [Doc 2, 3].")

        mock_repo.record_turn.assert_called_once_with(session_id, "Q", "It rose and then fell.")

    def test_get_prior_context_strips_tags_from_previously_stored_answers(self, api, mock_repo):
        mock_repo.get_last_turn.return_value = SessionTurn(
            id=uuid.uuid4(), session_id=uuid.uuid4(), question="Q?",
            answer_text="Vote was 9-0 [Doc 1].", created_at="2026-01-01T00:00:00Z"
        )

        assert api.get_prior_context(uuid.uuid4()) == "[PRIOR TURN]\nQ: Q?\nA: Vote was 9-0."
