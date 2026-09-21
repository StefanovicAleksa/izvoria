# Path: tests/features/session_memory/service/test_session_memory_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.session_memory.service.session_memory_api import SessionMemoryAPI, strip_citation_tags
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.session_memory.exceptions.database import SessionMemorySystemError


def _turn(question: str, answer: str) -> SessionTurn:
    return SessionTurn(id=uuid.uuid4(), session_id=uuid.uuid4(), question=question, answer_text=answer,
                       created_at="2026-01-01T00:00:00Z")


class TestSessionMemoryAPI:
    @pytest.fixture
    def mock_repo(self): return MagicMock()

    @pytest.fixture
    def api(self, mock_repo):
        return SessionMemoryAPI(repository=mock_repo)

    @patch("app.features.session_memory.service.session_memory_api.settings")
    def test_get_history_is_capped_by_settings(self, mock_settings, api, mock_repo):
        mock_settings.CHAT_HISTORY_MAX_TURNS = 7
        session_id = uuid.uuid4()
        turns = [_turn("Q", "A")]
        mock_repo.get_history.return_value = turns

        assert api.get_history(session_id) == turns
        mock_repo.get_history.assert_called_once_with(session_id, 7)

    def test_get_history_wraps_repo_errors(self, api, mock_repo):
        mock_repo.get_history.side_effect = RuntimeError("DB down")

        with pytest.raises(SessionMemorySystemError):
            api.get_history(uuid.uuid4())

    def test_format_prior_context_none_without_turns(self):
        assert SessionMemoryAPI.format_prior_context([]) is None

    def test_format_prior_context_lists_every_turn_without_tags(self):
        turns = [_turn("What happened?", "The vote [Doc 1]."), _turn("And then?", "It passed [Doc 2, 3].")]

        assert SessionMemoryAPI.format_prior_context(turns) == (
            "[PRIOR TURN 1]\nQ: What happened?\nA: The vote.\n\n"
            "[PRIOR TURN 2]\nQ: And then?\nA: It passed."
        )

    def test_get_prior_context_formats_the_history(self, api, mock_repo):
        mock_repo.get_history.return_value = [_turn("Q?", "Vote was 9-0 [Doc 1].")]

        assert api.get_prior_context(uuid.uuid4()) == "[PRIOR TURN 1]\nQ: Q?\nA: Vote was 9-0."

    def test_get_prior_context_none_for_new_session(self, api, mock_repo):
        mock_repo.get_history.return_value = []

        assert api.get_prior_context(uuid.uuid4()) is None

    def test_record_chat_turn_keeps_conversation_tags(self, api, mock_repo):
        session_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        expected = _turn("Q", "A")
        mock_repo.record_turn.return_value = expected

        result = api.record_turn(session_id, "Q", "It rose [Doc 4].", standalone_query="Q standalone",
                                 evidence_chunk_ids=[chunk_id], first_doc_number=4)

        assert result == expected
        mock_repo.record_turn.assert_called_once_with(
            session_id, "Q", "It rose [Doc 4].",
            mode=SessionTurnMode.CHAT, standalone_query="Q standalone",
            evidence_chunk_ids=[chunk_id], first_doc_number=4
        )

    def test_record_deep_research_turn_strips_its_own_tags(self, api, mock_repo):
        session_id = uuid.uuid4()

        api.record_turn(session_id, "Q", "It rose [Doc 1] and then fell [Doc 2, 3].",
                        mode=SessionTurnMode.DEEP_RESEARCH)

        mock_repo.record_turn.assert_called_once_with(
            session_id, "Q", "It rose and then fell.",
            mode=SessionTurnMode.DEEP_RESEARCH, standalone_query=None,
            evidence_chunk_ids=[], first_doc_number=1
        )

    def test_record_turn_wraps_repo_errors(self, api, mock_repo):
        mock_repo.record_turn.side_effect = RuntimeError("DB down")

        with pytest.raises(SessionMemorySystemError):
            api.record_turn(uuid.uuid4(), "Q", "A")

    @pytest.mark.parametrize("text, expected", [
        ("Plain.", "Plain."),
        ("A [Doc 1].", "A."),
        ("A [doc 1, Doc 2] b [Docs 3-4]", "A b"),
    ])
    def test_strip_citation_tags(self, text, expected):
        assert strip_citation_tags(text) == expected
