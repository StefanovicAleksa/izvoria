# Path: integration_tests/features/session_memory/test_session_memory_integration.py

import uuid
import pytest

from app.features.session_memory.data.repositories import SessionMemoryRepository
from app.features.session_memory.service.session_memory_api import SessionMemoryAPI


@pytest.mark.integration
class TestSessionMemoryIntegration:

    @pytest.fixture
    def api(self, db_manager) -> SessionMemoryAPI:
        """Creates a REAL SessionMemoryAPI connected to the REAL Test Database."""
        repo = SessionMemoryRepository(db_manager)
        return SessionMemoryAPI(repo)

    def test_get_prior_context_none_for_new_session(self, api: SessionMemoryAPI):
        assert api.get_prior_context(uuid.uuid4()) is None

    def test_record_and_recall_last_turn(self, api: SessionMemoryAPI):
        session_id = uuid.uuid4()

        api.record_turn(session_id, "What happened at 9am?", "The defendant entered the bank.")

        context = api.get_prior_context(session_id)
        assert context is not None
        assert "What happened at 9am?" in context
        assert "The defendant entered the bank." in context

    def test_only_the_most_recent_turn_is_recalled(self, api: SessionMemoryAPI):
        session_id = uuid.uuid4()

        api.record_turn(session_id, "First question", "First answer")
        api.record_turn(session_id, "Second question", "Second answer")

        context = api.get_prior_context(session_id)
        assert "Second question" in context
        assert "First question" not in context

    def test_sessions_are_isolated_from_each_other(self, api: SessionMemoryAPI):
        session_a = uuid.uuid4()
        session_b = uuid.uuid4()

        api.record_turn(session_a, "Question for A", "Answer for A")

        assert api.get_prior_context(session_b) is None
