# Path: integration_tests/features/session_memory/test_session_memory_integration.py

import uuid
import pytest

from app.features.session_memory.data.repositories import SessionMemoryRepository
from app.features.session_memory.domain.enums import SessionTurnMode
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
        assert api.get_history(uuid.uuid4()) == []

    def test_history_round_trips_every_field_in_order(self, api: SessionMemoryAPI):
        session_id = uuid.uuid4()
        first_ids = [uuid.uuid4(), uuid.uuid4()]
        second_ids = [uuid.uuid4()]

        api.record_turn(session_id, "What happened at 9am?", "The defendant entered [Doc 1].",
                        standalone_query="What happened at 9am?", evidence_chunk_ids=first_ids)
        api.record_turn(session_id, "And then?", "He left [Doc 2][Doc 3].", mode=SessionTurnMode.REASONING,
                        standalone_query="What did the defendant do after 9am?",
                        evidence_chunk_ids=second_ids, first_doc_number=3)

        history = api.get_history(session_id)

        assert [t.question for t in history] == ["What happened at 9am?", "And then?"]
        assert history[0].mode == SessionTurnMode.CHAT
        assert history[0].answer_text == "The defendant entered [Doc 1]."
        assert history[0].evidence_chunk_ids == first_ids
        assert history[0].next_doc_number == 3
        assert history[1].mode == SessionTurnMode.REASONING
        assert history[1].answer_text == "He left."
        assert history[1].standalone_query == "What did the defendant do after 9am?"
        assert history[1].evidence_chunk_ids == second_ids
        assert history[1].next_doc_number == 4

    def test_prior_context_holds_the_whole_conversation(self, api: SessionMemoryAPI):
        session_id = uuid.uuid4()

        api.record_turn(session_id, "First question", "First answer [Doc 1]")
        api.record_turn(session_id, "Second question", "Second answer")

        context = api.get_prior_context(session_id)
        assert context == ("[PRIOR TURN 1]\nQ: First question\nA: First answer\n\n"
                           "[PRIOR TURN 2]\nQ: Second question\nA: Second answer")

    def test_sessions_are_isolated_from_each_other(self, api: SessionMemoryAPI):
        session_a = uuid.uuid4()
        session_b = uuid.uuid4()

        api.record_turn(session_a, "Question for A", "Answer for A")

        assert api.get_prior_context(session_b) is None
