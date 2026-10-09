# Path: integration_tests/features/session_memory/test_session_memory_integration.py

import uuid
import pytest

from app.core.config import settings
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


@pytest.mark.integration
class TestSessionMemoryHistoryIntegration:
    """Listing, re-reading and deleting whole conversations, as the desktop app's history pane does."""

    @pytest.fixture
    def api(self, db_manager) -> SessionMemoryAPI:
        return SessionMemoryAPI(SessionMemoryRepository(db_manager))

    def test_list_sessions_titles_by_first_question_and_orders_by_last_activity(self, api):
        quiet, busy = uuid.uuid4(), uuid.uuid4()
        api.record_turn(quiet, "Who signed the lease?", "The tenant [Doc 1].")
        api.record_turn(busy, "When was the hearing?", "In March [Doc 1].")
        api.record_turn(busy, "Who attended?", "Both parties [Doc 2].")

        listed = [s for s in api.list_sessions() if s.session_id in (quiet, busy)]

        assert [(s.session_id, s.title, s.turn_count) for s in listed] == [
            (busy, "When was the hearing?", 2), (quiet, "Who signed the lease?", 1)]
        assert listed[0].started_at <= listed[0].last_at

    def test_get_all_turns_is_the_whole_conversation_beyond_the_prompt_window(self, api):
        session_id = uuid.uuid4()
        count = settings.CHAT_HISTORY_MAX_TURNS + 5  # more than the model's prompt window keeps
        for i in range(count):
            api.record_turn(session_id, f"Q{i}", f"A{i}")

        turns = api.get_all_turns(session_id)

        assert [t.question for t in turns] == [f"Q{i}" for i in range(count)]
        assert len(api.get_history(session_id)) == settings.CHAT_HISTORY_MAX_TURNS

    def test_delete_session_removes_only_that_conversation(self, api):
        keep, drop = uuid.uuid4(), uuid.uuid4()
        api.record_turn(keep, "Keep?", "Yes")
        api.record_turn(drop, "Drop?", "Yes")
        api.record_turn(drop, "Really?", "Yes")

        assert api.delete_session(drop) == 2
        assert api.get_all_turns(drop) == []
        assert [t.question for t in api.get_all_turns(keep)] == ["Keep?"]

    def test_each_user_sees_only_their_own_turns(self, api: SessionMemoryAPI):
        """The hosted API: a conversation id guessed from another user reads as empty and lists as nothing."""
        session_id = uuid.uuid4()
        api.record_turn(session_id, "alice asks", "answer", owner_id="alice")

        assert api.get_history(session_id, owner_id="bob") == []
        assert api.get_all_turns(session_id, owner_id="bob") == []
        assert session_id not in {s.session_id for s in api.list_sessions(owner_id="bob")}
        assert api.delete_session(session_id, owner_id="bob") == 0
        assert [t.question for t in api.get_history(session_id, owner_id="alice")] == ["alice asks"]
        assert session_id in {s.session_id for s in api.list_sessions(owner_id="alice")}
