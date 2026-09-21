# Canvas: desktop_test_session_store_v1
# Path: frontend/tests/data/test_session_store_unit.py

import json
import uuid

from onyx_desktop.data.session_store import SessionStore

SESSION_A = uuid.UUID("44444444-4444-4444-4444-444444444444")
SESSION_B = uuid.UUID("55555555-5555-5555-5555-555555555555")


class TestSessionStore:
    def test_a_first_question_becomes_the_conversation_title(self, session_store):
        ref = session_store.remember_turn(SESSION_A, "What did the auditor conclude?")

        assert ref.title == "What did the auditor conclude?"
        assert ref.turn_count == 1

    def test_a_long_question_is_shortened_for_the_sidebar(self, session_store):
        ref = session_store.remember_turn(SESSION_A, "word " * 50)

        assert len(ref.title) <= 60
        assert ref.title.endswith("…")

    def test_later_turns_bump_the_count_and_keep_the_original_title(self, session_store):
        session_store.remember_turn(SESSION_A, "First question")
        ref = session_store.remember_turn(SESSION_A, "A follow-up")

        assert ref.title == "First question"
        assert ref.turn_count == 2
        assert len(session_store.list_sessions()) == 1

    def test_sessions_are_listed_newest_first(self, session_store):
        session_store.remember_turn(SESSION_A, "older")
        session_store.remember_turn(SESSION_B, "newer")

        titles = [s.title for s in session_store.list_sessions()]
        assert titles[0] == "newer"

    def test_forget_removes_only_that_session(self, session_store):
        session_store.remember_turn(SESSION_A, "keep me")
        session_store.remember_turn(SESSION_B, "drop me")

        session_store.forget(SESSION_B)

        remaining = session_store.list_sessions()
        assert [s.session_id for s in remaining] == [SESSION_A]

    def test_a_missing_file_reads_as_no_history(self, tmp_path):
        store = SessionStore(tmp_path / "nope" / "sessions.json")
        assert store.list_sessions() == []

    def test_a_corrupt_file_degrades_to_no_history_instead_of_crashing(self, tmp_path):
        path = tmp_path / "sessions.json"
        path.write_text("{not json at all", encoding="utf-8")

        assert SessionStore(path).list_sessions() == []

    def test_a_malformed_record_is_skipped_but_good_ones_survive(self, tmp_path):
        path = tmp_path / "sessions.json"
        path.write_text(json.dumps([
            {"session_id": "not-a-uuid", "title": "bad", "started_at": "2026-01-01T00:00:00"},
            {"session_id": str(SESSION_A), "title": "good", "started_at": "2026-01-01T00:00:00"},
        ]), encoding="utf-8")

        sessions = SessionStore(path).list_sessions()
        assert [s.title for s in sessions] == ["good"]

    def test_the_write_is_atomic_and_leaves_no_temp_files(self, session_store):
        session_store.remember_turn(SESSION_A, "question")

        leftovers = list(session_store.path.parent.glob(".sessions-*"))
        assert leftovers == []
        assert session_store.path.exists()

    def test_the_store_is_created_on_demand(self, tmp_path):
        store = SessionStore(tmp_path / "deep" / "nested" / "sessions.json")
        store.remember_turn(SESSION_A, "question")

        assert store.path.exists()
