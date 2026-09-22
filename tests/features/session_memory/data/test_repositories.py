# Path: tests/features/session_memory/data/test_repositories.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.session_memory.data.repositories import SessionMemoryRepository
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.session_memory.exceptions.database import SessionMemorySystemError


class TestSessionMemoryRepository:
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
        return SessionMemoryRepository(mock_db_manager)

    @pytest.fixture
    def session(self, mock_db_manager):
        return mock_db_manager.session.return_value.__enter__.return_value

    @staticmethod
    def _model(session_id: uuid.UUID, question: str = "What happened?", first_doc_number: int = 1):
        model = MagicMock()
        model.id = uuid.uuid4()
        model.session_id = session_id
        model.question = question
        model.answer_text = "The defendant [Doc 1]."
        model.mode = "chat"
        model.standalone_query = None
        model.evidence_chunk_ids = [uuid.uuid4()]
        model.first_doc_number = first_doc_number
        model.created_at = datetime.now(timezone.utc)
        return model

    def test_record_turn_persists_every_field(self, repository, session):
        session_id = uuid.uuid4()
        chunk_ids = [uuid.uuid4(), uuid.uuid4()]

        result = repository.record_turn(
            session_id, "What happened?", "The defendant [Doc 3].",
            mode=SessionTurnMode.REASONING, standalone_query="What did the defendant do?",
            evidence_chunk_ids=chunk_ids, first_doc_number=3
        )

        added = session.add.call_args[0][0]
        assert added.mode == "reasoning"
        assert added.evidence_chunk_ids == chunk_ids
        assert added.first_doc_number == 3
        assert result.session_id == session_id
        assert result.answer_text == "The defendant [Doc 3]."
        assert result.mode == SessionTurnMode.REASONING
        assert result.standalone_query == "What did the defendant do?"
        assert result.evidence_chunk_ids == chunk_ids
        assert result.next_doc_number == 5

    def test_record_turn_defaults_to_a_chat_turn_without_evidence(self, repository, session):
        result = repository.record_turn(uuid.uuid4(), "Q", "A")

        added = session.add.call_args[0][0]
        assert added.mode == "chat"
        assert added.evidence_chunk_ids == []
        assert added.first_doc_number == 1
        assert result.next_doc_number == 1

    def test_record_turn_db_error(self, repository, session):
        session.add.side_effect = db_exc.SystemError("DB Offline")

        with pytest.raises(SessionMemorySystemError):
            repository.record_turn(uuid.uuid4(), "Q", "A")

    def test_get_history_returns_newest_turns_oldest_first(self, repository, session):
        session_id = uuid.uuid4()
        newest, older = self._model(session_id, "Second", 2), self._model(session_id, "First", 1)
        session.execute.return_value.scalars.return_value.all.return_value = [newest, older]

        result = repository.get_history(session_id, limit=2)

        assert [t.question for t in result] == ["First", "Second"]
        stmt = session.execute.call_args[0][0]
        assert stmt._limit_clause.value == 2

    def test_get_history_empty(self, repository, session):
        session.execute.return_value.scalars.return_value.all.return_value = []

        assert repository.get_history(uuid.uuid4(), limit=20) == []

    def test_get_history_reads_a_pre_rename_mode_as_chat_instead_of_raising(self, repository, session):
        """
        'deep_research' rows predate the rename to 'reasoning'. SESSION_TURNS_UPGRADE_DDL backfills them on
        boot, but a database reached without start() must still yield its turns: losing which entrypoint
        answered a turn is survivable, losing the turn is not.
        """
        session_id = uuid.uuid4()
        stale = self._model(session_id, "What did they say?")
        stale.mode = "deep_research"
        session.execute.return_value.scalars.return_value.all.return_value = [stale]

        result = repository.get_history(session_id, limit=20)

        assert len(result) == 1
        assert result[0].mode == SessionTurnMode.CHAT
        assert result[0].question == "What did they say?"
        assert result[0].answer_text == stale.answer_text
        assert result[0].evidence_chunk_ids == stale.evidence_chunk_ids

    def test_get_last_turn_found(self, repository, session):
        session_id = uuid.uuid4()
        session.execute.return_value.scalars.return_value.all.return_value = [self._model(session_id)]

        result = repository.get_last_turn(session_id)

        assert result is not None
        assert result.session_id == session_id

    def test_get_last_turn_none(self, repository, session):
        session.execute.return_value.scalars.return_value.all.return_value = []

        assert repository.get_last_turn(uuid.uuid4()) is None

    def test_get_history_db_error(self, repository, session):
        session.execute.side_effect = db_exc.ConnectionError("Unreachable")

        with pytest.raises(SessionMemorySystemError):
            repository.get_history(uuid.uuid4(), limit=5)
