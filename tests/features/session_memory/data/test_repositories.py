# Path: tests/features/session_memory/data/test_repositories.py

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from app.core.database import exceptions as db_exc
from app.features.session_memory.data.repositories import SessionMemoryRepository
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

    def _clean_mock_model(self, session_id: uuid.UUID):
        mock_model = MagicMock()
        mock_model.id = uuid.uuid4()
        mock_model.session_id = session_id
        mock_model.question = "What happened?"
        mock_model.answer_text = "The defendant."
        mock_model.created_at = datetime.now(timezone.utc)
        return mock_model

    def test_record_turn_success(self, repository, mock_db_manager):
        session_id = uuid.uuid4()
        result = repository.record_turn(session_id, "What happened?", "The defendant.")

        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.assert_called_once()
        assert result.session_id == session_id
        assert result.question == "What happened?"
        assert result.answer_text == "The defendant."

    def test_record_turn_db_error(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.add.side_effect = db_exc.SystemError("DB Offline")

        with pytest.raises(SessionMemorySystemError):
            repository.record_turn(uuid.uuid4(), "Q", "A")

    def test_get_last_turn_found(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session_id = uuid.uuid4()
        mock_model = self._clean_mock_model(session_id)
        session.execute.return_value.scalars.return_value.first.return_value = mock_model

        result = repository.get_last_turn(session_id)

        assert result is not None
        assert result.session_id == session_id

    def test_get_last_turn_none(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalars.return_value.first.return_value = None

        result = repository.get_last_turn(uuid.uuid4())

        assert result is None

    def test_get_last_turn_db_error(self, repository, mock_db_manager):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.side_effect = db_exc.ConnectionError("Unreachable")

        with pytest.raises(SessionMemorySystemError):
            repository.get_last_turn(uuid.uuid4())
