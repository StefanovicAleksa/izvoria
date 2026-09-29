# Path: tests/features/search_history/data/test_repositories.py

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock

import pytest

from app.core.database import exceptions as db_exc
from app.features.search_history.data.repositories import SearchHistoryRepository
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.search_history.exceptions import SearchHistoryDataError, SearchHistorySystemError


class TestSearchHistoryRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        ctx = MagicMock()
        db_manager.session.return_value = ctx
        ctx.__enter__.return_value = MagicMock()
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return SearchHistoryRepository(mock_db_manager)

    @pytest.fixture
    def session(self, mock_db_manager):
        return mock_db_manager.session.return_value.__enter__.return_value

    @staticmethod
    def _model(**overrides):
        model = MagicMock()
        model.id = uuid.uuid4()
        model.query = "termination clause"
        model.mode = "verified"
        model.result_limit = 5
        model.chunk_ids = [uuid.uuid4(), uuid.uuid4()]
        model.scores = [0.9, 0.4]
        model.execution_time_ms = 812.0
        model.created_at = datetime(2026, 9, 28, tzinfo=timezone.utc)
        for key, value in overrides.items():
            setattr(model, key, value)
        return model

    def test_record_persists_the_ranked_hits(self, repository, session):
        chunk_ids = [uuid.uuid4(), uuid.uuid4()]

        record = repository.record("termination clause", SearchHistoryMode.VERIFIED, 5, chunk_ids, [0.9, 0.4], 812)

        added = session.add.call_args[0][0]
        assert (added.query, added.mode, added.result_limit) == ("termination clause", "verified", 5)
        assert added.chunk_ids == chunk_ids
        assert added.scores == [0.9, 0.4]
        assert added.execution_time_ms == 812.0
        assert record.chunk_ids == chunk_ids and record.mode is SearchHistoryMode.VERIFIED

    def test_record_refuses_scores_that_do_not_line_up_with_the_chunks(self, repository, session):
        with pytest.raises(ValueError):
            repository.record("q", SearchHistoryMode.STANDARD, 5, [uuid.uuid4()], [], 1)
        session.add.assert_not_called()

    def test_record_maps_database_errors(self, repository, session):
        session.add.side_effect = db_exc.SystemError("DB Offline")
        with pytest.raises(SearchHistorySystemError):
            repository.record("q", SearchHistoryMode.STANDARD, 5, [], [], 1)

    def test_list_returns_summaries_with_hit_counts(self, repository, session):
        row = Mock(id=uuid.uuid4(), query="q", mode="standard", hit_count=3,
                   created_at=datetime(2026, 9, 28, tzinfo=timezone.utc))
        session.execute.return_value.all.return_value = [row]

        (summary,) = repository.list(limit=50)

        assert (summary.id, summary.query, summary.mode, summary.hit_count) == (
            row.id, "q", SearchHistoryMode.STANDARD, 3)
        assert "LIMIT" in str(session.execute.call_args[0][0].compile(compile_kwargs={"literal_binds": True}))

    def test_get_returns_the_full_record(self, repository, session):
        model = self._model()
        session.get.return_value = model

        record = repository.get(model.id)

        assert record.chunk_ids == model.chunk_ids and record.scores == [0.9, 0.4]

    def test_get_missing_is_none(self, repository, session):
        session.get.return_value = None
        assert repository.get(uuid.uuid4()) is None

    @pytest.mark.parametrize("rowcount, existed", [(1, True), (0, False)])
    def test_delete_says_whether_it_existed(self, repository, session, rowcount, existed):
        session.execute.return_value.rowcount = rowcount
        assert repository.delete(uuid.uuid4()) is existed

    def test_delete_maps_database_errors(self, repository, session):
        session.execute.side_effect = db_exc.DataError("bad uuid")
        with pytest.raises(SearchHistoryDataError):
            repository.delete(uuid.uuid4())
