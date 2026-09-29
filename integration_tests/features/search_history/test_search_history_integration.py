# Path: integration_tests/features/search_history/test_search_history_integration.py

import uuid

import pytest

from app.features.search_history.data.repositories import SearchHistoryRepository
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.search_history.service.search_history_api import SearchHistoryAPI


@pytest.mark.integration
class TestSearchHistoryIntegration:

    @pytest.fixture
    def api(self, db_manager) -> SearchHistoryAPI:
        """A REAL SearchHistoryAPI on the REAL test database (migrated to head by the session fixture)."""
        return SearchHistoryAPI(SearchHistoryRepository(db_manager))

    def test_a_search_round_trips_with_its_ranked_hits(self, api):
        chunk_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

        stored = api.record("notice period", SearchHistoryMode.VERIFIED, 5, chunk_ids, [0.91, 0.52, 0.10], 1234.5)
        loaded = api.get(stored.id)

        assert loaded == stored
        assert loaded.chunk_ids == chunk_ids  # rank order kept
        assert loaded.scores == [0.91, 0.52, 0.10]
        assert (loaded.mode, loaded.result_limit, loaded.execution_time_ms) == (SearchHistoryMode.VERIFIED, 5, 1234.5)

    def test_hits_are_not_foreign_keys(self, api):
        """Chunk ids that no longer exist (a re-ingested file) are stored and read back, not rejected."""
        gone = [uuid.uuid4()]
        stored = api.record("q", SearchHistoryMode.STANDARD, 5, gone, [0.4], 10)
        assert api.get(stored.id).chunk_ids == gone

    def test_an_empty_search_is_still_history(self, api):
        stored = api.record("nothing matches this", SearchHistoryMode.STANDARD, 5, [], [], 3)
        summary = next(s for s in api.list() if s.id == stored.id)
        assert summary.hit_count == 0

    def test_list_is_newest_first_with_hit_counts(self, api):
        older = api.record("older", SearchHistoryMode.STANDARD, 5, [uuid.uuid4()], [0.3], 1)
        newer = api.record("newer", SearchHistoryMode.VERIFIED, 5, [uuid.uuid4(), uuid.uuid4()], [0.9, 0.8], 1)

        summaries = [s for s in api.list() if s.id in (older.id, newer.id)]

        assert [(s.query, s.hit_count, s.mode) for s in summaries] == [
            ("newer", 2, SearchHistoryMode.VERIFIED), ("older", 1, SearchHistoryMode.STANDARD)]

    def test_delete_removes_only_that_search(self, api):
        keep = api.record("keep", SearchHistoryMode.STANDARD, 5, [], [], 1)
        drop = api.record("drop", SearchHistoryMode.STANDARD, 5, [], [], 1)

        assert api.delete(drop.id) is True
        assert api.delete(drop.id) is False
        assert api.get(drop.id) is None
        assert api.get(keep.id) is not None
