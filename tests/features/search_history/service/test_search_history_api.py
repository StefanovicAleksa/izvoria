# Path: tests/features/search_history/service/test_search_history_api.py

import uuid
from unittest.mock import Mock

import pytest

from app.core.database import exceptions as db_exc
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.search_history.exceptions import SearchHistoryDataError, SearchHistorySystemError
from app.features.search_history.service.search_history_api import DEFAULT_LIST_LIMIT, SearchHistoryAPI


class TestSearchHistoryAPI:
    @pytest.fixture
    def repo(self):
        return Mock()

    @pytest.fixture
    def api(self, repo):
        return SearchHistoryAPI(repo)

    def test_record_delegates_with_every_field(self, api, repo):
        ids = (uuid.uuid4(), uuid.uuid4())
        api.record("q", SearchHistoryMode.STANDARD, 5, ids, (0.5, 0.2), 120.0)
        repo.record.assert_called_once_with("q", SearchHistoryMode.STANDARD, 5, list(ids), [0.5, 0.2], 120.0,
                                            owner_id=None)

    def test_list_uses_the_default_limit(self, api, repo):
        api.list()
        repo.list.assert_called_once_with(DEFAULT_LIST_LIMIT, owner_id=None)

    def test_get_and_delete_delegate(self, api, repo):
        record_id = uuid.uuid4()
        api.get(record_id)
        api.delete(record_id)
        repo.get.assert_called_once_with(record_id, owner_id=None)
        repo.delete.assert_called_once_with(record_id, owner_id=None)

    @pytest.mark.parametrize("method, args", [
        ("list", ()), ("get", (uuid.uuid4(),)), ("delete", (uuid.uuid4(),)),
        ("record", ("q", SearchHistoryMode.STANDARD, 5, [], [], 1.0)),
    ])
    def test_errors_become_feature_errors(self, api, repo, method, args):
        getattr(repo, method).side_effect = db_exc.ConnectionError("down")
        with pytest.raises(SearchHistorySystemError):
            getattr(api, method)(*args)

    def test_a_malformed_record_is_a_data_error(self, api, repo):
        repo.record.side_effect = ValueError("A search record needs exactly one score per chunk.")
        with pytest.raises(SearchHistoryDataError):
            api.record("q", SearchHistoryMode.STANDARD, 5, [uuid.uuid4()], [], 1.0)


class TestSearchHistoryOwnership:
    def test_the_owner_reaches_every_repository_call(self):
        import uuid as _uuid
        from unittest.mock import MagicMock as _Mock
        repo = _Mock()
        api = SearchHistoryAPI(repo)
        record_id = _uuid.uuid4()

        api.record("q", SearchHistoryMode.VERIFIED, 3, [], [], 1.0, owner_id="u1")
        api.list(10, owner_id="u1")
        api.get(record_id, owner_id="u1")
        api.delete(record_id, owner_id="u1")

        repo.record.assert_called_once_with("q", SearchHistoryMode.VERIFIED, 3, [], [], 1.0, owner_id="u1")
        repo.list.assert_called_once_with(10, owner_id="u1")
        repo.get.assert_called_once_with(record_id, owner_id="u1")
        repo.delete.assert_called_once_with(record_id, owner_id="u1")
