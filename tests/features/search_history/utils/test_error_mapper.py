# Path: tests/features/search_history/utils/test_error_mapper.py

import pytest

from app.core.database import exceptions as db_exc
from app.features.search_history import exceptions as exc
from app.features.search_history.utils.error_mapper import map_search_history_error


class TestSearchHistoryErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (db_exc.DBNotNullError("Missing field"), exc.SearchHistoryDataError),
            (db_exc.DataError("Bad data"), exc.SearchHistoryDataError),
            (ValueError("scores do not line up"), exc.SearchHistoryDataError),
            (db_exc.DBUnreachableError("Network down"), exc.SearchHistorySystemError),
            (db_exc.ConnectionError("Refused"), exc.SearchHistorySystemError),
            (db_exc.SystemError("Disk full"), exc.SearchHistorySystemError),
            (RuntimeError("Unexpected cosmic ray"), exc.SearchHistorySystemError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        mapped = map_search_history_error(input_exception)
        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception is input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.SearchHistoryDataError("Already mapped")
        assert map_search_history_error(original) is original
