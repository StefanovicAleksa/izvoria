# Path: tests/features/session_memory/utils/test_error_mapper.py

import pytest

from app.core.database import exceptions as db_exc
from app.features.session_memory.utils.error_mapper import map_session_memory_error
from app.features.session_memory import exceptions as exc


class TestSessionMemoryErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (db_exc.DBNotNullError("Missing field"), exc.SessionMemoryDataError),
            (db_exc.DataError("Bad data"), exc.SessionMemoryDataError),
            (db_exc.DBUnreachableError("Network down"), exc.SessionMemorySystemError),
            (db_exc.ConnectionError("Refused"), exc.SessionMemorySystemError),
            (RuntimeError("Unexpected cosmic ray"), exc.SessionMemorySystemError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        mapped_error = map_session_memory_error(input_exception)
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.SessionMemoryDataError("Already mapped")
        mapped = map_session_memory_error(original)
        assert mapped is original
