# Path: tests/features/source_profile/utils/test_error_mapper.py

import json

import pytest

from app.core.database import exceptions as db_exc
from app.core.model_manager.exceptions import ModelManagerError
from app.features.source_profile import exceptions as exc
from app.features.source_profile.utils.error_mapper import map_source_profile_error


class TestSourceProfileErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (json.JSONDecodeError("Expecting value", "x", 0), exc.ProfileParseError),
            (db_exc.DataError("Bad data"), exc.SourceProfileDataError),
            (db_exc.DBForeignKeyError("No such source"), exc.SourceProfileDataError),
            (ValueError("empty title"), exc.SourceProfileDataError),
            (db_exc.DBUnreachableError("Network down"), exc.SourceProfileSystemError),
            (db_exc.SystemError("Disk full"), exc.SourceProfileSystemError),
            (RuntimeError("llama_decode returned -1"), exc.ProfileGenerationError),
            (RuntimeError("Unexpected cosmic ray"), exc.SourceProfileError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        mapped = map_source_profile_error(input_exception, context="ctx")
        assert type(mapped) is expected_output_type
        assert mapped.original_exception is input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.SourceProfileDataError("Already mapped")
        assert map_source_profile_error(original) is original

    def test_model_manager_errors_pass_through_unchanged(self):
        original = ModelManagerError("out of VRAM")
        assert map_source_profile_error(original) is original
