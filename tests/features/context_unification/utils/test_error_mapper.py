# Canvas: test_context_unification_error_mapper_v1
# Path: tests/features/context_unification/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.context_unification.utils.error_mapper import map_unification_error
from app.features.context_unification import exceptions as exc


class TestContextUnificationErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (SQLAlchemyError("DB offline"), "save unified transcript", exc.UnifiedTranscriptPersistenceError),
            (core_db_exc.DatabaseError("General DB"), "unknown", exc.ContextUnificationDatabaseError),
            (ValidationError.from_exception_data("Validation failed", line_errors=[]), "assembly", exc.ContextUnificationPipelineError),
            (Exception("Cosmic ray"), "unknown", exc.ContextUnificationError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped = map_unification_error(input_exception, context=context_str)
        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.TranscriptMismatchError("Already mapped")
        mapped = map_unification_error(original)
        assert mapped is original