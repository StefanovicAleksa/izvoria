# Canvas: test_contextual_error_mapper_v1
# Path: tests/features/contextual_retrieval/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error
from app.features.contextual_retrieval import exceptions as exc

class TestContextualErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (SQLAlchemyError("DB down"), "transcript", exc.ContextualPersistenceError),
            (core_db_exc.DatabaseError("General DB"), "unknown", exc.ContextualDatabaseError),
            (FileNotFoundError("gguf missing"), "", exc.SLMModelLoadError),
            (Exception("exceeded maximum context window"), "", exc.ContextWindowExceededError),
            (MemoryError("out of memory"), "", exc.SLMInferenceError),
            (ValueError("index out of bounds"), "chunk processing", exc.TranscriptChunkingError),
            (Exception("random cosmic ray"), "", exc.ContextualRetrievalError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped = map_contextual_error(input_exception, context=context_str)
        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.SLMInferenceError("Already mapped")
        mapped = map_contextual_error(original)
        assert mapped is original