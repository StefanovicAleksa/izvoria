# Canvas: test_rag_retrieval_error_mapper_v2
# Path: tests/features/rag_retrieval/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError

from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.rag_retrieval.utils.error_mapper import map_retrieval_error
from app.features.rag_retrieval import exceptions as exc

class TestRAGRetrievalErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (ProgrammingError("syntax error in tsquery", {}, None), exc.InvalidSearchSyntaxError),
            (SQLAlchemyError("generic db error"), exc.SearchExecutionError),
            (ModelManagerError("VRAM full"), exc.PromptEmbeddingError),
            (RuntimeError("token indices sequence length is longer than context"), exc.ContextLengthExceededError),
            (exc.InvalidSearchSyntaxError("Already mapped"), exc.InvalidSearchSyntaxError), # Pass-through
            (ValueError("Random generic error"), exc.RetrievalError),  # Fallback wraps generic errors
        ]
    )
    def test_map_retrieval_error_translates_correctly(self, input_exception, expected_output_type):
        mapped_error = map_retrieval_error(input_exception)
        assert isinstance(mapped_error, expected_output_type)