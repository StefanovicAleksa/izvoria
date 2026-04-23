# Canvas: test_rag_error_mapper_v1
# Path: tests/features/rag_storage/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.features.rag_storage.utils.error_mapper import map_storage_error
from app.features.rag_storage import exceptions as exc

class TestRAGStorageErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (SQLAlchemyError("vector dimension mismatch"), exc.VectorDatabaseError),
            (SQLAlchemyError("generic db error"), exc.VectorDatabaseError),
            (RuntimeError("CUDA out of memory"), exc.EmbeddingInferenceError),
            (RuntimeError("Embedding model not loaded"), exc.EmbeddingModelNotLoadedError),
            (PermissionError("Cannot write to disk"), exc.SparseIndexError),
            (ValueError("Random application error"), ValueError),  # Fallback returns original
        ]
    )
    def test_map_storage_error_translates_correctly(self, input_exception, expected_output_type):
        mapped_error = map_storage_error(input_exception)
        assert isinstance(mapped_error, expected_output_type)