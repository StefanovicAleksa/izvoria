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
            (ValueError("Random application error"), exc.RAGStorageError),  # Fallback is always a feature error
        ]
    )
    def test_map_storage_error_translates_correctly(self, input_exception, expected_output_type):
        mapped_error = map_storage_error(input_exception)
        assert isinstance(mapped_error, expected_output_type)

    def test_map_storage_error_passes_feature_errors_through(self):
        original = exc.VectorDatabaseError("already mapped")
        assert map_storage_error(original) is original

    def test_map_storage_error_keeps_original_exception(self):
        raw = SQLAlchemyError("generic db error")
        mapped = map_storage_error(raw, context="Saving chunks")
        assert mapped.original_exception is raw
        assert "Saving chunks" in str(mapped)
