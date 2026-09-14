# Path: tests/features/rag_reranking/utils/test_error_mapper.py

import pytest

from app.core.model_manager.exceptions import OutOfVramError
from app.features.rag_reranking import exceptions as exc
from app.features.rag_reranking.utils.error_mapper import map_rerank_error


class TestRerankErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_type",
        [
            (FileNotFoundError("missing weights"), exc.RerankerModelLoadError),
            (OutOfVramError("Insufficient VRAM"), exc.RerankerModelLoadError),
            (RuntimeError("CUDA out of memory"), exc.RerankInferenceError),
            (RuntimeError("tensor size mismatch"), exc.RerankInferenceError),
            (ValueError("something else"), exc.RerankingError),
        ]
    )
    def test_maps_to_typed_errors(self, input_exception, expected_type):
        mapped = map_rerank_error(input_exception, context="ctx")
        assert isinstance(mapped, expected_type)
        assert mapped.original_exception is input_exception

    def test_passes_feature_errors_through(self):
        original = exc.RerankInferenceError("already typed")
        assert map_rerank_error(original) is original
