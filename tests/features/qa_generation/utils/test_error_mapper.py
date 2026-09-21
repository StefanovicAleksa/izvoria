# Canvas: test_error_mapper_v3
# Path: tests/features/qa_generation/utils/test_error_mapper.py

import json
import pytest
from pydantic import ValidationError

from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.qa_generation.utils.error_mapper import map_qa_error
from app.features.qa_generation import exceptions as exc


class TestQAGenerationErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            # 0. Self-Critique JSON hallucinations
            (json.JSONDecodeError("Expecting value", "", 0), exc.JSONCritiqueParseError),
            (ValidationError.from_exception_data("Mock Error", []), exc.JSONCritiqueParseError),

            # 1. Hardware/VRAM limits
            (RuntimeError("CUDA out of memory"), exc.GenerationOOMError),
            (MemoryError("Out of memory"), exc.GenerationOOMError),

            # 2. Context Window Overflows
            (ValueError("token indices sequence length is longer than context"), exc.ContextWindowExceededError),
            (RuntimeError("n_ctx limit reached"), exc.ContextWindowExceededError),

            # 3. Inference Engine Crashes
            (Exception("llama_cpp generation failed"), exc.LLMInferenceCrashError),
            (RuntimeError("segmentation fault in engine"), exc.LLMInferenceCrashError),

            # 4. Core Infrastructure Failures
            (ModelManagerError("Weights missing"), exc.GenerationModelError),

            # 5. Pass-through (already mapped)
            (exc.GenerationOOMError("Already mapped"), exc.GenerationOOMError),

            # 6. Fallback (Generic wrap)
            (Exception("Unexpected cosmic ray"), exc.QAGenerationError),
        ]
    )
    def test_map_qa_error_translates_correctly(self, input_exception, expected_output_type):
        """Verifies that the error mapper correctly buckets strings and types into QA domain exceptions."""
        mapped_error = map_qa_error(input_exception, context="unit-test-context")

        assert isinstance(mapped_error, expected_output_type)

        # Verify original exception is linked, UNLESS it's an identity pass-through
        if mapped_error is not input_exception and hasattr(mapped_error, "original_exception"):
            assert mapped_error.original_exception == input_exception

    def test_map_qa_error_prevents_double_wrapping(self):
        """Ensures that if an exception is already a QAGenerationError, it is returned as-is."""
        original = exc.ContextWindowExceededError("Context limit reached")
        mapped = map_qa_error(original)

        assert mapped is original