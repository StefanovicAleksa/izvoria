# Canvas: test_retrieval_scrubber_error_mapper_v1
# Path: tests/features/retrieval_hallucination_scrubber/utils/test_error_mapper.py

import pytest
import json
import torch
from pydantic import ValidationError

from app.features.retrieval_hallucination_scrubber.utils.error_mapper import map_scrubber_error
from app.features.retrieval_hallucination_scrubber import exceptions as exc


class TestRetrievalScrubberErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), "eval", exc.SLMOOMError),
            (json.JSONDecodeError("Expecting value", "", 0), "eval", exc.JSONScrubParseError),
            (ValidationError.from_exception_data("Mock Error", []), "eval", exc.JSONScrubParseError),
            (Exception("llama_cpp context window crashed"), "eval", exc.SLMInferenceError),
            (Exception("Random cosmic ray"), "unknown", exc.RetrievalScrubberError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped_error = map_scrubber_error(input_exception, context=context_str)
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.JSONScrubParseError("Already mapped")
        mapped = map_scrubber_error(original)
        assert mapped is original