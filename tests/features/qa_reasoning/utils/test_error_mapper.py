# Path: tests/features/qa_reasoning/utils/test_error_mapper.py

import pytest
import json
import torch
from pydantic import ValidationError

from app.features.qa_reasoning.utils.error_mapper import map_reasoner_error
from app.features.qa_reasoning import exceptions as exc


class TestQAReasoningErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), "reason", exc.ReasonerOOMError),
            (json.JSONDecodeError("Expecting value", "", 0), "reason", exc.JSONReasonerParseError),
            (ValidationError.from_exception_data("Mock Error", []), "reason", exc.JSONReasonerParseError),
            (Exception("llama_cpp context window crashed"), "reason", exc.ReasonerInferenceError),
            (Exception("Random cosmic ray"), "unknown", exc.QAReasoningError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped_error = map_reasoner_error(input_exception, context=context_str)
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.JSONReasonerParseError("Already mapped")
        mapped = map_reasoner_error(original)
        assert mapped is original
