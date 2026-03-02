# Canvas: test_nlp_error_mapper_v1
# Path: tests/features/nlp_router/test_error_mapper.py

import pytest
import json

from app.features.nlp_router.utils.error_mapper import map_nlp_router_error
from app.features.nlp_router import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, expected_output_type",
    [
        (FileNotFoundError("Model not found"), exc.NLPRouterModelLoadError),
        (json.JSONDecodeError("Expecting value", "", 0), exc.NLPRouterJSONParseError),
        (ValueError("Requested tokens exceed context window"), exc.NLPRouterContextLimitError),
        (RuntimeError("llama.cpp crashed during generation"), exc.NLPRouterInferenceError),
        (Exception("Random unexpected cosmic ray"), exc.NLPRouterError),
    ]
)
def test_map_nlp_router_error_translates_correctly(input_exception, expected_output_type):
    # Act
    mapped_error = map_nlp_router_error(input_exception, context="test_context")

    # Assert
    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception
    assert "test_context" in str(mapped_error)