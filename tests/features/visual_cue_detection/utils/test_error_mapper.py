# Canvas: test_visual_cue_error_mapper_v1
# Path: tests/features/visual_cue_detection/utils/test_error_mapper.py

import json
import pytest

from app.features.visual_cue_detection.utils.error_mapper import map_detection_error
from app.features.visual_cue_detection import exceptions as exc


class TestVisualCueErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (json.JSONDecodeError("Expecting value", "", 0), "parsing", exc.CueParseError),
            (ValueError("Invalid json format"), "parsing", exc.CueParseError),
            (RuntimeError("CUDA out of memory"), "inference", exc.LLMCueGenerationError),
            (Exception("llama_cpp crashed"), "inference", exc.LLMCueGenerationError),
            (Exception("Random cosmic ray bit flip"), "unknown", exc.VisualCueDetectionError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped = map_detection_error(input_exception, context=context_str)

        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.CueParseError("Already mapped")
        mapped = map_detection_error(original)
        assert mapped is original