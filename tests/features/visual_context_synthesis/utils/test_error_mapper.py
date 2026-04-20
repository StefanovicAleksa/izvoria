# Canvas: test_visual_synthesis_error_mapper_v1
# Path: tests/features/visual_context_synthesis/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.visual_context_synthesis.utils.error_mapper import map_visual_synthesis_error
from app.features.visual_context_synthesis import exceptions as exc


class TestVisualSynthesisErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (MemoryError("out of memory"), "slm loading", exc.VisualSynthesisOOMError),
            (FileNotFoundError("gguf missing"), "loading", exc.VisualSLMLoadError),
            (Exception("llama_cpp crashed"), "inference", exc.VisualSLMInferenceError),
            (ValueError("math domain error"), "fuzzy grouping", exc.FuzzyGroupingError),
            (SQLAlchemyError("DB offline"), "save timeline", exc.VisualTimelinePersistenceError),
            (core_db_exc.DatabaseError("General DB"), "unknown", exc.VisualSynthesisDatabaseError),
            (Exception("Cosmic ray"), "unknown", exc.VisualSynthesisError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped = map_visual_synthesis_error(input_exception, context=context_str)
        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.FuzzyGroupingError("Already mapped")
        mapped = map_visual_synthesis_error(original)
        assert mapped is original