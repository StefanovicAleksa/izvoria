# Canvas: test_visual_analysis_error_mapper_v1
# Path: tests/features/visual_analysis/utils/test_error_mapper.py

import pytest
from PIL import UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.core.model_manager import exceptions as core_model_exc
from app.features.visual_analysis.utils.error_mapper import map_visual_analysis_error
from app.features.visual_analysis import exceptions as exc


class TestVisualAnalysisErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (SQLAlchemyError("DB lock"), "saving chunk 123", exc.VisualPersistenceError),
            (core_db_exc.DatabaseError("Connection lost"), "querying", exc.VisualDatabaseError),
            (FileNotFoundError("frame_001.jpg missing"), "fetching image", exc.ImageDecodeError),
            (UnidentifiedImageError("corrupted jpeg header"), "decoding image", exc.ImageDecodeError),
            (RuntimeError("llama_cpp inference failed"), "multimodal prompt", exc.VLMInferenceError),
            (ValueError("tensor size mismatch"), "tensor multimodal", exc.VLMInferenceError),
            (Exception("Cosmic ray bit flip"), "unknown context", exc.VisualAnalysisError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        """Verifies that the error mapper correctly buckets strings and types."""
        mapped = map_visual_analysis_error(input_exception, context=context_str)

        assert isinstance(mapped, expected_output_type)
        assert mapped.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        """Ensures we don't infinitely wrap domain exceptions."""
        original = exc.ImageDecodeError("Already mapped")
        mapped = map_visual_analysis_error(original)
        assert mapped is original

    def test_map_error_bubbles_up_vram_errors(self):
        """
        CRITICAL: The Visual feature must NOT swallow OutOfVramError,
        otherwise the Core Orchestrator cannot trigger its emergency release.
        """
        oom_error = core_model_exc.OutOfVramError("12GB GPU exhausted")
        with pytest.raises(core_model_exc.OutOfVramError):
            # The mapper should actively re-raise this, not return a new mapped error
            map_visual_analysis_error(oom_error)