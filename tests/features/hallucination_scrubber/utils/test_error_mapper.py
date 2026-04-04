# Canvas: test_error_mapper_v1
# Path: tests/features/hallucination_scrubber/utils/test_error_mapper.py

import pytest
import torch
from sqlalchemy.exc import SQLAlchemyError

from app.features.hallucination_scrubber.utils.error_mapper import map_scrubber_error
from app.features.hallucination_scrubber import exceptions as exc


class TestScrubberErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context, expected_output_type",
        [
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), "Batch Inference", exc.ScrubberOOMError),
            (FileNotFoundError("config.json missing"), "Load DeBERTa", exc.ZeroShotLoadError),
            (RuntimeError("pipeline tensor shape mismatch"), "DeBERTa Infer", exc.DebertaInferenceError),
            (SQLAlchemyError("DB Transaction Failed"), "transcript save", exc.ScrubbedTranscriptSaveError),
            (SQLAlchemyError("Connection lost"), "Random DB check", exc.ScrubbingDatabaseError),
            (ValueError("clause split failed"), "clause segmentation", exc.TextSegmentationError),
            (Exception("Random cosmic ray"), "Unknown Context", exc.HallucinationScrubberError),
        ]
    )
    def test_map_scrubber_error_translates_correctly(self, input_exception, context, expected_output_type):
        mapped_error = map_scrubber_error(input_exception, context=context)
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.ScrubberOOMError("Already mapped")
        mapped = map_scrubber_error(original)
        assert mapped is original