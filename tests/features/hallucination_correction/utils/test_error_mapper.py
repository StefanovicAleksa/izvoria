# Canvas: test_correction_error_mapper_v1
# Path: tests/features/hallucination_correction/utils/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.features.hallucination_correction.utils.error_mapper import map_correction_error
from app.features.hallucination_correction import exceptions as exc


class TestCorrectionErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (SQLAlchemyError("DB connection dropped"), exc.CorrectionDatabaseError),
            (ValueError("Segment timestamp overlap impossible"), exc.CorruptSegmentDataError),
            (exc.MissingPrerequisiteError("No VAD map"), exc.MissingPrerequisiteError), # Returns as-is
            (Exception("Random cosmic ray"), exc.HallucinationCorrectionError),
        ]
    )
    def test_map_correction_error_translates_correctly(self, input_exception, expected_output_type):
        """Verifies that the error mapper correctly buckets strings and types."""
        mapped_error = map_correction_error(input_exception)

        assert isinstance(mapped_error, expected_output_type)
        if not isinstance(input_exception, exc.HallucinationCorrectionError):
            assert mapped_error.original_exception == input_exception