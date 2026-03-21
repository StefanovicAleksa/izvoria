# Path: tests/features/domain_adaptation/utils/test_error_mapper.py

import pytest
import json
from sqlalchemy.exc import SQLAlchemyError
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error
from app.features.domain_adaptation import exceptions as exc

class TestDomainAdaptationErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (SQLAlchemyError("DB drop"), exc.DomainDatabaseError),
            (ValueError("audio slice out of bounds"), exc.AudioMemoryProcessingError),
            (json.JSONDecodeError("Expecting value", "", 0), exc.LLMRouterInferenceError),
            (FileNotFoundError("finance.json missing"), exc.GlossaryManifestError),
            (RuntimeError("ctranslate2 compute type failed"), exc.ScoutInferenceError),
            (Exception("Random cosmic ray"), exc.DomainAdaptationError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        # FIXED: Added 'audio slice' to trigger the RAM error trap
        context = "llama execution" if expected_output_type == exc.LLMRouterInferenceError \
            else "whisper scout" if expected_output_type == exc.ScoutInferenceError \
            else "audio slice" if expected_output_type == exc.AudioMemoryProcessingError \
            else "general test context"

        mapped_error = map_domain_adaptation_error(input_exception, context=context)

        assert isinstance(mapped_error, expected_output_type)