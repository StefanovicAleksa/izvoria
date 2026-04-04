# Canvas: test_phonetic_error_mapper_v3
# Path: tests/features/domain_phonetic_correction/utils/test_error_mapper.py

import pytest
import json
import torch
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error
from app.features.domain_phonetic_correction import exceptions as exc

class TestPhoneticErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), exc.PhoneticOOMError),
            (FileNotFoundError("Missing gguf file"), exc.GGUFLoadError),
            (json.JSONDecodeError("Expecting value", "", 0), exc.JSONPatchParseError),
            (ValidationError.from_exception_data("Mock Error", []), exc.JSONPatchParseError),
            (SQLAlchemyError("DB Transaction Failed"), exc.PhoneticDatabaseError),
            (Exception("Llama.cpp generation crashed"), exc.LLMPatchGenerationError),
            # NEW ASSERTION: Proves IndexError triggers the correct domain exception
            (IndexError("list index out of range"), exc.PatchApplicationError),
            (Exception("Random cosmic ray"), exc.PhoneticCorrectionError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        mapped_error = map_phonetic_error(input_exception, context="test context")
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.JSONPatchParseError("Already mapped")
        mapped = map_phonetic_error(original)
        assert mapped is original