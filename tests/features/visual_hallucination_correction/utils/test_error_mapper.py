# Canvas: test_hallucination_error_mapper_v1
# Path: tests/features/visual_hallucination_correction/utils/test_error_mapper.py

import pytest
import json
import torch
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError

from app.core.database import exceptions as core_db_exc
from app.features.visual_hallucination_correction.utils.error_mapper import map_hallucination_error
from app.features.visual_hallucination_correction import exceptions as exc


class TestHallucinationErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, context_str, expected_output_type",
        [
            (torch.cuda.OutOfMemoryError("CUDA out of memory"), "audit_gen", exc.SLMOOMError),
            (MemoryError("out of memory"), "adapter_load", exc.SLMOOMError),
            (json.JSONDecodeError("Expecting value", "", 0), "audit_gen", exc.JSONAuditParseError),
            (ValidationError.from_exception_data("Mock Error", []), "audit_gen", exc.JSONAuditParseError),
            (SQLAlchemyError("DB Transaction Failed"), "audit persistence", exc.AuditPersistenceError),
            (core_db_exc.DatabaseError("General DB issue"), "unknown", exc.VisualHallucinationDatabaseError),
            (Exception("llama_cpp context window crashed"), "audit_gen", exc.SLMInferenceError),
            (Exception("Random cosmic ray"), "unknown", exc.VisualHallucinationError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, context_str, expected_output_type):
        mapped_error = map_hallucination_error(input_exception, context=context_str)
        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_error_prevents_double_wrapping(self):
        original = exc.JSONAuditParseError("Already mapped")
        mapped = map_hallucination_error(original)
        assert mapped is original