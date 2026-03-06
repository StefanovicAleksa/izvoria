# Canvas: test_error_mapper_v1
# Path: tests/features/asr_error_correction/test_error_mapper.py

import pytest
import json
import torch
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.features.asr_error_correction.utils.error_mapper import map_asr_correction_error
from app.features.asr_error_correction import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, context, expected_output_type",
    [
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), "Phase 1", exc.AsrCorrectionOOMError),
        (FileNotFoundError("Model missing"), "Load Model", exc.GGUFModelLoadError),
        (json.JSONDecodeError("Expecting value", "", 0), "Extract Glossary", exc.GlossaryJSONParseError),
        (core_db_exc.DatabaseError("Connection lost"), "save_glossary", exc.GlossaryPersistenceError),
        (SQLAlchemyError("Transaction failed"), "save_audit", exc.AuditTrailSaveError),
        (RuntimeError("tensor dimension mismatch"), "T5", exc.T5InferenceError),
        (Exception("llama_cpp generation failed"), "LLM", exc.LLMInferenceError),
        (ValueError("Random error"), "Unknown", exc.AsrCorrectionError),
    ]
)
def test_map_asr_correction_error_translates_correctly(input_exception, context, expected_output_type):
    mapped_error = map_asr_correction_error(input_exception, context=context)

    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception