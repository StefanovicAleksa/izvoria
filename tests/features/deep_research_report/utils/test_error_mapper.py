# Path: tests/features/deep_research_report/utils/test_error_mapper.py

import json

import pytest
from pydantic import BaseModel, ValidationError

from app.core.database import exceptions as db_exc
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.deep_research_report import exceptions as exc
from app.features.deep_research_report.utils.error_mapper import map_report_error


def _validation_error() -> ValidationError:
    class Model(BaseModel):
        value: int

    try:
        Model(value="not an int")
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


@pytest.mark.parametrize("raised,expected", [
    (db_exc.SystemError("connection refused"), exc.ReportPersistenceError),
    (OSError("read-only file system"), exc.ReportFileError),
    (IOError("disk full"), exc.ReportFileError),
    (RuntimeError("CUDA error: out of memory"), exc.ReportOOMError),
    (RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"), exc.ReportOOMError),
    (json.JSONDecodeError("Expecting value", "", 0), exc.JSONPlanParseError),
    (RuntimeError("could not parse json reply"), exc.JSONPlanParseError),
    (RuntimeError("llama_cpp assertion failed"), exc.ReportInferenceError),
    (RuntimeError("generation aborted"), exc.ReportInferenceError),
    (RuntimeError("context window exceeded"), exc.ReportInferenceError),
    (ValueError("something else entirely"), exc.DeepResearchReportError),
])
def test_maps_raw_exceptions_to_typed_report_exceptions(raised, expected):
    mapped = map_report_error(raised, context="unit test")

    assert isinstance(mapped, expected)
    assert mapped.original_exception is raised
    assert "unit test" in str(mapped)


def test_a_pydantic_validation_error_is_a_plan_parse_failure():
    mapped = map_report_error(_validation_error(), context="scoping")

    assert isinstance(mapped, exc.JSONPlanParseError)


def test_an_already_typed_report_error_passes_through_unchanged():
    original = exc.OutlineValidationError("duplicate keys")

    assert map_report_error(original, context="approval") is original


def test_model_manager_errors_pass_through_unwrapped():
    """
    VRAM admission and orchestration failures keep their own type all the way to the engine. Flattening
    an OutOfVramError into a generic report error would hide the one cause an operator can act on.
    """
    original = ModelManagerError("Insufficient VRAM")

    assert map_report_error(original, context="writing") is original


def test_cancellation_is_not_translated_into_a_failure():
    original = exc.ReportCancelledError("stopped by the user")

    assert map_report_error(original) is original


def test_oom_is_detected_before_the_generic_inference_branch():
    """An OOM message also mentions the model; the more specific branch has to win."""
    mapped = map_report_error(RuntimeError("llama_cpp: CUDA out of memory during generation"))

    assert isinstance(mapped, exc.ReportOOMError)
