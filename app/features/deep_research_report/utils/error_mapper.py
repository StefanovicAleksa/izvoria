# Path: app/features/deep_research_report/utils/error_mapper.py

import json
import logging

from pydantic import ValidationError

from app.core.database import exceptions as db_exc
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.deep_research_report import exceptions as exc

logger = logging.getLogger(__name__)


def map_report_error(e: Exception, context: str = "") -> Exception:
    """
    Translates raw system, Llama.cpp, JSON and database exceptions into typed report exceptions.

    Two deliberate pass-throughs:
      - ModelManagerError, so VRAM admission and orchestration failures keep their own type all the way up
        to the engine instead of being flattened into a generic report error.
      - ReportCancelledError, because a cancelled run is a user decision, not a failure to translate.
    """
    if isinstance(e, (exc.DeepResearchReportError, ModelManagerError)):
        return e

    error_msg = str(e).lower()

    if isinstance(e, db_exc.DatabaseError):
        return exc.ReportPersistenceError(
            f"Report persistence failed: {str(e)} | Context: {context}",
            original_exception=e
        )

    if isinstance(e, (OSError, IOError)):
        return exc.ReportFileError(
            f"Could not write the rendered report: {str(e)} | Context: {context}",
            original_exception=e
        )

    # String matching, to avoid importing torch here purely to identify an OOM.
    if "out of memory" in error_msg or "cuda_error_out_of_memory" in error_msg:
        return exc.ReportOOMError(
            f"VRAM exhausted during report authoring: {str(e)} | Context: {context}",
            original_exception=e
        )

    if isinstance(e, (json.JSONDecodeError, ValidationError)) or "json" in error_msg:
        return exc.JSONPlanParseError(
            f"Report model returned an unusable plan: {str(e)} | Context: {context}",
            original_exception=e
        )

    if "llama_cpp" in error_msg or "generation" in error_msg or "context window" in error_msg:
        return exc.ReportInferenceError(
            f"Report model crashed during inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    logger.error(f"Unhandled Deep Research Report Error: {str(e)}", exc_info=True)
    return exc.DeepResearchReportError(
        f"Unexpected failure in Deep Research Report: {str(e)} | Context: {context}",
        original_exception=e
    )
