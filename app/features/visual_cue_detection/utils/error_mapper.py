# Canvas: visual_cue_error_mapper_v2
# Path: app/features/visual_cue_detection/utils/error_mapper.py

import json
from app.features.visual_cue_detection import exceptions as exc


def map_detection_error(e: Exception, context: str = "") -> exc.VisualCueDetectionError:
    """
    Translates raw JSON parsing and Llama-cpp exceptions into typed Domain Exceptions.
    """
    error_msg = str(e).lower()

    if isinstance(e, exc.VisualCueDetectionError):
        return e

    # 1. JSON Parsing Errors
    # FIX: Explicitly catch JSONDecodeError without requiring the word "json" to be in its string
    if isinstance(e, json.JSONDecodeError) or (isinstance(e, ValueError) and "json" in error_msg):
        return exc.CueParseError(
            f"SLM failed to generate valid JSON for visual cue detection: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 2. Llama-cpp Inference Errors
    if isinstance(e, RuntimeError) or "llama" in error_msg or "context" in error_msg:
        return exc.LLMCueGenerationError(
            f"The local SLM crashed during inference: {str(e)} | Context: {context}",
            original_exception=e
        )

    # 3. Fallback Catch-all
    return exc.VisualCueDetectionError(
        f"Unexpected failure in Visual Cue Detection: {str(e)} | Context: {context}",
        original_exception=e
    )