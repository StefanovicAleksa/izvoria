# Canvas: desktop_error_mapper_v1
# Path: frontend/onyx_desktop/utils/error_mapper.py

"""
Translates engine exceptions into the desktop hierarchy, with a sentence the operator can act on.

Matching is by class name walked over the exception's MRO rather than by importing
`app.engine.exceptions`, for two reasons: this module stays importable (and unit-testable) with
no engine, no torch and no CUDA present, and a subclass of a known engine error still maps
correctly instead of falling through to the generic case.
"""

from typing import Dict, Tuple, Type

from onyx_desktop.exceptions.base import DesktopError
from onyx_desktop.exceptions.gateway import (
    ApplianceBootError,
    ApplianceBusyError,
    ApplianceOfflineError,
    EngineCallFailedError,
    InvalidRequestError,
    OutOfMemoryError,
)

# Engine exception class name -> (desktop exception, operator-facing sentence).
# Names come from app/engine/exceptions/ and the feature exception modules.
_BY_NAME: Dict[str, Tuple[Type[DesktopError], str]] = {
    "EngineOfflineError": (
        ApplianceOfflineError,
        "The appliance is not running. Start it before asking a question.",
    ),
    "EngineBusyError": (
        ApplianceBusyError,
        "The appliance is ingesting media and cannot answer questions until it finishes. "
        "Ingestion and live answering share the same GPU.",
    ),
    "EngineBootError": (
        ApplianceBootError,
        "The appliance failed to start. Check that the database is running and the model "
        "weights are present.",
    ),
    "OutOfVramError": (
        OutOfMemoryError,
        "Not enough GPU memory for this step. Close other GPU programs and try again.",
    ),
    "GenerationOOMError": (
        OutOfMemoryError,
        "The answer model ran out of GPU memory. A shorter conversation or fewer sources "
        "will usually fit.",
    ),
    "CudaCrashError": (
        EngineCallFailedError,
        "The GPU driver reported a fault. The appliance needs to be restarted.",
    ),
    "ContextLengthExceededError": (
        InvalidRequestError,
        "This conversation is too long for the answer model's context. Start a new "
        "conversation to continue.",
    ),
    "ContextWindowExceededError": (
        InvalidRequestError,
        "This conversation is too long for the answer model's context. Start a new "
        "conversation to continue.",
    ),
    "ModelNotFoundError": (
        ApplianceBootError,
        "A model file is missing. Run ./models_download.sh to fetch the weights.",
    ),
    "ModelLoadError": (
        ApplianceBootError,
        "A model could not be loaded. Check the model path and the weights file.",
    ),
    "MissingClipError": (
        EngineCallFailedError,
        "The source clip for that citation is missing from the vault.",
    ),
    "DatabaseError": (
        EngineCallFailedError,
        "The appliance could not reach its database. Check that the database container "
        "is running.",
    ),
    "DatabaseConnectionError": (
        EngineCallFailedError,
        "The appliance could not reach its database. Check that the database container "
        "is running.",
    ),
}

_GENERIC = "The appliance could not complete that request."


def map_engine_error(exc: BaseException, context: str = "") -> DesktopError:
    """
    Returns the desktop exception for `exc`, already carrying a user-facing sentence.

    A DesktopError passes through untouched so a mapped error is never double-wrapped.
    `context` names the operation ("Chat", "Standard search") and is appended to the generic
    message only, where it is the difference between a useless error and a locatable one.
    """
    if isinstance(exc, DesktopError):
        return exc

    for klass in type(exc).__mro__:
        match = _BY_NAME.get(klass.__name__)
        if match is not None:
            desktop_class, message = match
            return desktop_class(message, original_exception=exc)

    # pydantic rejects a blank prompt before any model loads; surface that as the user's mistake
    # rather than as an appliance failure.
    if type(exc).__name__ == "ValidationError":
        return InvalidRequestError(
            "That request was not valid. A question cannot be empty, and at most 20 sources "
            "can be requested.",
            original_exception=exc,
        )

    detail = f"{_GENERIC} ({context})" if context else _GENERIC
    return EngineCallFailedError(detail, original_exception=exc)
