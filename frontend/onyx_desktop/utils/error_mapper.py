# Canvas: desktop_error_mapper_v1
# Path: frontend/onyx_desktop/utils/error_mapper.py

"""
Translates engine exceptions into the desktop hierarchy, with a plain sentence the person using it can act on.

The sentences never name the appliance's components (models, GPU, database, pipeline stages): the details go
to the log with the original exception, and the screen says what happened and what to do.

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
        "The appliance is busy adding material and cannot answer questions until it finishes.",
    ),
    "EngineBootError": (
        ApplianceBootError,
        "The appliance could not start. Try again, and contact your administrator if it keeps failing.",
    ),
    "OutOfVramError": (
        OutOfMemoryError,
        "The appliance does not have enough free memory for this request right now. Try again shortly.",
    ),
    "GenerationOOMError": (
        OutOfMemoryError,
        "This request was too large to answer. A shorter conversation or fewer sources usually works.",
    ),
    "CudaCrashError": (
        EngineCallFailedError,
        "The appliance ran into a hardware fault and needs to be restarted.",
    ),
    "ContextLengthExceededError": (
        InvalidRequestError,
        "This conversation has grown too long. Start a new conversation to continue.",
    ),
    "ContextWindowExceededError": (
        InvalidRequestError,
        "This conversation has grown too long. Start a new conversation to continue.",
    ),
    "ModelNotFoundError": (
        ApplianceBootError,
        "The appliance is missing some of its files. Contact your administrator.",
    ),
    "ModelLoadError": (
        ApplianceBootError,
        "The appliance could not load one of its components. Contact your administrator.",
    ),
    "MissingClipError": (
        EngineCallFailedError,
        "The source file for that citation is no longer available.",
    ),
    "DatabaseError": (
        EngineCallFailedError,
        "The appliance could not reach its storage. Contact your administrator if this continues.",
    ),
    "DatabaseConnectionError": (
        EngineCallFailedError,
        "The appliance could not reach its storage. Contact your administrator if this continues.",
    ),
}

_GENERIC = "The appliance could not complete that request."


def map_engine_error(exc: BaseException, context: str = "") -> DesktopError:
    """
    Returns the desktop exception for `exc`, already carrying a user-facing sentence.

    A DesktopError passes through untouched so a mapped error is never double-wrapped.
    `context` names the operation ("Chat", "Standard search"). It is not shown: the worker logs it beside the
    original exception, which is where locating a failure happens, while the screen gets a plain sentence.
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

    return EngineCallFailedError(_GENERIC, original_exception=exc)


# ---------- per-file ingestion problems ----------

# Engine messages for a file that was not added, matched to one plain sentence each. The engine's own wording
# names its components and internal ids, which a person adding files neither needs nor should see.
_FILE_PROBLEMS = (
    (("already ingested as",), "It was added before in another language."),
    (("heard", "not transcribed"), "Its spoken language does not match the one selected."),
    (("unsupported", "not supported"), "This file type is not supported."),
    (("corrupt", "could not be read", "cannot be read", "no audio"), "The file could not be read."),
)
_FILE_PROBLEM_FALLBACK = "It could not be processed."


def file_problem(message: str) -> str:
    lowered = (message or "").lower()
    for needles, sentence in _FILE_PROBLEMS:
        if any(needle in lowered for needle in needles):
            return sentence
    return _FILE_PROBLEM_FALLBACK


def file_label(key: str) -> str:
    """The file's own name when the engine keyed the error by path; otherwise the key as it is."""
    return str(key).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or str(key)
