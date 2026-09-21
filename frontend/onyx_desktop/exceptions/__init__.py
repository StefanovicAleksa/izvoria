# Canvas: desktop_exceptions_init_v1
# Path: frontend/onyx_desktop/exceptions/__init__.py

from onyx_desktop.exceptions.base import DesktopError
from onyx_desktop.exceptions.gateway import (
    ApplianceBootError,
    ApplianceBusyError,
    ApplianceOfflineError,
    EngineCallFailedError,
    InvalidRequestError,
    OutOfMemoryError,
)

__all__ = [
    "DesktopError",
    "ApplianceBootError",
    "ApplianceBusyError",
    "ApplianceOfflineError",
    "EngineCallFailedError",
    "InvalidRequestError",
    "OutOfMemoryError",
]
