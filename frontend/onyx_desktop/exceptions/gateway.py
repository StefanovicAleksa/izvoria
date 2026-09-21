# Canvas: desktop_exceptions_gateway_v1
# Path: frontend/onyx_desktop/exceptions/gateway.py

from onyx_desktop.exceptions.base import DesktopError


class ApplianceOfflineError(DesktopError):
    """A call was made before the engine was started, or after it was stopped."""
    pass


class ApplianceBusyError(DesktopError):
    """
    The hardware is locked by a batch ingestion. Live search and chat are mutually exclusive
    with ingestion while both compete for the same GPU.
    """
    pass


class ApplianceBootError(DesktopError):
    """The engine could not be built or started (database down, model weights missing)."""
    pass


class OutOfMemoryError(DesktopError):
    """A model did not fit in VRAM for this request."""
    pass


class EngineCallFailedError(DesktopError):
    """Any other failure inside an engine call."""
    pass


class InvalidRequestError(DesktopError):
    """The request was rejected before reaching the engine (e.g. a blank prompt)."""
    pass
