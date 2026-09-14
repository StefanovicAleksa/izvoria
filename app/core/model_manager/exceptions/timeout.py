# File: app/core/model_manager/exceptions/timeout.py
from app.core.model_manager.exceptions.base import ModelManagerError


class ModelAcquireTimeoutError(ModelManagerError):
    """
    Raised when an acquire call with `wait_for_clear=True` gives up waiting for the
    currently-resident models to be released. Without this bound, a single leaked
    model would park the caller (and the single-worker preload queue) forever.
    """
    pass
