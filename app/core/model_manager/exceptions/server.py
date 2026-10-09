# File: app/core/model_manager/exceptions/server.py
from app.core.model_manager.exceptions.base import ModelManagerError


class LlamaServerError(ModelManagerError):
    """
    Raised when the local llama-server (LLM_BACKEND=llama_server) can't be reached or rejects a request. A model
    error like any other: callers up the stack handle it as a ModelManagerError.
    """
    pass


class ResidencyError(ModelManagerError):
    """
    Raised when a concurrent (hosted) engine asks for a model that is neither pinned nor external: loading or
    releasing it while other requests run could free a model somebody is still using.
    """
    pass
