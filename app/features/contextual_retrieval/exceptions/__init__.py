# Canvas: contextual_exceptions_init_v1
# Path: app/features/contextual_retrieval/exceptions/__init__.py

from .base import ContextualRetrievalError
from .database import ContextualDatabaseError, ContextualPersistenceError
from .model import ContextualModelError, SLMModelLoadError, SLMInferenceError, ContextWindowExceededError
from .process import ContextualPipelineError, TranscriptChunkingError

__all__ = [
    "ContextualRetrievalError",
    "ContextualDatabaseError",
    "ContextualPersistenceError",
    "ContextualModelError",
    "SLMModelLoadError",
    "SLMInferenceError",
    "ContextWindowExceededError",
    "ContextualPipelineError",
    "TranscriptChunkingError"
]