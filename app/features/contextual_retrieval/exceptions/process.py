# Canvas: contextual_exceptions_process_v1
# Path: app/features/contextual_retrieval/exceptions/process.py

from .base import ContextualRetrievalError

class ContextualPipelineError(ContextualRetrievalError):
    """Base exception for parsing, chunking, and data-flow errors."""
    pass

class TranscriptChunkingError(ContextualPipelineError):
    """Raised when the deterministic chunker fails to group sentences or encounters malformed time bounds."""
    pass