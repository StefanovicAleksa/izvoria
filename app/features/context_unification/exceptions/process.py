# Canvas: context_unification_exceptions_process_v1
# Path: app/features/context_unification/exceptions/process.py

from .base import ContextUnificationError

class ContextUnificationPipelineError(ContextUnificationError):
    """Base exception for data-flow and string assembly errors."""
    pass

class TranscriptMismatchError(ContextUnificationPipelineError):
    """
    Raised if the Phase 9 (Audio) and Phase 14 (Visual) transcripts
    belong to completely different source IDs.
    """
    pass