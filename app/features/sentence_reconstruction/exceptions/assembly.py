# Canvas: sentence_exceptions_assembly_v1
# Path: app/features/sentence_reconstruction/exceptions/assembly.py

from .base import SentenceReconstructionError

class DeterministicAssemblyError(SentenceReconstructionError):
    """Base exception for errors occurring during the pure-Python sentence grouping phase."""
    pass

class MissingDiarizationError(DeterministicAssemblyError):
    """
    Raised when the words lack a speaker_id.
    Phase 7 strictly requires the Phase 6 Diarization 'Speaker Wall' to function safely.
    """
    pass

class TokenMismatchError(DeterministicAssemblyError):
    """
    Raised if the Hugging Face pipeline returns a different number of punctuation
    tags than the number of words we passed in. This is a critical edge case in Token Classification.
    """
    pass