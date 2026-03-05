# Canvas: sentence_alignment_exceptions_v2
# Path: app/features/sentence_reconstruction/exceptions/alignment.py

from .base import SentenceReconstructionError

class SegmentAlignmentError(SentenceReconstructionError):
    """Raised when the virtual slicing logic fails to map a sentence back to a Whisper segment."""
    pass

class ZeroDurationSegmentError(SegmentAlignmentError):
    """Raised during proportional math if a raw segment has a duration of 0.0 seconds (Division by Zero)."""
    pass

class SentenceLengthMismatchError(SegmentAlignmentError):
    """Raised if the combined length of the split sentences doesn't match the original segment length."""
    pass