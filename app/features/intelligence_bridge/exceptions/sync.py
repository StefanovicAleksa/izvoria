from .base import IntelligenceBridgeError

class SubFeatureFailureError(IntelligenceBridgeError):
    """Raised when Scene Detector, NLP Router, or Transcription fails to provide required inputs."""
    pass

class TimelineMergeError(IntelligenceBridgeError):
    """Raised when the pHash deduplication or array merging math fails."""
    pass

class BridgeDatabaseError(IntelligenceBridgeError):
    """Raised when the database fails to save or retrieve the map."""
    pass