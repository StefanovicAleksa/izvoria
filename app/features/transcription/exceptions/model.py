# filename: transcription_exceptions_model_v1.py
# path: app/features/transcription/exceptions/model.py
from app.features.transcription.exceptions.base import TranscriptionError

class TranscriptionInferenceError(TranscriptionError):
    """Raised when the Faster-Whisper model fails during the actual transcription pass."""
    pass

class TranscriptionAudioError(TranscriptionError):
    """
    Raised when the provided audio file cannot be sliced, or the timestamps
    provided by the VAD map are invalid (e.g., out of bounds).
    """
    pass

class CTranslateEngineError(TranscriptionError):
    """
    Raised when the underlying CTranslate2 runtime crashes.
    Commonly caused by incompatible compute types (e.g., requesting INT8 on unsupported hardware).
    """
    pass

class TranscriptionModelFilePathError(TranscriptionError):
    """Raised when the specified Faster-Whisper model directory cannot be found on disk."""
    pass