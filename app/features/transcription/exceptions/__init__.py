# filename: transcription_exceptions_init_v1.py
# path: app/features/transcription/exceptions/__init__.py
from .base import TranscriptionError
from .database import TranscriptionDatabaseError
from .language import LanguageMismatchError
from .model import (
    TranscriptionInferenceError,
    TranscriptionAudioError,
    CTranslateEngineError,
    TranscriptionModelFilePathError
)

__all__ = [
    "TranscriptionError",
    "TranscriptionDatabaseError",
    "LanguageMismatchError",
    "TranscriptionInferenceError",
    "TranscriptionAudioError",
    "CTranslateEngineError",
    "TranscriptionModelFilePathError"
]