# path: app/features/transcription/exceptions/language.py
from app.features.transcription.exceptions.base import TranscriptionError


class LanguageMismatchError(TranscriptionError):
    """
    Raised when Whisper confidently hears a different language than the appliance transcribes.

    Forcing Whisper's language on speech in another language doesn't fail: it translates, or invents fluent text
    in the forced language. That output would look grounded in the index while being nothing anyone said, so
    the file is rejected instead.
    """
    pass
