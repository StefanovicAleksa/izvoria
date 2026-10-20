# Path: app/features/source_profile/exceptions/model.py

from .base import SourceProfileError


class ProfileGenerationError(SourceProfileError):
    """Raised when the language model fails while writing a profile or picking a question's source."""
    pass


class ProfileParseError(SourceProfileError):
    """Raised when the language model's reply is not the JSON the prompt asked for."""
    pass
