# Path: app/features/source_profile/exceptions/database.py

from .base import SourceProfileError


class SourceProfileDataError(SourceProfileError):
    """Raised when a profile is invalid (an empty title) or names a source that does not exist."""
    pass


class SourceProfileSystemError(SourceProfileError):
    """Raised when the profile store's underlying database is unavailable."""
    pass
