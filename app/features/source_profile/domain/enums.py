# Path: app/features/source_profile/domain/enums.py

from enum import Enum


class ProfileOrigin(str, Enum):
    """Who wrote a profile. An EDITED profile is a person's, and nothing generated ever replaces it."""
    GENERATED = "generated"
    NAMED = "named"
    EDITED = "edited"
