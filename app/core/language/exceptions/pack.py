# Path: app/core/language/exceptions/pack.py

from app.core.language.exceptions.base import LanguageError


class LanguagePackNotFoundError(LanguageError):
    """No pack exists for the configured language: the appliance has not been set up for it."""
    pass


class InvalidLanguagePackError(LanguageError):
    """A pack exists but is malformed: bad TOML, a missing table, a regex that doesn't compile."""
    pass
