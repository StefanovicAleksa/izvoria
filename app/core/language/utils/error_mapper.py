# Path: app/core/language/utils/error_mapper.py

import tomllib

from pydantic import ValidationError

from app.core.language import exceptions as exc


def map_language_error(e: Exception, context: str = "") -> exc.LanguageError:
    """Translates file, TOML and validation failures into typed Language exceptions."""
    if isinstance(e, exc.LanguageError):
        return e

    where = f" ({context})" if context else ""

    if isinstance(e, FileNotFoundError):
        return exc.LanguagePackNotFoundError(f"Language pack not found{where}.", original_exception=e)

    if isinstance(e, (tomllib.TOMLDecodeError, ValidationError, KeyError, TypeError, ValueError, LookupError)):
        return exc.InvalidLanguagePackError(f"Language pack is invalid{where}: {e}", original_exception=e)

    return exc.LanguageError(f"Unexpected language pack failure{where}: {e}", original_exception=e)
