# Path: tests/core/language/utils/test_error_mapper.py

import tomllib

import pytest
from pydantic import BaseModel, ValidationError

from app.core.language import exceptions as exc
from app.core.language.utils.error_mapper import map_language_error


class _Model(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    try:
        _Model(value="x")
    except ValidationError as e:
        return e


def _toml_error() -> tomllib.TOMLDecodeError:
    try:
        tomllib.loads("= broken")
    except tomllib.TOMLDecodeError as e:
        return e


class TestMapLanguageError:
    @pytest.mark.parametrize("raw, expected", [
        (FileNotFoundError("gone"), exc.LanguagePackNotFoundError),
        (_toml_error(), exc.InvalidLanguagePackError),
        (_validation_error(), exc.InvalidLanguagePackError),
        (KeyError("search"), exc.InvalidLanguagePackError),
        (RuntimeError("disk on fire"), exc.LanguageError),
    ])
    def test_maps_to_typed_exception(self, raw, expected):
        mapped = map_language_error(raw, context="sr.toml")

        assert type(mapped) is expected
        assert mapped.original_exception is raw
        assert "sr.toml" in str(mapped)

    def test_language_errors_pass_through_unwrapped(self):
        original = exc.LanguagePackNotFoundError("no pack")
        assert map_language_error(original) is original
