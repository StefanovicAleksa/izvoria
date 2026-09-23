# Path: tests/conftest.py

import pytest

from app.core.config import settings
from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository


@pytest.fixture
def language_pack():
    """
    Loads one of the shipped packs in data/languages by code. The packs are configuration checked in beside the
    code, read like an import; a test patches `get_language_pack` with the result to run as that language.
    """
    repository = TomlLanguagePackRepository(settings.BASE_DIR / "data" / "languages")
    return repository.load


@pytest.fixture
def appliance_language(monkeypatch):
    """
    Switches the whole process to another shipped language for one test: every `get_language_pack()` caller, in
    every module, then sees that pack. Model-path overrides are not re-applied; unit tests load no models.
    """
    def activate(code: str) -> None:
        monkeypatch.setattr(settings, "APPLIANCE_LANGUAGE", code)
    return activate
