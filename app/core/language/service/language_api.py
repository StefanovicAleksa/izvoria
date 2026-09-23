# Path: app/core/language/service/language_api.py

"""
The public entrypoint for language rules.

Most callers are pure helpers deep inside a feature (a phonetic key, a citation parser, a stopword filter), so
the pack is served by a cached module-level function rather than threaded through every constructor: the
language is fixed for the life of the process, exactly like `settings`.
"""

from functools import lru_cache

from app.core.config import settings
from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository
from app.core.language.domain.entities import LanguagePack


@lru_cache(maxsize=None)
def _load(packs_dir: str, code: str) -> LanguagePack:
    return TomlLanguagePackRepository(packs_dir).load(code)


def get_language_pack() -> LanguagePack:
    """The pack for this appliance's language (`settings.APPLIANCE_LANGUAGE`), loaded once and cached."""
    return _load(str(settings.LANGUAGE_PACKS_DIR), settings.APPLIANCE_LANGUAGE)


def clear_language_pack_cache() -> None:
    """Forgets loaded packs, for tests that switch the configured language."""
    _load.cache_clear()
