# Path: app/core/language/service/language_api.py

"""
The public entrypoint for language rules.

Most callers are pure helpers deep inside a feature (a phonetic key, a citation parser, a stopword filter), so
the pack is served by a cached module-level function rather than threaded through every constructor. It follows
the ACTIVE language (`settings.APPLIANCE_LANGUAGE`), which `settings.use_language()` switches between engine
jobs; only one job runs at a time, so a job never sees the language change under it.
"""

from functools import lru_cache
from typing import List, Optional

from app.core.config import settings
from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository
from app.core.language.domain.entities import LanguagePack


@lru_cache(maxsize=None)
def _load(packs_dir: str, code: str) -> LanguagePack:
    return TomlLanguagePackRepository(packs_dir).load(code)


def get_language_pack(code: Optional[str] = None) -> LanguagePack:
    """
    The pack for the active language (`settings.APPLIANCE_LANGUAGE`), or for `code` -- e.g. a stored report
    rendered in the language it was written in. Each pack is loaded once and cached.
    """
    return _load(str(settings.LANGUAGE_PACKS_DIR), code or settings.APPLIANCE_LANGUAGE)


def get_supported_language_packs() -> List[LanguagePack]:
    """Every language this appliance can ingest and answer in, in code order."""
    return [get_language_pack(code) for code in settings.supported_languages()]


def clear_language_pack_cache() -> None:
    """Forgets loaded packs, for tests that switch the configured language."""
    _load.cache_clear()
