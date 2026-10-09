# Path: app/core/language/service/language_api.py

"""
The public entrypoint for language rules.

Most callers are pure helpers deep inside a feature (a phonetic key, a citation parser, a stopword filter), so
the pack is served by a cached module-level function rather than threaded through every constructor. It follows
the ACTIVE language (`settings.APPLIANCE_LANGUAGE`), which `settings.use_language()` switches between engine
jobs; only one job runs at a time, so a job never sees the language change under it.

The hosted API answers several users at once, each in the language they picked for their question, so it
wraps every request in `language_scope(code)`: a context-local override that `active_language()` and
`get_language_pack()` honour on that request's thread only. Settings are not touched, so concurrent requests
never see each other's language. Work handed to another thread must carry the context with it
(`submit_in_context`). The appliance never sets an override and behaves exactly as before.
"""

import contextvars
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Callable, Iterator, List, Optional

from app.core.config import settings
from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository
from app.core.language.domain.entities import LanguagePack


@lru_cache(maxsize=None)
def _load(packs_dir: str, code: str) -> LanguagePack:
    return TomlLanguagePackRepository(packs_dir).load(code)


_language_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("onyx_language", default=None)


def active_language() -> str:
    """The language answers are written in here and now: this request's override, else the appliance's."""
    return _language_override.get() or settings.APPLIANCE_LANGUAGE


@contextmanager
def language_scope(code: str) -> Iterator[str]:
    """Answers in `code` for the duration of the block, on this thread (and contexts copied from it) only."""
    if code not in settings.supported_languages():
        raise ValueError(
            f"'{code}' is not a supported language. Available: {', '.join(settings.supported_languages())}."
        )
    token = _language_override.set(code)
    try:
        yield code
    finally:
        _language_override.reset(token)


def submit_in_context(executor: Executor, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """executor.submit, with the caller's context (its language override) carried to the worker thread."""
    context = contextvars.copy_context()
    return executor.submit(context.run, fn, *args, **kwargs)


def get_language_pack(code: Optional[str] = None) -> LanguagePack:
    """
    The pack for the active language (`active_language()`), or for `code` -- e.g. a stored report
    rendered in the language it was written in. Each pack is loaded once and cached.
    """
    return _load(str(settings.LANGUAGE_PACKS_DIR), code or active_language())


def get_supported_language_packs() -> List[LanguagePack]:
    """Every language this appliance can ingest and answer in, in code order."""
    return [get_language_pack(code) for code in settings.supported_languages()]


def clear_language_pack_cache() -> None:
    """Forgets loaded packs, for tests that switch the configured language."""
    _load.cache_clear()
