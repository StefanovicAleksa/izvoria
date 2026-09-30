# Canvas: desktop_i18n_v1
# Path: frontend/onyx_desktop/utils/i18n.py

"""
The app's display language: the language its own labels, buttons and messages are written in.

It is independent of the appliance's answer language (the sidebar's "Answer in", `settings.APPLIANCE_LANGUAGE`):
someone can read the app in Serbian and have it answer in English. Nothing here reaches the engine.

English is the source language. Code writes English and wraps it in `tr()`; the English text is the key, and
`locales/<code>.toml` maps it to another language. A string a catalog lacks falls back to English, so a
half-finished catalog never shows a blank or a key. Pure Python, no Qt, so every layer can use it.

    tr("Start appliance")
    tr("Step {current} of {total}", current=2, total=5)
    tr_n(files, "{n} file", "{n} files")     # plural forms per language
    N_("Offline")                            # marks a string for translation where it is defined, not shown

A catalog lists its plural forms per string ("one", "few", "many" for Serbian), chosen by its `plural` rule.
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Union

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parents[1] / "locales"
BASE = "en"


@dataclass(frozen=True)
class DisplayLanguage:
    code: str
    name: str   # in its own language ("Srpski"), which is how a person looks for their language in a list


@dataclass(frozen=True)
class _Catalog:
    language: DisplayLanguage
    plural: str
    strings: Dict[str, Union[str, List[str]]]


def _english_plural(n: int) -> int:
    return 0 if n == 1 else 1


def _slavic_plural(n: int) -> int:
    """Serbian, Croatian, Bosnian, Russian...: 1, 21, 31 -> one; 2-4, 22-24 -> few; everything else -> many."""
    if n % 10 == 1 and n % 100 != 11:
        return 0
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return 1
    return 2


PLURAL_RULES: Dict[str, Callable[[int], int]] = {"english": _english_plural, "slavic": _slavic_plural}

_ENGLISH = _Catalog(DisplayLanguage(BASE, "English"), "english", {})
_catalogs: Dict[str, _Catalog] = {}
_current: _Catalog = _ENGLISH


def _load() -> Dict[str, _Catalog]:
    if _catalogs:
        return _catalogs
    _catalogs[BASE] = _ENGLISH
    for path in sorted(LOCALES_DIR.glob("*.toml")):
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        meta = data.get("meta", {})
        code = meta.get("code", path.stem)
        plural = meta.get("plural", "english")
        if plural not in PLURAL_RULES:
            raise ValueError(f"{path.name}: unknown plural rule '{plural}' (known: {', '.join(PLURAL_RULES)})")
        _catalogs[code] = _Catalog(DisplayLanguage(code, meta.get("name", code)), plural, data.get("strings", {}))
    return _catalogs


def available() -> List[DisplayLanguage]:
    """English first, then every catalog in `locales/`. Adding a language is adding a file."""
    return [catalog.language for catalog in _load().values()]


def use(code: str) -> None:
    global _current
    catalogs = _load()
    if code not in catalogs:
        raise ValueError(f"No display language '{code}'. Available: {', '.join(catalogs)}.")
    _current = catalogs[code]


def current() -> str:
    return _current.language.code


def N_(text: str) -> str:
    """Marks `text` for translation where it is defined (a table of labels); `tr()` translates it where shown."""
    return text


def _format(template: str, values: dict) -> str:
    if not values:
        return template
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        # A catalog typo in a placeholder must not take a screen down: show the English instead
        logger.warning(f"Display language '{current()}': bad placeholder in {template!r}")
        return template


def tr(text: str, **values) -> str:
    translated = _current.strings.get(text)
    if isinstance(translated, str):
        result = _format(translated, values)
        if result is not translated or not values:
            return result
    return _format(text, values)


def tr_n(n: int, one: str, other: str, **values) -> str:
    """`one`/`other` are the English forms; the catalog keys the translation's forms by `one`."""
    values = {"n": n, **values}
    forms = _current.strings.get(one)
    if isinstance(forms, list) and forms:
        index = min(PLURAL_RULES[_current.plural](n), len(forms) - 1)
        return _format(forms[index], values)
    return _format(one if _english_plural(n) == 0 else other, values)
