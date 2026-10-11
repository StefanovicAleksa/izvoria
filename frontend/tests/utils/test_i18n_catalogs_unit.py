# Canvas: desktop_test_i18n_catalogs_v1
# Path: frontend/tests/utils/test_i18n_catalogs_unit.py

"""
Keeps every display language complete as the app and the appliance grow.

- Every string the code shows is found by reading the source (the arguments of `tr`, `tr_n`, `N_` and the
  `ui/translate.py` helpers), and every catalog must translate all of them. A new label with no translation
  fails here, naming it, instead of quietly showing English to a Serbian reader.
- Every language the appliance serves (`data/languages/*.toml`) must have a display catalog, so adding a language
  pack is not finished until the app can be read in it too.
"""

import ast
import re
import string
import tomllib
from pathlib import Path
from typing import Dict, Set

import pytest

from izvoria_desktop.utils import i18n

FRONTEND = Path(__file__).resolve().parents[2]
SOURCE = FRONTEND / "izvoria_desktop"
SERVED_PACKS = FRONTEND.parent / "data" / "languages"

# helper name -> index of the argument holding the text
_TEXT_ARGUMENT = {"tr": 0, "N_": 0, "label": 0, "text": 1, "tooltip": 1, "placeholder": 1, "title": 1, "items": 1}


def _strings_in(node: ast.AST) -> Set[str]:
    """String literals an argument can evaluate to: a literal, either branch of `a if c else b`, a list of them."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _strings_in(node.body) | _strings_in(node.orelse)
    if isinstance(node, (ast.List, ast.Tuple)):
        return set().union(*(_strings_in(element) for element in node.elts)) if node.elts else set()
    return set()


def _call_name(node: ast.Call) -> str:
    return node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")


def source_strings() -> Dict[str, Set[str]]:
    """{"plain": keys, "plural": (one, other) keys by their `one` form}."""
    plain: Set[str] = set()
    plural: Set[str] = set()
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "tr_n" and len(node.args) >= 3:
                plural |= _strings_in(node.args[1])
            elif name in _TEXT_ARGUMENT and len(node.args) > _TEXT_ARGUMENT[name]:
                plain |= _strings_in(node.args[_TEXT_ARGUMENT[name]])
    plain.discard("")
    return {"plain": plain, "plural": plural}


def _catalog(code: str) -> dict:
    with (i18n.LOCALES_DIR / f"{code}.toml").open("rb") as handle:
        return tomllib.load(handle)


def _placeholders(text: str) -> Set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


TRANSLATED = [language.code for language in i18n.available() if language.code != i18n.BASE]


class TestCatalogs:
    def test_the_source_scan_finds_the_apps_strings(self):
        found = source_strings()
        assert "Start appliance" in found["plain"]
        assert "Analyzing audio" in found["plain"]
        assert "{n} file" in found["plural"]

    @pytest.mark.parametrize("code", TRANSLATED)
    def test_every_string_is_translated(self, code):
        strings = _catalog(code)["strings"]
        found = source_strings()
        missing = sorted((found["plain"] | found["plural"]) - set(strings))
        assert not missing, f"{code}.toml is missing {len(missing)} string(s):\n" + "\n".join(missing)

    @pytest.mark.parametrize("code", TRANSLATED)
    def test_nothing_stale_is_left_in_a_catalog(self, code):
        found = source_strings()
        stale = sorted(set(_catalog(code)["strings"]) - found["plain"] - found["plural"])
        assert not stale, f"{code}.toml translates text the app no longer shows:\n" + "\n".join(stale)

    @pytest.mark.parametrize("code", TRANSLATED)
    def test_translations_keep_their_placeholders(self, code):
        """A translation that drops or renames `{n}` would show the English, or a wrong number."""
        catalog = _catalog(code)
        forms_needed = {"slavic": 3, "english": 2}[catalog["meta"]["plural"]]
        plural_keys = source_strings()["plural"]
        for key, value in catalog["strings"].items():
            if key in plural_keys:
                assert isinstance(value, list) and len(value) == forms_needed, key
                translations = value
            else:
                assert isinstance(value, str), key
                translations = [value]
            for translated in translations:
                assert _placeholders(translated) == _placeholders(key), (key, translated)

    def test_every_served_language_can_be_read_in_the_app(self):
        served = {path.stem for path in SERVED_PACKS.glob("*.toml")}
        displayable = {language.code for language in i18n.available()}
        assert served, "no served language packs found"
        assert served <= displayable, (
            f"no display catalog for {sorted(served - displayable)}: add "
            f"frontend/izvoria_desktop/locales/<code>.toml")


class TestTranslation:
    @pytest.fixture(autouse=True)
    def _english_after(self):
        yield
        i18n.use(i18n.BASE)

    def test_english_is_the_source_and_first(self):
        assert i18n.available()[0].code == "en"
        assert i18n.tr("Start appliance") == "Start appliance"

    def test_a_catalog_translates_and_fills_placeholders(self):
        i18n.use("sr")
        assert i18n.tr("Start appliance") == "Pokreni sistem"
        assert i18n.tr("Step {current} of {total}", current=2, total=5) == "Korak 2 od 5"

    @pytest.mark.parametrize("n,expected", [
        (1, "1 fajl"), (2, "2 fajla"), (4, "4 fajla"), (5, "5 fajlova"), (11, "11 fajlova"),
        (12, "12 fajlova"), (21, "21 fajl"), (22, "22 fajla"), (25, "25 fajlova"),
    ])
    def test_serbian_plurals(self, n, expected):
        i18n.use("sr")
        assert i18n.tr_n(n, "{n} file", "{n} files") == expected

    def test_an_untranslated_string_falls_back_to_english(self):
        i18n.use("sr")
        assert i18n.tr("Some text no catalog has") == "Some text no catalog has"

    def test_an_unknown_language_is_refused(self):
        with pytest.raises(ValueError, match="xx"):
            i18n.use("xx")
