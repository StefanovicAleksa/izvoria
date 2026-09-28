# Path: tests/core/language/utils/test_script.py

import pytest

from app.core.language.utils.script import cyrillic_to_latin, fold_to_ascii, normalize_script


class TestCyrillicToLatin:
    @pytest.mark.parametrize("cyrillic, latin", [
        ("лекар", "lekar"),
        ("Лекари су прегледали пацијенте.", "Lekari su pregledali pacijente."),
        ("ђак ћевап чаша шума жаба", "đak ćevap čaša šuma žaba"),
        ("љубав њива џеп", "ljubav njiva džep"),
        ("Љубав Њива Џеп", "Ljubav Njiva Džep"),        # a capitalised digraph letter starts a word: title case
        ("ЉУБАВ ЊИВА ЏЕП", "LJUBAV NJIVA DŽEP"),        # inside a word in capitals: both letters capitals
        ("Љ", "Lj"),
        ("ЕКГ и МРИ", "EKG i MRI"),
    ])
    def test_transliterates_every_serbian_letter(self, cyrillic, latin):
        assert cyrillic_to_latin(cyrillic) == latin

    def test_latin_digits_and_punctuation_pass_through(self):
        assert cyrillic_to_latin("Dr. Petrović, 13:50 — ok?") == "Dr. Petrović, 13:50 — ok?"

    def test_mixed_scripts_become_one(self):
        assert cyrillic_to_latin("Pacijent има infarkt") == "Pacijent ima infarkt"


class TestNormalizeScript:
    def test_latin_script_transliterates(self):
        assert normalize_script("Београд", "latin") == "Beograd"

    def test_no_script_changes_nothing(self):
        assert normalize_script("Београд", None) == "Београд"


class TestFoldToAscii:
    @pytest.mark.parametrize("text, folded", [
        ("Lečenje dijabetesa", "Lecenje dijabetesa"),
        ("Đorđe Šćepanović", "Djordje Scepanovic"),
        ("plain", "plain"),
    ])
    def test_drops_diacritics(self, text, folded):
        assert fold_to_ascii(text) == folded
