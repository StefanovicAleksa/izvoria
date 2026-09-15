import pytest

from app.features.domain_phonetic_correction.utils.phonetic_key import letters_only, phonetic_key, spelled_letters


class TestLettersOnly:
    def test_keeps_lowercase_letters_and_digits(self) -> None:
        assert letters_only("Non-farm Payrolls, E.C.B. 13.50") == "nonfarmpayrollsecb1350"


class TestPhoneticKey:
    @pytest.mark.parametrize("a, b", [
        ("Bund", "bunt"),          # voiced/voiceless pair
        ("Draghi", "Dragi"),        # silent h
        ("Schnabel", "Sknabel"),    # sch
        ("phase", "fase"),          # ph
        ("quantitative", "kwantitative"),
    ])
    def test_similar_sounds_share_a_key(self, a: str, b: str) -> None:
        assert phonetic_key(a) == phonetic_key(b)

    def test_vowels_are_dropped_except_the_first_letter(self) -> None:
        assert phonetic_key("easing") == "asnk"

    def test_repeated_symbols_collapse(self) -> None:
        assert phonetic_key("Pepp") == "p"

    def test_empty_for_text_without_letters(self) -> None:
        assert phonetic_key("73.5") == ""


class TestSpelledLetters:
    @pytest.mark.parametrize("tokens, expected", [
        (["E", "C", "B"], "ecb"),
        (["tee", "el", "tee", "are", "oh"], "tltro"),
        (["see", "P.", "eye"], "cpi"),
    ])
    def test_letter_names_become_letters(self, tokens: list, expected: str) -> None:
        assert spelled_letters(tokens) == expected

    def test_empty_when_any_token_is_not_a_letter_name(self) -> None:
        assert spelled_letters(["the", "sea", "a", "grr"]) == ""

    def test_empty_for_no_tokens(self) -> None:
        assert spelled_letters([]) == ""
