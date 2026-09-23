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


class TestPhoneticKeyInOtherLanguages:
    """The same key built from the Serbian and Slovenian packs' alphabets, folds and letter names."""

    @pytest.mark.parametrize("heard, term", [
        ("Ćirić", "Čirić"),        # č and ć are the commonest Serbian mis-hearing
        ("džak", "đak"),           # so are dž and đ
        ("infarkd", "infarkt"),    # voiced/voiceless pairs
        ("zrak", "srak"),
    ])
    def test_serbian_confusable_spellings_share_a_key(self, appliance_language, heard, term):
        appliance_language("sr")
        assert phonetic_key(heard) == phonetic_key(term)

    def test_serbian_letters_are_kept_not_stripped(self, appliance_language):
        appliance_language("sr")
        assert letters_only("Šećer-12, Đorđe!") == "šećer12đorđe"

    def test_english_pack_still_strips_letters_outside_a_to_z(self, appliance_language):
        appliance_language("en")
        assert letters_only("Šećer") == "eer"

    @pytest.mark.parametrize("code, tokens, letters", [
        ("sr", ["e", "ka", "ge"], "ekg"),
        ("sr", ["ce", "te"], "ct"),
        ("sl", ["em", "er", "i"], "mri"),
        ("sl", ["me", "re", "i"], "mri"),
    ])
    def test_spelled_out_acronyms_use_native_letter_names(self, appliance_language, code, tokens, letters):
        appliance_language(code)
        assert spelled_letters(tokens) == letters
