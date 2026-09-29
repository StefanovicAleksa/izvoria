# Path: tests/scripts/eval/cross_language/test_language_id.py

import pytest

from app.core.language.service.language_api import get_language_pack
from scripts.eval.cross_language.language_id import detect_language, distinctive_stopwords


@pytest.fixture(scope="module")
def distinctive():
    return distinctive_stopwords({c: get_language_pack(c).text.stopwords for c in ("en", "sr")})


class TestDetectLanguage:
    @pytest.mark.parametrize("text, expected", [
        ("The researchers planned to test the model against children of different ages [Doc 1].", "en"),
        ("Istraživači su planirali da testiraju model na deci različitog uzrasta, jer je to bilo važno [Doc 1].",
         "sr"),
    ])
    def test_tells_the_languages_apart(self, distinctive, text, expected):
        assert detect_language(text, distinctive) == expected

    def test_quoted_excerpts_and_citations_do_not_count(self, distinctive):
        text = 'The speaker says "ugovor je raskinut i to je bilo to" and that is why the contract ended [Doc 2].'

        assert detect_language(text, distinctive) == "en"

    def test_too_little_text_is_undecided(self, distinctive):
        assert detect_language("GPT-3 [Doc 1].", distinctive) is None

    def test_shared_words_decide_nothing(self):
        distinctive = distinctive_stopwords({"sr": ["je", "i"], "sl": ["je", "in"]})

        assert distinctive == {"sr": frozenset({"i"}), "sl": frozenset({"in"})}
