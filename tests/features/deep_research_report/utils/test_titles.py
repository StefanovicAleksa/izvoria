# Path: tests/features/deep_research_report/utils/test_titles.py

import pytest

from app.features.deep_research_report.utils.titles import MAX_TITLE_CHARS, clean_title


class TestCleanTitle:
    @pytest.mark.parametrize("raw, expected", [
        ("Supply and Demand", "Supply and Demand"),
        ('"Supply and Demand."', "Supply and Demand"),
        ("# Supply and Demand", "Supply and Demand"),
        ("Title: Supply and Demand", "Supply and Demand"),
        ("**Supply and Demand**", "Supply and Demand"),
        ("  Supply\n and   Demand  ", "Supply and Demand"),
        ("Do Rate Cuts Lift Equities?", "Do Rate Cuts Lift Equities?"),
        ("Ponuda i tražnja: osnovni model", "Ponuda i tražnja: osnovni model"),
        ("", ""),
        (None, ""),
        ("  \"\"  ", ""),
    ])
    def test_tidies_what_a_model_or_a_person_typed(self, raw, expected):
        assert clean_title(raw) == expected

    def test_a_long_title_is_cut_at_a_word(self):
        title = clean_title("word " * 60)

        assert len(title) <= MAX_TITLE_CHARS
        assert title.endswith("word")
