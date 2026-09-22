# Path: tests/features/deep_research_report/utils/test_query_log.py

import pytest

from app.features.deep_research_report.domain.entities import QueryAttempt
from app.features.deep_research_report.utils.query_log import (
    already_asked, format_attempts, normalise_query,
)


def _attempt(text: str, chunks_added: int = 0, round_index: int = 1) -> QueryAttempt:
    return QueryAttempt(text=text, round_index=round_index, chunks_added=chunks_added)


class TestNormaliseQuery:
    @pytest.mark.parametrize("left,right", [
        ("What was sealed?", "what was sealed"),
        ("  What   was  sealed ", "What was sealed?"),
        ("What was sealed.", "What was sealed"),
        ("WHAT WAS SEALED!", "what was sealed"),
        ("What was sealed:", "What was sealed"),
    ])
    def test_differences_that_cannot_change_retrieval_compare_equal(self, left, right):
        """
        The fallback query is built from a title and brief; the model writes its own. The two can differ
        only in case, spacing or trailing punctuation and still be one search.
        """
        assert normalise_query(left) == normalise_query(right)

    @pytest.mark.parametrize("left,right", [
        ("what was sealed", "what was stored"),
        ("who sealed the bag", "who sealed the box"),
        ("balance sheet runoff", "balance sheet runoff pace"),
    ])
    def test_genuinely_different_questions_stay_different(self, left, right):
        """
        Over-matching is the dangerous direction: it would drop a round that had something to find.
        """
        assert normalise_query(left) != normalise_query(right)

    def test_an_empty_query_normalises_to_empty(self):
        assert normalise_query("   ") == ""
        assert normalise_query(None) == ""


class TestAlreadyAsked:
    def test_every_attempt_is_returned_in_comparable_form(self):
        asked = already_asked([_attempt("What was sealed?"), _attempt("Who signed it")])

        assert asked == {"what was sealed", "who signed it"}

    def test_nothing_asked_yet_is_an_empty_set(self):
        assert already_asked([]) == set()


class TestFormatAttempts:
    def test_round_one_gets_no_block_at_all(self):
        """
        Deep mode and the first round must see the prompt they have always seen -- not a heading with
        nothing under it.
        """
        assert format_attempts([]) == ""

    def test_a_productive_query_reports_what_it_added(self):
        block = format_attempts([_attempt("who sealed the bag", chunks_added=3)])

        assert "who sealed the bag" in block
        assert "3 new excerpt(s)" in block

    def test_a_query_that_found_nothing_says_so_in_the_strongest_terms(self):
        """
        This is the entry the model leans on: zero means those words are absent from the archive, so a
        synonym of them will fail too.
        """
        block = format_attempts([_attempt("balance sheet runoff", chunks_added=0)])

        assert "NOTHING NEW" in block
        assert "not in this archive" in block

    def test_the_block_is_headed_so_the_prompt_rule_can_refer_to_it(self):
        assert format_attempts([_attempt("x")]).startswith("\nSEARCHES ALREADY RUN\n")

    def test_a_long_query_is_truncated_rather_than_crowding_out_the_evidence_index(self):
        block = format_attempts([_attempt("word " * 200, chunks_added=1)])

        assert len(block) < 300
        assert "…" in block
