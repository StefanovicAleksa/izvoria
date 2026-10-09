# Path: tests/scripts/eval/test_glossary_ablation.py

import pytest

from scripts.eval.glossary_ablation import (
    aggregate, compare, find_occurrences, render_report, score, term_variants, CONFIGS
)

GLOSSARY = {
    "base_terms": ["basis point"],
    "sub_domains": {
        "EUROPEAN_CENTRAL_BANK": {"anchor_words": ["ECB", "Bund"], "terms": ["Deposit Facility Rate"]},
    },
    "term_details": {"Bund": {"forms": ["Bundes"]}},
}


class TestTermVariants:
    def test_terms_forms_and_plurals_map_to_their_term(self):
        variants = term_variants(GLOSSARY, plural_suffix="s")

        assert variants[("bund",)] == "Bund"
        assert variants[("bunds",)] == "Bund"
        assert variants[("bundes",)] == "Bund"
        assert variants[("basis", "points")] == "basis point"
        assert variants[("deposit", "facility", "rate")] == "Deposit Facility Rate"

    def test_no_plural_variant_without_a_suffix(self):
        assert ("bunds",) not in term_variants(GLOSSARY, plural_suffix="")

    def test_bare_list_sub_domains_are_read(self):
        variants = term_variants({"sub_domains": {"LEGAL": ["affidavit"]}})

        assert variants == {("affidavit",): "affidavit"}


class TestFindOccurrences:
    def test_longest_match_wins_and_spans_do_not_overlap(self):
        variants = {("rate",): "Rate", ("deposit", "facility", "rate"): "Deposit Facility Rate"}

        found = find_occurrences(["the", "deposit", "facility", "rate", "and", "rate"], variants)

        assert found == [(1, 4, "Deposit Facility Rate"), (5, 6, "Rate")]


class TestScore:
    def test_term_error_and_ceiling(self):
        variants = term_variants(GLOSSARY, plural_suffix="s")

        result = score("The ECB bought Bunds today.", "the ecb bought burns to day", variants)

        # Errors: bunds->burns (term word), today->to, +day
        assert (result.errors, result.term_occurrences, result.term_errors) == (3, 2, 1)
        assert result.ter == pytest.approx(0.5)
        assert result.ceiling == pytest.approx(1 / 3)

    def test_no_terms_gives_no_rate(self):
        result = score("hello world", "hello world", {})

        assert (result.ter, result.ceiling) == (None, None)


class TestCompare:
    def test_counts_fixed_and_broken_terms_and_other_words(self):
        variants = term_variants(GLOSSARY, plural_suffix="s")
        reference = "the bund and the ecb rose"
        before = score(reference, "the burn and the ecb rose", variants)
        after = score(reference, "the bund and the ec be rows", variants)

        result = compare(before, after)

        assert result["fixed"] == ["Bund"]
        assert result["broken"] == ["ECB"]
        assert (result["other_words_fixed"], result["other_words_broken"]) == (0, 1)


class TestAggregateAndReport:
    def test_totals_pool_words_rather_than_average_rates(self):
        variants = term_variants(GLOSSARY)
        scores = [score("ecb", "ecb", variants), score("a b c d", "a x c d", variants)]

        totals = aggregate(scores)

        assert (totals["wer"], totals["term_occurrences"], totals["ter"], totals["ceiling"]) == (0.2, 1, 0.0, 0.0)

    def test_report_has_a_row_per_config_and_correction(self):
        totals = aggregate([score("ecb", "ecb", term_variants(GLOSSARY))])
        result = {"totals": {config: totals for config in CONFIGS}, "oracle_subdomains": False,
                  "sources": {"clip": {}},
                  "corrections": {"C vs B": {"terms_fixed": 2, "terms_broken": 0,
                                             "other_words_fixed": 0, "other_words_broken": 1}}}

        report = render_report("en", "finance", result)

        assert all(f"| {config} |" in report for config in CONFIGS)
        assert "| C vs B | 2 | 0 | 0 | 1 |" in report


class TestTermDisagreements:
    def test_lists_only_changes_that_touch_a_term_with_context(self):
        from scripts.eval.glossary_ablation import term_disagreements
        variants = term_variants(GLOSSARY, plural_suffix="s")

        found = term_disagreements("they sold the burn and the cat sat", "they sold the bund and the bat sat",
                                   variants, context=2)

        assert found == [{"terms": ["Bund"], "before": "burn", "after": "bund",
                          "context": "sold the [burn -> bund] and the"}]

    def test_a_term_lost_is_reported_too(self):
        from scripts.eval.glossary_ablation import term_disagreements

        found = term_disagreements("the ecb rose", "the ec be rose", term_variants(GLOSSARY))

        assert [d["terms"] for d in found] == [["ECB"]]
