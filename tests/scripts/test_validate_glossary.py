# Path: tests/scripts/test_validate_glossary.py

import copy
import json

from scripts.validate_glossary import check, main, merge, split_answer


def _glossary(code: str, **overrides) -> dict:
    glossary = {
        "industry": "finance", "language": code,
        "base_terms": ["basis point"],
        "sub_domains": {"ECB": {"anchor_words": ["ECB", "Bund"], "terms": ["Bundesbank"]}},
        "term_details": {"basis point": {"shared": False}, "ECB": {"shared": True}, "Bund": {"shared": True},
                         "Bundesbank": {"shared": True}},
    }
    glossary.update(overrides)
    return glossary


def _pair() -> dict:
    sr = _glossary("sr", base_terms=["bazni poen"])
    sr["term_details"] = {"bazni poen": {"shared": False, "forms": ["baznog poena"]},
                          "ECB": {"shared": True}, "Bund": {"shared": True}, "Bundesbank": {"shared": True}}
    return {"en": _glossary("en"), "sr": sr}


class TestCheck:
    def test_a_consistent_pair_passes(self):
        findings = check(_pair())

        assert (findings.errors, findings.warnings) == ([], [])

    def test_sub_domain_labels_must_match_across_languages(self):
        glossaries = _pair()
        glossaries["sr"]["sub_domains"]["EXTRA"] = {"anchor_words": ["Lagarde"], "terms": []}
        glossaries["sr"]["term_details"]["Lagarde"] = {"shared": False}

        assert check(glossaries).errors == ["[en] sub-domain EXTRA is missing (labels must match in every language)"]

    def test_shared_term_must_be_spelled_identically_everywhere(self):
        glossaries = _pair()
        glossaries["sr"]["sub_domains"]["ECB"]["terms"] = ["Bundesbanka"]
        glossaries["sr"]["term_details"]["Bundesbanka"] = glossaries["sr"]["term_details"].pop("Bundesbank")

        assert check(glossaries).errors == [
            "[en] 'Bundesbank' is shared but [sr] doesn't list it with identical spelling",
            "[sr] 'Bundesbanka' is shared but [en] doesn't list it with identical spelling",
        ]

    def test_serbian_cyrillic_is_rejected_in_terms_and_forms(self):
        glossaries = _pair()
        glossaries["sr"]["term_details"]["bazni poen"]["forms"] = ["базног поена"]

        assert check(glossaries).errors == ["[sr] 'базног поена' is in Cyrillic; sr glossaries must be Latin script"]

    def test_anchor_in_two_sub_domains_and_anchor_repeated_as_term(self):
        glossary = _glossary("en")
        glossary["sub_domains"]["FED"] = {"anchor_words": ["ecb"], "terms": []}
        glossary["sub_domains"]["ECB"]["terms"].append("Bund")
        glossary["term_details"]["ecb"] = {"shared": False}

        assert check({"en": glossary}).errors == [
            "[en] ECB: 'Bund' is both an anchor word and a term",
            "[en] anchor 'ecb' is in both ECB and FED",
        ]

    def test_missing_details_forms_repeating_term_and_reviewed_term_in_glossary(self):
        glossary = _glossary("en", review=[{"term": "Bund", "reason": "unsure"}])
        del glossary["term_details"]["Bundesbank"]
        glossary["term_details"]["ECB"]["forms"] = ["ECB"]

        assert check({"en": glossary}).errors == [
            "[en] 'Bundesbank' has no term_details entry",
            "[en] term_details['ECB'].forms repeats the term itself",
            "[en] 'Bund' is under review but also in the glossary",
        ]

    def test_hand_written_glossary_only_warns(self):
        glossary = {"base_terms": ["hawkish"], "sub_domains": {"FED": {"anchor_words": ["FOMC"], "terms": []}}}

        findings = check({"en": glossary})

        assert findings.errors == []
        assert findings.warnings == ["[en] no term_details (hand-written glossary?): shared/forms can't be checked"]


class TestSplitAndMerge:
    def test_review_items_go_to_their_language(self):
        answer = {**_pair(), "review": [{"term": "x", "language": "sr", "reason": "?"}]}

        glossaries = split_answer(answer)

        assert set(glossaries) == {"en", "sr"}
        assert "review" not in glossaries["en"]
        assert glossaries["sr"]["review"] == [{"term": "x", "language": "sr", "reason": "?"}]

    def test_merge_replaces_sub_domains_and_unions_the_rest(self):
        existing = _glossary("en", review=[{"term": "a"}])
        new = {"industry": "finance", "language": "en", "base_terms": ["basis point", "dovish"],
               "sub_domains": {"ECB": {"anchor_words": ["ECB"], "terms": []},
                               "FED": {"anchor_words": ["FOMC"], "terms": []}},
               "term_details": {"FOMC": {"shared": True}, "dovish": {"shared": True}},
               "review": [{"term": "a"}, {"term": "b"}]}

        merged = merge(copy.deepcopy(existing), new)

        assert merged["base_terms"] == ["basis point", "dovish"]
        assert merged["sub_domains"] == new["sub_domains"]
        assert set(merged["term_details"]) == {"basis point", "ECB", "Bund", "Bundesbank", "FOMC", "dovish"}
        assert merged["review"] == [{"term": "a"}, {"term": "b"}]


class TestMain:
    def test_from_claude_write_splits_into_language_files(self, tmp_path):
        answer = tmp_path / "answer.json"
        answer.write_text(json.dumps(_pair()), encoding="utf-8")

        status = main(["--from-claude", str(answer), "--write", "--dir", str(tmp_path / "glossaries")])

        assert status == 0
        written = json.loads((tmp_path / "glossaries" / "sr" / "finance.json").read_text(encoding="utf-8"))
        assert written["base_terms"] == ["bazni poen"]
        assert main(["finance", "--languages", "en", "sr", "--dir", str(tmp_path / "glossaries")]) == 0

    def test_errors_block_writing(self, tmp_path):
        glossaries = _pair()
        glossaries["sr"]["base_terms"] = ["базни поен"]
        answer = tmp_path / "answer.json"
        answer.write_text(json.dumps(glossaries), encoding="utf-8")

        status = main(["--from-claude", str(answer), "--write", "--dir", str(tmp_path / "glossaries")])

        assert status == 1
        assert not (tmp_path / "glossaries").exists()
