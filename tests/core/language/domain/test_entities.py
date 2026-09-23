# Path: tests/core/language/domain/test_entities.py

import pytest

from app.core.language.domain.entities import ENGLISH_REFUSAL


class TestLocalizePrompt:
    def test_english_prompts_come_back_unchanged(self, language_pack):
        prompt = f'Rule 1: if unsupported, say "{ENGLISH_REFUSAL}"'

        assert language_pack("en").localize_prompt(prompt) == prompt

    def test_refusal_is_swapped_and_output_language_is_appended(self, language_pack):
        pack = language_pack("sr")

        localized = pack.localize_prompt(f'Otherwise reply exactly: {ENGLISH_REFUSAL}')

        assert ENGLISH_REFUSAL not in localized
        assert localized.startswith(f"Otherwise reply exactly: {pack.answers.refusal}")
        assert localized.endswith(f"<OUTPUT_LANGUAGE>\n{pack.answers.output_instruction}\n</OUTPUT_LANGUAGE>")
        assert "Serbian" in localized and "Latin script" in localized

    def test_transcript_prompt_gets_the_never_translate_rule(self, language_pack):
        pack = language_pack("sl")

        assert pack.localize_transcript_prompt("Pick A or B.").endswith(
            f"<TRANSCRIPT_LANGUAGE>\n{pack.answers.transcript_instruction}\n</TRANSCRIPT_LANGUAGE>"
        )
        assert language_pack("en").localize_transcript_prompt("Pick A or B.") == "Pick A or B."


class TestIsRefusal:
    @pytest.mark.parametrize("code", ["en", "sr", "sl"])
    def test_own_refusal_and_english_fallback_both_count(self, language_pack, code):
        pack = language_pack(code)

        assert pack.is_refusal(f"  {pack.answers.refusal}  ")
        assert pack.is_refusal(ENGLISH_REFUSAL.rstrip("."))
        assert pack.is_refusal(ENGLISH_REFUSAL + " The excerpts discuss something else.")

    def test_an_answer_is_not_a_refusal(self, language_pack):
        assert not language_pack("sr").is_refusal("Pacijent je stabilan [Doc 1].")


class TestReportNouns:
    def test_english_nouns_agree_with_the_count(self, language_pack):
        report = language_pack("en").report

        assert (report.noun("source", 1), report.noun("source", 2), report.noun("source", 0)) == \
            ("source", "sources", "sources")

    def test_unknown_noun_is_empty(self, language_pack):
        assert language_pack("sr").report.noun("source", 5) == ""
