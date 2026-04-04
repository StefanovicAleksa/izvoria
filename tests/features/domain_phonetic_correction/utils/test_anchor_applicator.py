# Canvas: test_anchor_applicator_v3
# Path: tests/features/domain_phonetic_correction/utils/test_anchor_applicator.py

from app.features.domain_phonetic_correction.utils.anchor_applicator import AnchorPatchApplicator


class TestAnchorPatchApplicator:
    def test_apply_preserves_anchor_case(self):
        text = "The doctor noted, With a cute angina today, the patient is unstable."
        original = "with a cute angina today"
        correction = "with acute angina today"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == "The doctor noted, With acute angina today, the patient is unstable."

    def test_apply_preserves_anchor_spacing(self):
        text = "calculate   the eat bit da \t margin now."
        original = "the eat bit da margin"
        correction = "the EBITDA margin"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == "calculate   the EBITDA \t margin now."

    def test_apply_handles_no_anchors_perfectly(self):
        text = "Eat bit da is crucial."
        original = "eat bit da"
        correction = "EBITDA"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == "EBITDA is crucial."

    def test_apply_ignores_unmatched_strings(self):
        text = "The patient was fine."
        original = "with a cute angina"
        correction = "with acute angina"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == text

    def test_apply_handles_full_phrase_case_change_without_crashing(self):
        # NEW TEST: Proves the IndexError is fixed when orig and corr are phonetically identical
        text = "He has an ebitda margin."
        original = "ebitda margin"
        correction = "EBITDA margin"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == "He has an EBITDA margin."

    def test_apply_handles_identical_words_different_case(self):
        # NEW TEST: Another variation of the edge case
        text = "This is a test."
        original = "a test"
        correction = "A TEST"

        result = AnchorPatchApplicator.apply(text, original, correction)

        assert result == "This is A TEST."