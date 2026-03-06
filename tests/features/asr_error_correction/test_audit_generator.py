# Canvas: test_audit_generator_v1
# Path: tests/features/asr_error_correction/test_audit_generator.py

import pytest
from app.features.asr_error_correction.utils.audit_generator import validate_correction_integrity
from app.features.asr_error_correction.exceptions.pipeline import HallucinationTrapError


def test_validate_correction_integrity_passes_on_minor_fix():
    original = "I think it is a m r i scan."
    corrected = "I think it is an MRI scan."
    # Should not raise an exception
    validate_correction_integrity(original, corrected, similarity_threshold=0.6)


def test_validate_correction_integrity_raises_on_hallucination():
    original = "The patient has a slight cough."
    corrected = "The patient was diagnosed with severe pneumonia and requires immediate surgery."

    with pytest.raises(HallucinationTrapError, match="Model attempted a full rewrite"):
        validate_correction_integrity(original, corrected, similarity_threshold=0.6)


def test_validate_correction_integrity_normalizes_whitespace():
    original = "This   is  spaced   weirdly."
    corrected = "This is spaced weirdly."
    # Even though raw strings differ, normalized ones match perfectly
    validate_correction_integrity(original, corrected, similarity_threshold=0.9)