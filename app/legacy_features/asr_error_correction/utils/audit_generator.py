import difflib

from app.legacy_features.asr_error_correction.exceptions.pipeline import HallucinationTrapError


def validate_correction_integrity(original_text: str, corrected_text: str, similarity_threshold: float = 0.6) -> None:
    """
    The ultimate safeguard against 'The Hallucination of Correctness'.
    Checks if Phase 3 (Qwen 2.5) rewrote the sentence rather than just fixing the anomaly.

    If the Levenshtein distance between the original and corrected text drops below the threshold,
    we raise an exception to abort the correction and preserve the original audio reality.
    """
    # Quick normalization for comparison
    norm_original = " ".join(original_text.lower().split())
    norm_corrected = " ".join(corrected_text.lower().split())

    similarity = difflib.SequenceMatcher(None, norm_original, norm_corrected).ratio()

    if similarity < similarity_threshold:
        raise HallucinationTrapError(
            f"Correction integrity check failed. Model attempted a full rewrite instead of a targeted fix. "
            f"Similarity: {similarity:.2f} (Threshold: {similarity_threshold})\n"
            f"Original: '{original_text}'\n"
            f"Corrected: '{corrected_text}'"
        )