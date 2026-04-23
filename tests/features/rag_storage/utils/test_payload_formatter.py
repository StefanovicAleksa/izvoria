# Canvas: test_payload_formatter_v1
# Path: tests/features/rag_storage/utils/test_payload_formatter.py

import pytest
from app.features.rag_storage.utils.payload_formatter import normalize_payload_text

class TestPayloadFormatter:
    @pytest.mark.parametrize(
        "input_text, expected_output",
        [
            ("Normal text.", "Normal text."),
            ("   Too   much   space   ", "Too much space"),
            ("Line 1\nLine 2\tTabbed", "Line 1 Line 2 Tabbed"),
            ("Hidden\u200bSpace\ufeffHere", "HiddenSpaceHere"),  # Zero-width spaces removed
            ("Ligature ﬁ", "Ligature fi"),  # NFKC normalization
            (None, ""),
            ("", ""),
        ]
    )
    def test_normalize_payload_text(self, input_text, expected_output):
        """Verifies that text is properly normalized for BM25 and Dense embeddings."""
        result = normalize_payload_text(input_text)
        assert result == expected_output