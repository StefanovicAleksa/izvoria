# Canvas: test_payload_formatter_v1
# Path: tests/features/rag_storage/utils/test_payload_formatter.py

import uuid

import pytest
from app.features.rag_storage.utils.payload_formatter import (
    build_payload, normalize_payload_text, source_line, with_source_line
)

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

class TestSourceLine:
    def test_with_and_without_a_topic(self):
        assert source_line("Lecture 4", "Bayes.") == "[Source: Lecture 4 — Bayes.]"
        assert source_line(" Lecture 4 ") == "[Source: Lecture 4]"

    def test_with_source_line_replaces_an_existing_line(self):
        assert with_source_line("[Source: old]\n[Audio Context: a]", "[Source: new]") == \
            "[Source: new]\n[Audio Context: a]"
        assert with_source_line("", "[Source: new]") == "[Source: new]"

    def test_brackets_in_a_title_or_topic_cannot_end_the_line_early(self):
        assert source_line("Exhibit [A]", "See [Doc 2].") == "[Source: Exhibit (A) — See (Doc 2).]"

    def test_the_line_is_exactly_what_a_relevance_judge_strips(self):
        from app.features.rag_retrieval.domain.entities import RetrievedChunk
        context = with_source_line("[Audio Context: a lecture]", source_line("Lecture [4]", "Bayes [rule]."))
        payload = normalize_payload_text(build_payload(context, "The prior is updated."))
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=payload, start_sec=0,
                               end_sec=1, relevance_score=1.0)

        assert chunk.content_payload == "[Audio Context: a lecture] The prior is updated."

    def test_build_payload_matches_the_assembled_shape(self):
        assert build_payload("[Audio Context: a]", "text") == "[Audio Context: a]\n\ntext"
        assert build_payload("", "text") == "text"
