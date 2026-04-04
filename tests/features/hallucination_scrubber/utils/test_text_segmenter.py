# Canvas: test_text_segmenter_v1
# Path: tests/features/hallucination_scrubber/utils/test_text_segmenter.py

import pytest
from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
from app.features.hallucination_scrubber.utils.text_segmenter import TextSegmenter


class TestTextSegmenter:
    def test_split_into_clauses_handles_multiple_punctuation(self):
        text = "This is valid. Thank you for watching! Please subscribe."
        clauses = TextSegmenter.split_into_clauses(text)

        assert len(clauses) == 3
        assert clauses[0] == "This is valid."
        assert clauses[1] == "Thank you for watching!"
        assert clauses[2] == "Please subscribe."

    def test_split_into_clauses_empty_string(self):
        assert TextSegmenter.split_into_clauses("") == []

    def test_reconstruct_segment_salvages_correct_words(self):
        # Build a dirty segment with timestamped words
        words = [
            TranscriptWord(start_sec=1.0, end_sec=1.2, word="Valid", probability=0.9),
            TranscriptWord(start_sec=1.2, end_sec=1.5, word="financial", probability=0.9),
            TranscriptWord(start_sec=1.5, end_sec=1.8, word="data.", probability=0.9),
            TranscriptWord(start_sec=1.8, end_sec=2.0, word="Thanks", probability=0.9),
            TranscriptWord(start_sec=2.0, end_sec=2.2, word="for", probability=0.9),
            TranscriptWord(start_sec=2.2, end_sec=2.5, word="watching.", probability=0.9),
        ]
        dirty_segment = TranscriptSegment(
            start_sec=1.0, end_sec=2.5,
            text="Valid financial data. Thanks for watching.",
            speaker_id="S0", words=words
        )

        surviving = ["Valid financial data."]
        deleted = ["Thanks for watching."]

        clean_segment = TextSegmenter.reconstruct_segment(dirty_segment, surviving, deleted)

        assert clean_segment is not None
        assert clean_segment.text == "Valid financial data."
        assert len(clean_segment.words) == 3
        assert clean_segment.words[0].word == "Valid"
        assert clean_segment.words[-1].word == "data."

    def test_reconstruct_segment_full_deletion_returns_none(self):
        segment = TranscriptSegment(start_sec=1.0, end_sec=2.0, text="Please subscribe.", speaker_id="S0", words=[])

        clean_segment = TextSegmenter.reconstruct_segment(segment, surviving_clauses=[],
                                                          deleted_clauses=["Please subscribe."])

        assert clean_segment is None