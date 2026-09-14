# Path: tests/features/contextual_retrieval/utils/test_transcript_formatter.py

import uuid
import pytest

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.contextual_retrieval.utils.transcript_formatter import (
    format_timestamp,
    format_speaker_turns,
    ordered_speaker_labels
)


def _sentence(text: str, speaker: str, start: float) -> ReconstructedSentence:
    return ReconstructedSentence(sentence_id=uuid.uuid4(), text=text, speaker_id=speaker,
                                 start_sec=start, end_sec=start + 1)


class TestTranscriptFormatter:
    @pytest.mark.parametrize("seconds, expected", [
        (0.0, "00:00:00"),
        (751.9, "00:12:31"),
        (3723.0, "01:02:03"),
        (-4.0, "00:00:00"),
    ])
    def test_format_timestamp(self, seconds, expected):
        assert format_timestamp(seconds) == expected

    def test_consecutive_same_speaker_sentences_merge_into_one_turn(self):
        sentences = [
            _sentence("Did you see him?", "Speaker_0", 751.0),
            _sentence("Take your time.", "Speaker_0", 753.0),
            _sentence("Yes.", "Speaker_1", 756.0),
            _sentence("  ", "Speaker_1", 757.0),
            _sentence("Thank you.", "Speaker_0", 758.0),
        ]

        assert format_speaker_turns(sentences) == (
            "[Speaker_0 @ 00:12:31]: Did you see him? Take your time.\n"
            "[Speaker_1 @ 00:12:36]: Yes.\n"
            "[Speaker_0 @ 00:12:38]: Thank you."
        )

    def test_empty_input(self):
        assert format_speaker_turns([]) == ""

    def test_ordered_speaker_labels_first_appearance_without_blanks(self):
        sentences = [
            _sentence("A.", "Speaker_1", 0), _sentence("B.", "Speaker_0", 1),
            _sentence("C.", "Speaker_1", 2), _sentence(" ", "Speaker_2", 3)
        ]
        assert ordered_speaker_labels(sentences) == ["Speaker_1", "Speaker_0"]

    def test_plain_turns_without_timestamps(self):
        sentences = [_sentence("Hello.", "Speaker_0", 5.0), _sentence("Hi.", "Speaker_1", 6.0)]
        assert format_speaker_turns(sentences, include_timestamps=False) == "Speaker_0: Hello.\nSpeaker_1: Hi."
