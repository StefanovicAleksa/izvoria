# Canvas: test_aligner_v2
# Path: tests/features/sentence_reconstruction/test_aligner.py

import pytest
from app.features.transcription.domain.entities import TranscriptSegment
from app.features.sentence_reconstruction.utils.aligner import align_and_slice_sentences
from app.features.sentence_reconstruction.exceptions.alignment import (
    SegmentAlignmentError,
    ZeroDurationSegmentError,
    SentenceLengthMismatchError
)


def test_align_and_slice_proportional_math():
    segments = [TranscriptSegment(start_sec=0.0, end_sec=10.0, text="This is part one this is part two",
                                  speaker_id="Speaker_1")]
    split_sentences = ["This is part one.", "This is part two."]

    result = align_and_slice_sentences(segments, split_sentences)

    assert len(result) == 2
    assert result[0].start_sec == 0.0
    assert result[0].end_sec == 5.0
    assert result[1].start_sec == 5.0
    assert result[1].end_sec == 10.0
    assert result[0].speaker_id == "Speaker_1"


def test_aligner_enforces_diarization_dependency():
    segments = [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="No speaker", speaker_id=None)]

    with pytest.raises(SegmentAlignmentError, match="Missing speaker_id"):
        align_and_slice_sentences(segments, ["No speaker."])


def test_aligner_catches_zero_duration():
    segments = [TranscriptSegment(start_sec=5.0, end_sec=5.0, text="Zero", speaker_id="Speaker_1")]

    with pytest.raises(ZeroDurationSegmentError):
        align_and_slice_sentences(segments, ["Zero."])


def test_aligner_catches_empty_neural_output():
    segments = [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="Valid", speaker_id="Speaker_1")]

    with pytest.raises(SentenceLengthMismatchError):
        # FIXED: Pass absolute empty strings so total character count evaluates to 0
        align_and_slice_sentences(segments, ["", ""])