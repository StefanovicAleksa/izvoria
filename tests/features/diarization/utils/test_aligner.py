# Canvas: test_diarization_aligner_v3
# Path: tests/features/diarization/utils/test_aligner.py

import uuid
from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.utils.aligner import align_transcript_with_diarization


def _word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(start_sec=start, end_sec=end, word=text, probability=0.9)


def test_aligner_exact_match_and_retains_words():
    """When the text and speaker have the exact same timestamps, the word-level data is carried forward."""
    source_id = uuid.uuid4()
    test_words = [_word("Hello", 1.0, 3.0)]

    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=1.0, end_sec=3.0, text="Hello", words=test_words)]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[DiarizationSegment(start_sec=1.0, end_sec=3.0, speaker_id="Speaker_0")]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert len(result.segments) == 1
    assert result.segments[0].speaker_id == "Speaker_0"
    assert result.segments[0].text == "Hello"
    assert result.segments[0].words == test_words


def test_aligner_segment_without_words_uses_longest_overlap():
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=1.0, end_sec=4.0, text="Yes, I did.")]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[
            DiarizationSegment(start_sec=0.0, end_sec=2.0, speaker_id="Speaker_0"),  # Overlaps by 1.0s
            DiarizationSegment(start_sec=2.0, end_sec=5.0, speaker_id="Speaker_1")   # Overlaps by 2.0s (Winner)
        ]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Speaker_1"


def test_aligner_splits_segment_at_mid_segment_speaker_change():
    """A Whisper segment holding a question and its answer must not attribute the answer to the asker."""
    source_id = uuid.uuid4()
    words = [_word(" Did", 0.0, 0.4), _word(" you", 0.4, 0.8), _word(" see?", 0.8, 1.4), _word(" Yes.", 2.3, 2.8)]
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=0.0, end_sec=2.8, text="Did you see? Yes.", words=words,
                                    avg_logprob=-0.3, no_speech_prob=0.05)]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[
            DiarizationSegment(start_sec=0.0, end_sec=2.0, speaker_id="Speaker_0"),
            DiarizationSegment(start_sec=2.0, end_sec=3.0, speaker_id="Speaker_1")
        ]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert [(s.speaker_id, s.text, s.start_sec, s.end_sec) for s in result.segments] == [
        ("Speaker_0", "Did you see?", 0.0, 1.4),
        ("Speaker_1", "Yes.", 2.3, 2.8),
    ]
    assert result.segments[0].words == words[:3]
    assert result.segments[1].words == words[3:]
    assert result.segments[1].avg_logprob == -0.3


def test_aligner_word_in_diarization_gap_takes_nearest_speaker():
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=10.0, end_sec=12.0, text="Noise", words=[_word("Noise", 10.0, 12.0)])]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[
            DiarizationSegment(start_sec=1.0, end_sec=3.0, speaker_id="Speaker_0"),
            DiarizationSegment(start_sec=12.5, end_sec=14.0, speaker_id="Speaker_1")
        ]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Speaker_1"


def test_aligner_empty_diarization_map_yields_unknown():
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=10.0, end_sec=12.0, text="Noise")]
    )

    result = align_transcript_with_diarization(transcript, DiarizationMap(source_id=source_id, segments=[]))

    assert result.segments[0].speaker_id == "Unknown"


def test_aligner_handles_overlapping_diarization_segments():
    """A long earlier segment that still overlaps the word must be considered despite a shorter one in between."""
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=5.0, end_sec=6.0, text="Word", words=[_word("Word", 5.0, 6.0)])]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[
            DiarizationSegment(start_sec=0.0, end_sec=10.0, speaker_id="Speaker_0"),
            DiarizationSegment(start_sec=2.0, end_sec=3.0, speaker_id="Speaker_1"),
        ]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Speaker_0"
