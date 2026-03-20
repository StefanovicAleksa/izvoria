# Canvas: test_diarization_aligner_v2
# Path: tests/features/diarization/test_aligner.py

import uuid
from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment
from app.features.diarization.utils.aligner import align_transcript_with_diarization


def test_aligner_exact_match_and_retains_words():
    """When the text and speaker have the exact same timestamps, and verifies the downstream bug is fixed."""
    source_id = uuid.uuid4()

    # Inject word-level data
    test_words = [TranscriptWord(start_sec=1.0, end_sec=3.0, word="Hello", probability=0.99)]

    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=1.0, end_sec=3.0, text="Hello", words=test_words)]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[DiarizationSegment(start_sec=1.0, end_sec=3.0, speaker_id="Speaker_0")]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Speaker_0"
    assert result.segments[0].text == "Hello"

    # Prove the bug is fixed: The word-level data was successfully carried forward
    # when recreating the immutable TranscriptSegment
    assert len(result.segments[0].words) == 1
    assert result.segments[0].words[0].word == "Hello"


def test_aligner_longest_overlap_wins():
    """When a text segment bridges two speakers, the one it overlaps with MOST wins."""
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=1.0, end_sec=4.0, text="Yes, I did.")]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[
            DiarizationSegment(start_sec=0.0, end_sec=2.0, speaker_id="Speaker_0"),  # Overlaps by 1.0s
            DiarizationSegment(start_sec=2.0, end_sec=5.0, speaker_id="Speaker_1")  # Overlaps by 2.0s (Winner)
        ]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Speaker_1"


def test_aligner_no_overlap_defaults_to_unknown():
    """If there is no mathematical intersection, it safely falls back to 'Unknown'."""
    source_id = uuid.uuid4()
    transcript = Transcript(
        source_id=source_id,
        segments=[TranscriptSegment(start_sec=10.0, end_sec=12.0, text="Noise")]
    )
    diarization_map = DiarizationMap(
        source_id=source_id,
        segments=[DiarizationSegment(start_sec=1.0, end_sec=3.0, speaker_id="Speaker_0")]
    )

    result = align_transcript_with_diarization(transcript, diarization_map)

    assert result.segments[0].speaker_id == "Unknown"