# Canvas: test_speaker_grouper_v1
# Path: tests/features/sentence_reconstruction/test_speaker_grouper.py

from app.features.transcription.domain.entities import TranscriptSegment
from app.features.sentence_reconstruction.utils.speaker_grouper import group_by_speaker


def test_group_by_speaker_empty_list():
    assert group_by_speaker([]) == []


def test_group_by_speaker_single_block():
    segments = [
        TranscriptSegment(start_sec=0.0, end_sec=1.0, text="A", speaker_id="Speaker_1"),
        TranscriptSegment(start_sec=1.0, end_sec=2.0, text="B", speaker_id="Speaker_1")
    ]
    grouped = group_by_speaker(segments)

    assert len(grouped) == 1
    assert len(grouped[0]) == 2
    assert grouped[0][0].speaker_id == "Speaker_1"


def test_group_by_speaker_alternating_speakers():
    segments = [
        TranscriptSegment(start_sec=0.0, end_sec=1.0, text="A", speaker_id="Speaker_1"),
        TranscriptSegment(start_sec=1.0, end_sec=2.0, text="B", speaker_id="Speaker_2"),
        TranscriptSegment(start_sec=2.0, end_sec=3.0, text="C", speaker_id="Speaker_2"),
        TranscriptSegment(start_sec=3.0, end_sec=4.0, text="D", speaker_id="Speaker_1")
    ]
    grouped = group_by_speaker(segments)

    assert len(grouped) == 3
    assert len(grouped[0]) == 1  # Speaker 1
    assert len(grouped[1]) == 2  # Speaker 2
    assert len(grouped[2]) == 1  # Speaker 1 back again