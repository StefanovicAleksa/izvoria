# Canvas: speaker_grouper_v2
# Path: app/features/sentence_reconstruction/utils/speaker_grouper.py

from typing import List
from app.features.transcription.domain.entities import TranscriptSegment

def group_by_speaker(segments: List[TranscriptSegment]) -> List[List[TranscriptSegment]]:
    """
    Creates a 'Speaker Wall' by grouping consecutive transcript segments
    belonging to the same speaker. This guarantees the deterministic sentence
    reconstruction never crosses speaker boundaries.
    """
    if not segments:
        return []

    grouped_blocks: List[List[TranscriptSegment]] = []
    current_block: List[TranscriptSegment] = [segments[0]]
    current_speaker = segments[0].speaker_id

    for seg in segments[1:]:
        if seg.speaker_id == current_speaker:
            current_block.append(seg)
        else:
            grouped_blocks.append(current_block)
            current_block = [seg]
            current_speaker = seg.speaker_id

    # Append the final accumulated block
    if current_block:
        grouped_blocks.append(current_block)

    return grouped_blocks