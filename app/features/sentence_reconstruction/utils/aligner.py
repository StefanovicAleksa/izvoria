# Canvas: aligner_v1
# Path: app/features/sentence_reconstruction/utils/aligner.py

import uuid
from typing import List

from app.features.transcription.domain.entities import TranscriptSegment
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.sentence_reconstruction.exceptions.alignment import (
    SegmentAlignmentError,
    ZeroDurationSegmentError,
    SentenceLengthMismatchError
)


def align_and_slice_sentences(
        original_segments: List[TranscriptSegment],
        split_sentences: List[str]
) -> List[ReconstructedSentence]:
    """
    Maps grammatically complete sentences back to the original acoustic Whisper segments.
    Uses Proportional Linear Interpolation for timestamps.
    """
    if not original_segments or not split_sentences:
        return []

    # 1. Enforce strict pipeline dependency (Diarization MUST be present)
    speaker_id = original_segments[0].speaker_id
    if not speaker_id:
        raise SegmentAlignmentError(
            "Missing speaker_id. Diarization phase must complete successfully "
            "before Sentence Reconstruction can be executed."
        )

    # 2. Establish absolute time boundaries
    block_start = original_segments[0].start_sec
    block_end = original_segments[-1].end_sec
    total_duration = block_end - block_start

    if total_duration <= 0.0:
        raise ZeroDurationSegmentError(
            "Cannot virtual-slice an acoustic block with zero or negative duration."
        )

    # 3. Calculate Character Math
    # We use the total length of the newly generated sentences as the denominator
    # because the neural model may have injected or removed punctuation.
    total_new_chars = sum(len(s) for s in split_sentences)

    if total_new_chars == 0:
        raise SentenceLengthMismatchError("The neural model returned empty text. Cannot align.")

    reconstructed_sentences: List[ReconstructedSentence] = []
    current_time = block_start

    # 4. Interpolate and Slice
    for sentence_text in split_sentences:
        sentence_len = len(sentence_text)

        # Proportional Time Allocation
        ratio = sentence_len / total_new_chars
        allocated_duration = total_duration * ratio

        sentence_start = current_time
        sentence_end = current_time + allocated_duration

        # Identify overlapping original acoustic segments for the UI audit trail
        # Overlap condition: segment starts before sentence ends AND segment ends after sentence starts
        nested_segments = [
            seg for seg in original_segments
            if seg.start_sec < sentence_end and seg.end_sec > sentence_start
        ]

        reconstructed_sentences.append(ReconstructedSentence(
            sentence_id=uuid.uuid4(),
            text=sentence_text,
            speaker_id=speaker_id,
            start_sec=round(sentence_start, 3),
            end_sec=round(sentence_end, 3),
            segments=nested_segments
        ))

        # Advance the temporal cursor
        current_time = sentence_end

    return reconstructed_sentences