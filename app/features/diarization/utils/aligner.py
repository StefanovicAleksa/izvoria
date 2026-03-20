# Canvas: diarization_aligner_v2
# Path: app/features/diarization/utils/aligner.py

from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.diarization.domain.entities import DiarizationMap


def align_transcript_with_diarization(transcript: Transcript, diarization_map: DiarizationMap) -> Transcript:
    """
    Mathematically aligns the 'What' (Whisper text) with the 'Who' (NeMo speakers).
    Calculates the maximum time overlap between text segments and speaker segments.
    """
    if not transcript.segments:
        return transcript

    aligned_segments = []

    for t_seg in transcript.segments:
        best_speaker = "Unknown"
        max_overlap_duration = 0.0

        for d_seg in diarization_map.segments:
            # 1. Calculate the intersection (overlap) window
            overlap_start = max(t_seg.start_sec, d_seg.start_sec)
            overlap_end = min(t_seg.end_sec, d_seg.end_sec)

            # 2. If overlap_end > overlap_start, the segments intersect
            overlap_duration = overlap_end - overlap_start

            if overlap_duration > max_overlap_duration:
                max_overlap_duration = overlap_duration
                best_speaker = d_seg.speaker_id

        # 3. Create a new immutable segment with the identified speaker
        aligned_segments.append(
            TranscriptSegment(
                start_sec=t_seg.start_sec,
                end_sec=t_seg.end_sec,
                text=t_seg.text,
                speaker_id=best_speaker,
                words=t_seg.words  # CRITICAL FIX: Retain the word-level timestamp array
            )
        )

    # 4. Return a freshly reconstructed Transcript entity
    return Transcript(
        id=transcript.id,
        source_id=transcript.source_id,
        segments=aligned_segments,
        created_at=transcript.created_at
    )