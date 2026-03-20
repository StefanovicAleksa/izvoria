# Canvas: nlp_router_chunker_v1
# Path: app/features/nlp_router/utils/chunker.py

from typing import List
from app.features.transcription.domain.entities import Transcript
from app.legacy_features.nlp_router.domain.entities import TranscriptChunk


def chunk_transcript(transcript: Transcript, max_duration_sec: float = 180.0) -> List[TranscriptChunk]:
    """
    Safely splits a Whisper Transcript into manageable blocks (default 3 minutes)
    to prevent overflowing the LLM's token context window.
    """
    if not transcript.segments:
        return []

    chunks = []
    current_text = ""
    chunk_start = transcript.segments[0].start_sec

    for seg in transcript.segments:
        # Fallback to "Speaker" if diarization hasn't run or failed
        speaker = seg.speaker_id if seg.speaker_id else "Speaker"

        # Format the text so the LLM easily understands the timeline and who is talking
        line = f"[{seg.start_sec:.1f}s - {seg.end_sec:.1f}s] {speaker}: {seg.text}\n"
        current_text += line

        # If this block has reached our target duration, package it up and start a new one
        if seg.end_sec - chunk_start >= max_duration_sec:
            chunks.append(TranscriptChunk(
                start_sec=chunk_start,
                end_sec=seg.end_sec,
                text_block=current_text.strip()
            ))
            current_text = ""
            chunk_start = seg.end_sec

    # Catch the remaining trailing text
    if current_text:
        chunks.append(TranscriptChunk(
            start_sec=chunk_start,
            end_sec=transcript.segments[-1].end_sec,
            text_block=current_text.strip()
        ))

    return chunks