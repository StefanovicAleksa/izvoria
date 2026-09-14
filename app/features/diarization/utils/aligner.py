# Canvas: diarization_aligner_v3
# Path: app/features/diarization/utils/aligner.py

import bisect
from typing import List, Tuple

from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.diarization.domain.entities import DiarizationMap, DiarizationSegment


class _SpeakerIndex:
    """Sorted diarization segments with a prefix-max of end times for fast overlap lookups."""

    def __init__(self, segments: List[DiarizationSegment]):
        self._segments = sorted(segments, key=lambda s: s.start_sec)
        self._starts = [s.start_sec for s in self._segments]
        self._max_end_prefix: List[float] = []
        running = float("-inf")
        for seg in self._segments:
            running = max(running, seg.end_sec)
            self._max_end_prefix.append(running)

    def speaker_for_span(self, start_sec: float, end_sec: float) -> str:
        """
        Speaker with the maximum time overlap with [start_sec, end_sec]. If nothing overlaps, the
        nearest speaker segment wins, so a word that falls in a tiny diarization gap still gets a
        speaker. "Unknown" is only returned when the diarization map is empty.
        """
        if not self._segments:
            return "Unknown"

        idx = bisect.bisect_right(self._starts, end_sec)
        best_speaker, best_overlap = None, 0.0

        j = idx - 1
        while j >= 0 and self._max_end_prefix[j] > start_sec:
            seg = self._segments[j]
            overlap = min(end_sec, seg.end_sec) - max(start_sec, seg.start_sec)
            if overlap > best_overlap:
                best_speaker, best_overlap = seg.speaker_id, overlap
            j -= 1

        if best_speaker is not None:
            return best_speaker

        midpoint = (start_sec + end_sec) / 2.0
        candidates = [self._segments[k] for k in (idx - 1, idx) if 0 <= k < len(self._segments)]
        nearest = min(candidates, key=lambda s: max(s.start_sec - midpoint, midpoint - s.end_sec, 0.0))
        return nearest.speaker_id


def _speaker_runs(words: List[TranscriptWord], index: _SpeakerIndex) -> List[Tuple[str, List[TranscriptWord]]]:
    runs: List[Tuple[str, List[TranscriptWord]]] = []
    for word in words:
        speaker = index.speaker_for_span(word.start_sec, word.end_sec)
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(word)
        else:
            runs.append((speaker, [word]))
    return runs


def align_transcript_with_diarization(transcript: Transcript, diarization_map: DiarizationMap) -> Transcript:
    """
    Aligns the 'What' (Whisper text) with the 'Who' (NeMo speakers) at WORD level.

    Whisper segments often span a speaker change (a quick question and its answer). Each word is
    attributed to the speaker it overlaps most, and a segment containing a speaker change is split
    into one segment per speaker run, so no word is attributed to the wrong person. Segments
    without word timestamps fall back to segment-level attribution.
    """
    if not transcript.segments:
        return transcript

    index = _SpeakerIndex(diarization_map.segments)
    aligned_segments: List[TranscriptSegment] = []

    for t_seg in transcript.segments:
        if not t_seg.words:
            speaker = index.speaker_for_span(t_seg.start_sec, t_seg.end_sec)
            aligned_segments.append(t_seg.model_copy(update={"speaker_id": speaker}))
            continue

        runs = _speaker_runs(t_seg.words, index)

        if len(runs) == 1:
            aligned_segments.append(t_seg.model_copy(update={"speaker_id": runs[0][0]}))
            continue

        for speaker, words in runs:
            aligned_segments.append(t_seg.model_copy(update={
                "speaker_id": speaker,
                "start_sec": words[0].start_sec,
                "end_sec": words[-1].end_sec,
                "text": " ".join(w.word.strip() for w in words if w.word.strip()),
                "words": words
            }))

    return Transcript(
        id=transcript.id,
        source_id=transcript.source_id,
        segments=aligned_segments,
        created_at=transcript.created_at
    )
