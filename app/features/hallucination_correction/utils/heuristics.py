# Canvas: hallucination_correction_heuristics_v1
# Path: app/features/hallucination_correction/utils/heuristics.py

import re
from typing import Tuple

from app.features.transcription.domain.entities import TranscriptSegment
from app.features.vad.domain.entities import VadMap
from app.features.hallucination_correction.domain.entities import CorrectionConfig
from app.features.hallucination_correction.exceptions.validation import CorruptSegmentDataError


class CorrectionHeuristicsEngine:
    """
    A pure mathematical and pattern-matching engine designed to detect Whisper
    hallucinations without loading any additional AI models into VRAM.
    """

    @staticmethod
    def evaluate_segment(
            segment: TranscriptSegment,
            vad_map: VadMap,
            config: CorrectionConfig
    ) -> Tuple[bool, str]:
        """
        Evaluates a single transcript segment through the 3-step gauntlet.
        Returns (is_hallucination: bool, reason: str).
        If is_hallucination is False, the reason string is empty.
        """
        segment_duration = segment.end_sec - segment.start_sec
        if segment_duration <= 0:
            raise CorruptSegmentDataError(f"Segment has impossible duration: {segment_duration}s")

        # ==========================================
        # STEP 1: VAD STRICT INTERSECTION
        # ==========================================
        overlap_sum = 0.0

        for vad_seg in vad_map.segments:
            # Optimization: VAD segments are chronological. If we pass the transcript segment, break.
            if vad_seg.start_sec >= segment.end_sec:
                break

            # If the VAD segment ends before our transcript segment starts, keep searching.
            if vad_seg.end_sec <= segment.start_sec:
                continue

            # Calculate intersection
            overlap_start = max(segment.start_sec, vad_seg.start_sec)
            overlap_end = min(segment.end_sec, vad_seg.end_sec)

            if overlap_end > overlap_start:
                overlap_sum += (overlap_end - overlap_start)

        overlap_ratio = overlap_sum / segment_duration

        if overlap_ratio < config.min_vad_overlap_ratio:
            return True, "vad"

        # ==========================================
        # STEP 2: WHISPER METADATA MATH
        # ==========================================
        # Deleting on any single signal discarded real but hard-to-decode speech (accents, jargon,
        # poor audio). Only delete when low confidence coincides with silence or a repetition loop.
        low_logprob = segment.avg_logprob is not None and segment.avg_logprob < config.min_avg_logprob
        likely_silence = segment.no_speech_prob is not None and segment.no_speech_prob > config.max_no_speech_prob
        repetition_loop = segment.compression_ratio is not None and segment.compression_ratio > config.max_compression_ratio

        if low_logprob and (likely_silence or repetition_loop):
            return True, "metadata"

        # ==========================================
        # STEP 3: REGEX BLACKLIST
        # ==========================================
        text_to_check = segment.text.strip()
        for pattern in config.blacklist_patterns:
            if re.search(pattern, text_to_check):
                return True, "regex"

        # Survived the gauntlet
        return False, ""

    @staticmethod
    def is_low_confidence(segment: TranscriptSegment, config: CorrectionConfig) -> bool:
        """True when any single Whisper confidence signal crosses its threshold."""
        return (
            (segment.no_speech_prob is not None and segment.no_speech_prob > config.max_no_speech_prob)
            or (segment.avg_logprob is not None and segment.avg_logprob < config.min_avg_logprob)
            or (segment.compression_ratio is not None and segment.compression_ratio > config.max_compression_ratio)
        )