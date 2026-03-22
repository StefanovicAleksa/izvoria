# Canvas: smart_sampler_v4
# Path: app/features/domain_adaptation/utils/smart_sampler.py

from typing import List
from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.entities import TimeSlice
from app.features.domain_adaptation.exceptions.process import SmartSamplingError


class SmartSampler:
    """
    Intelligently samples the VAD Map to create a 'Trailer' of the media.
    Captures Start, Middle, and End zones to ensure full domain coverage
    without overflowing the LLM context window.
    """

    def __init__(self, max_speech_per_zone_sec: float = 120.0, min_duration_sec: float = 2.0):
        self.max_speech_per_zone_sec = max_speech_per_zone_sec
        self.min_duration_sec = min_duration_sec

    def calculate_optimal_slices(self, vad_map: VadMap) -> List[TimeSlice]:
        if not vad_map or not vad_map.segments:
            raise SmartSamplingError("Cannot sample audio: VAD Map is empty.")

        # Filter viable segments
        valid_segments = [
            s for s in vad_map.segments
            if (s.end_sec - s.start_sec) >= self.min_duration_sec
        ]

        if not valid_segments:
            valid_segments = vad_map.segments

        total_file_duration = vad_map.segments[-1].end_sec

        # If file is short, return everything
        if total_file_duration <= (self.max_speech_per_zone_sec * 3):
            return [TimeSlice(start_sec=s.start_sec, end_sec=s.end_sec) for s in valid_segments]

        mid_point = total_file_duration / 2.0
        start_segs, mid_segs, end_segs = [], [], []
        start_acc, mid_acc, end_acc = 0.0, 0.0, 0.0

        for seg in valid_segments:
            duration = seg.end_sec - seg.start_sec

            # Start Zone (<33%)
            if start_acc < self.max_speech_per_zone_sec and seg.start_sec < (total_file_duration * 0.33):
                start_segs.append(seg)
                start_acc += duration
                continue

            # Middle Zone (~50%)
            if mid_acc < self.max_speech_per_zone_sec and abs(seg.start_sec - mid_point) < (total_file_duration * 0.15):
                mid_segs.append(seg)
                mid_acc += duration
                continue

            # End Zone (>66%)
            if end_acc < self.max_speech_per_zone_sec and seg.start_sec > (total_file_duration * 0.66):
                end_segs.append(seg)
                end_acc += duration
                continue

        combined = sorted(list(set(start_segs + mid_segs + end_segs)), key=lambda x: x.start_sec)
        return [TimeSlice(start_sec=s.start_sec, end_sec=s.end_sec) for s in combined]