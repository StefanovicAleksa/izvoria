# Canvas: smart_sampler_v1
# Path: app/features/domain_adaptation/utils/smart_sampler.py

from typing import List

from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.entities import TimeSlice
from app.features.domain_adaptation.exceptions.process import SmartSamplingError


class SmartSampler:
    """
    Pure utility to calculate the most contextually dense audio blocks
    based on VAD segment durations.
    """

    def __init__(self, max_slices: int = 10, min_duration_sec: float = 5.0):
        self.max_slices = max_slices
        self.min_duration_sec = min_duration_sec

    def calculate_optimal_slices(self, vad_map: VadMap) -> List[TimeSlice]:
        """
        Scans the VAD map and returns the `max_slices` longest continuous speech blocks.
        """
        if not vad_map or not vad_map.segments:
            raise SmartSamplingError("Cannot sample audio: VAD Map is empty or invalid.")

        valid_slices = []

        # 1. Filter out micro-segments and calculate durations
        for segment in vad_map.segments:
            duration = segment.end_sec - segment.start_sec
            if duration >= self.min_duration_sec:
                valid_slices.append({
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "duration": duration,
                    "confidence": segment.confidence
                })

        if not valid_slices:
            raise SmartSamplingError(f"No speech segments longer than {self.min_duration_sec}s found.")

        # 2. Sort by duration (descending) to find the densest blocks of speech
        valid_slices.sort(key=lambda x: x["duration"], reverse=True)

        # 3. Take the top N slices
        top_slices = valid_slices[:self.max_slices]

        # 4. Re-sort chronologically so the LLM reads the transcript in order
        top_slices.sort(key=lambda x: x["start_sec"])

        return [
            TimeSlice(start_sec=s["start_sec"], end_sec=s["end_sec"])
            for s in top_slices
        ]