# Canvas: smart_sampler_v2
# Path: app/features/domain_adaptation/utils/smart_sampler.py

from typing import List

from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.entities import TimeSlice
from app.features.domain_adaptation.exceptions.process import SmartSamplingError


class SmartSampler:
    """
    Calculates uniform contextual audio blocks across the entire timeline
    to ensure the Scout model gets a true cross-section of the conversation.
    """

    def __init__(self, max_slices: int = 10, min_duration_sec: float = 3.0):
        self.max_slices = max_slices
        self.min_duration_sec = min_duration_sec

    def calculate_optimal_slices(self, vad_map: VadMap) -> List[TimeSlice]:
        if not vad_map or not vad_map.segments:
            raise SmartSamplingError("Cannot sample audio: VAD Map is empty or invalid.")

        # 1. Filter viable segments
        valid_segments = [
            s for s in vad_map.segments
            if (s.end_sec - s.start_sec) >= self.min_duration_sec
        ]

        if not valid_segments:
            raise SmartSamplingError(f"No speech segments longer than {self.min_duration_sec}s found.")

        # If we have fewer segments than requested slices, just return them all
        if len(valid_segments) <= self.max_slices:
            return [TimeSlice(start_sec=s.start_sec, end_sec=s.end_sec) for s in valid_segments]

        # 2. Bucket the timeline uniformly
        total_duration = valid_segments[-1].end_sec
        bucket_size = total_duration / self.max_slices

        selected_slices = []

        # 3. Pick the longest/highest-confidence segment from each bucket
        for i in range(self.max_slices):
            bucket_start = i * bucket_size
            bucket_end = (i + 1) * bucket_size

            # Find segments that fall primarily in this bucket
            bucket_segs = [
                s for s in valid_segments
                if (bucket_start <= s.start_sec < bucket_end)
            ]

            if bucket_segs:
                # Pick the densest segment in this specific time window
                best_seg = max(bucket_segs, key=lambda x: x.end_sec - x.start_sec)
                selected_slices.append(TimeSlice(start_sec=best_seg.start_sec, end_sec=best_seg.end_sec))

        # 4. Ensure chronological order for the LLM
        selected_slices.sort(key=lambda x: x.start_sec)

        return selected_slices