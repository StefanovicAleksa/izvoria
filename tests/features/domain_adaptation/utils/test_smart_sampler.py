# Canvas: test_smart_sampler_v3
# Path: tests/features/domain_adaptation/utils/test_smart_sampler.py

import uuid
import pytest
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.exceptions.process import SmartSamplingError


class TestSmartSampler:
    def test_calculate_optimal_slices_extracts_trailer(self):
        # Target 10 seconds max per zone (Start, Middle, End)
        sampler = SmartSampler(max_speech_per_zone_sec=10.0, min_duration_sec=1.0)
        mock_vad = VadMap(
            source_id=uuid.uuid4(),
            segments=[
                VadSegment(start_sec=0.0, end_sec=5.0, confidence=0.9),   # Start Zone (5s)
                VadSegment(start_sec=6.0, end_sec=12.0, confidence=0.9),  # Start Zone (6s) -> Zone hits 11s total, stops
                VadSegment(start_sec=15.0, end_sec=20.0, confidence=0.9), # Start Zone (Skipped, zone full)
                VadSegment(start_sec=45.0, end_sec=50.0, confidence=0.9), # Middle Zone (5s) (Total length is 100s)
                VadSegment(start_sec=90.0, end_sec=98.0, confidence=0.9)  # End Zone (8s)
            ]
        )

        slices = sampler.calculate_optimal_slices(mock_vad)

        assert len(slices) == 4
        # Verify chronological ordering
        assert slices[0].start_sec == 0.0
        assert slices[1].start_sec == 6.0
        assert slices[2].start_sec == 45.0
        assert slices[3].start_sec == 90.0

    def test_calculate_optimal_slices_raises_error_on_empty(self):
        sampler = SmartSampler()
        mock_vad = VadMap(source_id=uuid.uuid4(), segments=[])

        with pytest.raises(SmartSamplingError):
            sampler.calculate_optimal_slices(mock_vad)