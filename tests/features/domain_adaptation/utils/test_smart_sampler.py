# Canvas: test_smart_sampler_v2
# Path: tests/features/domain_adaptation/utils/test_smart_sampler.py

import uuid
import pytest
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
from app.features.domain_adaptation.exceptions.process import SmartSamplingError


class TestSmartSampler:
    def test_calculate_optimal_slices_finds_densest_blocks(self):
        sampler = SmartSampler(max_slices=2, min_duration_sec=2.0)
        mock_vad = VadMap(
            source_id=uuid.uuid4(),
            segments=[
                VadSegment(start_sec=10.0, end_sec=11.0, confidence=0.9),  # 1s (Ignored)
                VadSegment(start_sec=20.0, end_sec=30.0, confidence=0.9),  # 10s (Dense)
                VadSegment(start_sec=40.0, end_sec=45.0, confidence=0.9),  # 5s  (Less Dense)
                VadSegment(start_sec=50.0, end_sec=70.0, confidence=0.9)   # 20s (Densest)
            ]
        )

        slices = sampler.calculate_optimal_slices(mock_vad)

        # Should return max 2 slices, skipping the 1s and 5s blocks
        assert len(slices) == 2
        # Chronological ordering test
        assert slices[0].start_sec == 20.0
        assert slices[0].end_sec == 30.0
        assert slices[1].start_sec == 50.0
        assert slices[1].end_sec == 70.0

    def test_calculate_optimal_slices_raises_error_on_empty(self):
        sampler = SmartSampler()
        mock_vad = VadMap(source_id=uuid.uuid4(), segments=[])

        with pytest.raises(SmartSamplingError):
            sampler.calculate_optimal_slices(mock_vad)