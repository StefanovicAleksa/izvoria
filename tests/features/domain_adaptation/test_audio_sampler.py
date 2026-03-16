# Canvas: test_audio_sampler_v1
# Path: tests/features/domain_adaptation/test_audio_sampler.py

import uuid
from pathlib import Path
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.domain_adaptation.utils.audio_sampler import AudioSampler


def test_extract_samples_bypasses_silent_gaps(mocker):
    # Arrange
    sampler = AudioSampler(slice_duration_sec=20.0)
    mock_vad = VadMap(
        source_id=uuid.uuid4(),
        segments=[
            VadSegment(start_sec=10.0, end_sec=50.0, confidence=0.9),
            VadSegment(start_sec=100.0, end_sec=150.0, confidence=0.9),
            VadSegment(start_sec=200.0, end_sec=250.0, confidence=0.9)
        ]
    )

    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("app.features.domain_adaptation.utils.audio_sampler.AudioSampler._slice_audio_ffmpeg",
                 return_value=Path("/tmp/mock.wav"))

    # Act
    samples = sampler.extract_samples(Path("/mock/video.mp4"), mock_vad)

    # Assert
    assert len(samples) == 3
    # 15% of 250s = 37.5s (Falls in segment 1)
    assert samples[0].start_sec == 10.0
    # 50% of 250s = 125s (Falls in segment 2)
    assert samples[1].start_sec == 100.0
    # 85% of 250s = 212.5s (Falls in segment 3)
    assert samples[2].start_sec == 200.0


def test_extract_samples_ignores_micro_slices(mocker):
    sampler = AudioSampler(slice_duration_sec=20.0)
    mock_vad = VadMap(
        source_id=uuid.uuid4(),
        segments=[VadSegment(start_sec=10.0, end_sec=11.0, confidence=0.9)]  # Only 1s long
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    samples = sampler.extract_samples(Path("/mock/video.mp4"), mock_vad)
    assert len(samples) == 0  # Micro-slice skipped