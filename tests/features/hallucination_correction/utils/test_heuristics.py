# Canvas: test_correction_heuristics_v1
# Path: tests/features/hallucination_correction/utils/test_heuristics.py

import uuid
import pytest

from app.features.transcription.domain.entities import TranscriptSegment
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.hallucination_correction.domain.entities import CorrectionConfig
from app.features.hallucination_correction.utils.heuristics import CorrectionHeuristicsEngine
from app.features.hallucination_correction.exceptions.validation import CorruptSegmentDataError


class TestCorrectionHeuristicsEngine:
    @pytest.fixture
    def config(self):
        return CorrectionConfig()

    @pytest.fixture
    def vad_map(self):
        # Human speaking from 1.0s to 3.0s, and 5.0s to 6.0s
        return VadMap(
            source_id=uuid.uuid4(),
            segments=[
                VadSegment(start_sec=1.0, end_sec=3.0, confidence=0.9),
                VadSegment(start_sec=5.0, end_sec=6.0, confidence=0.9)
            ]
        )

    def test_evaluate_segment_survives_gauntlet(self, vad_map, config):
        """Valid segment inside a VAD block with good metadata."""
        segment = TranscriptSegment(
            start_sec=1.5,
            end_sec=2.5,
            text="Hello world.",
            no_speech_prob=0.1,
            avg_logprob=-0.2,
            compression_ratio=1.2
        )
        is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)
        assert is_hallucination is False
        assert reason == ""

    def test_evaluate_segment_corrupt_timestamps(self, vad_map, config):
        """Raises error if math is impossible."""
        segment = TranscriptSegment(start_sec=2.0, end_sec=1.0, text="Bad math")
        with pytest.raises(CorruptSegmentDataError):
            CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)

    def test_evaluate_segment_fails_vad_intersection(self, vad_map, config):
        """Segment falls completely inside the dead air gap (3.0s to 5.0s)."""
        segment = TranscriptSegment(start_sec=3.5, end_sec=4.5, text="I am a ghost.")
        is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)
        assert is_hallucination is True
        assert reason == "vad"

    def test_evaluate_segment_fails_no_speech_prob(self, vad_map, config):
        """Whisper admits it is likely silence."""
        segment = TranscriptSegment(start_sec=1.5, end_sec=2.5, text="...", no_speech_prob=0.99)
        is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)
        assert is_hallucination is True
        assert reason == "metadata"

    def test_evaluate_segment_fails_compression_ratio(self, vad_map, config):
        """Whisper stuck in a stutter loop."""
        segment = TranscriptSegment(start_sec=1.5, end_sec=2.5, text="Thank you thank you thank you", compression_ratio=3.5)
        is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)
        assert is_hallucination is True
        assert reason == "metadata"

    @pytest.mark.parametrize("bad_text", [
        "Please like and subscribe to my channel.",
        "Subtitles by Amara.org",
        "[Music]",
        "MBC Network"
    ])
    def test_evaluate_segment_fails_regex(self, vad_map, config, bad_text):
        """Matches global blacklisted training data leakage."""
        segment = TranscriptSegment(start_sec=1.5, end_sec=2.5, text=bad_text)
        is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(segment, vad_map, config)
        assert is_hallucination is True
        assert reason == "regex"