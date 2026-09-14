# Canvas: test_faster_whisper_adapter_v7
# Path: tests/features/transcription/data/test_faster_whisper_adapter.py

import uuid
import pytest
import torch
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.data.faster_whisper_adapter import (
    FasterWhisperAdapter,
    _plan_batched_clips,
    _widen_to_min_duration
)
from app.features.vad.domain.entities import VadMap, VadSegment

ADAPTER = "app.features.transcription.data.faster_whisper_adapter"


def _vad(*spans) -> VadMap:
    return VadMap(source_id=uuid.uuid4(), segments=[VadSegment(start_sec=s, end_sec=e, confidence=0.9) for s, e in spans])


def _raw_segment(start, end, text="Hello world", words=None, **metadata):
    seg = MagicMock()
    seg.start, seg.end, seg.text = start, end, text
    seg.words = words or []
    seg.no_speech_prob = metadata.get("no_speech_prob", 0.1)
    seg.avg_logprob = metadata.get("avg_logprob", -0.2)
    seg.compression_ratio = metadata.get("compression_ratio", 1.2)
    return seg


class TestClipPlanning:
    def test_packs_segments_across_short_gaps_up_to_the_clip_limit(self):
        clips, sequential = _plan_batched_clips(_vad((0, 10), (11, 20), (25, 60), (61, 62)).segments)

        assert clips == [(0, 20), (61, 62)]
        assert sequential == [(25, 60)]

    def test_starts_new_clip_when_limit_would_be_exceeded(self):
        clips, sequential = _plan_batched_clips(_vad((0, 15), (16, 29), (30, 40)).segments)

        assert clips == [(0, 29), (30, 40)]
        assert sequential == []

    def test_starts_new_clip_after_long_silence(self):
        clips, _ = _plan_batched_clips(_vad((0, 5), (8, 10)).segments)
        assert clips == [(0, 5), (8, 10)]

    @pytest.mark.parametrize("span, total, expected", [
        ((0.0, 0.2), 10.0, (0.0, 1.0)),
        ((9.9, 10.0), 10.0, (9.0, 10.0)),
        ((4.0, 4.5), 10.0, (3.75, 4.75)),
        ((2.0, 6.0), 10.0, (2.0, 6.0)),
    ])
    def test_widen_to_min_duration(self, span, total, expected):
        assert _widen_to_min_duration(span[0], span[1], total, 1.0) == pytest.approx(expected)


class TestFasterWhisperAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="/mock/path/faster-whisper-large-v3",
            required_vram_mb=4000.0
        )

    @pytest.fixture
    def mock_settings(self):
        with patch(f"{ADAPTER}.settings") as settings:
            settings.WHISPER_BATCH_SIZE = 1
            settings.WHISPER_MIN_SLICE_SEC = 1.0
            yield settings

    @patch("os.path.exists", return_value=True)
    @patch.object(FasterWhisperAdapter, "_get_whisper_model")
    def test_load_success(self, mock_get_whisper, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_whisper.return_value = mock_instance

        loaded_model = FasterWhisperAdapter().load(manifest)

        assert loaded_model.model_instance == mock_instance
        mock_get_whisper.assert_called_once_with(manifest.model_path)

    @patch(f"{ADAPTER}.torchaudio")
    def test_short_segments_are_widened_not_skipped(self, mock_torchaudio, manifest, mock_settings):
        """An isolated 0.1s "Yes." used to be skipped (< 6400 samples) and silently lost."""
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 10), 16000)
        whisper = MagicMock()
        whisper.transcribe.return_value = ([], MagicMock())

        FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((0.0, 0.1), (5.0, 5.5)),
            language="en"
        )

        assert whisper.transcribe.call_count == 2
        assert [len(c.kwargs["audio"]) for c in whisper.transcribe.call_args_list] == [16000, 16000]

    @patch(f"{ADAPTER}.torchaudio")
    def test_sequential_path_offsets_timestamps_and_extracts_metadata(self, mock_torchaudio, manifest, mock_settings):
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 10), 16000)
        whisper = MagicMock()
        word = MagicMock(start=0.1, end=0.4, word=" Hello", probability=0.95)
        whisper.transcribe.return_value = (
            [_raw_segment(0.0, 1.0, words=[word], no_speech_prob=0.95, avg_logprob=-2.0, compression_ratio=3.1)],
            MagicMock()
        )

        segments_out = FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((2.0, 3.5)),
            language="en",
            hotwords="strike, theta",
            initial_prompt="This recording discusses Options Trading."
        )

        assert len(segments_out) == 1
        seg = segments_out[0]
        assert (seg.start_sec, seg.end_sec) == (2.0, 3.0)
        assert (seg.words[0].start_sec, seg.words[0].end_sec) == (2.1, 2.4)
        assert (seg.no_speech_prob, seg.avg_logprob, seg.compression_ratio) == (0.95, -2.0, 3.1)
        kwargs = whisper.transcribe.call_args.kwargs
        assert kwargs["hotwords"] == "strike, theta"
        assert kwargs["initial_prompt"] == "This recording discusses Options Trading."
        assert kwargs["word_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False

    @patch.object(FasterWhisperAdapter, "_get_batched_pipeline")
    @patch(f"{ADAPTER}.torchaudio")
    def test_batched_path_with_sequential_fallback_for_long_speech(
            self, mock_torchaudio, mock_get_pipeline, manifest, mock_settings
    ):
        mock_settings.WHISPER_BATCH_SIZE = 8
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 70), 16000)

        whisper = MagicMock()
        whisper.transcribe.return_value = ([_raw_segment(0.0, 5.0, text="long speech")], MagicMock())
        pipeline = MagicMock()
        pipeline.transcribe.return_value = (
            [_raw_segment(61.2, 61.8, text="later"), _raw_segment(0.5, 4.0, text="first")], MagicMock()
        )
        mock_get_pipeline.return_value = pipeline

        segments_out = FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((0, 10), (11, 20), (25, 60), (61, 62)),
            language="en"
        )

        mock_get_pipeline.assert_called_once_with(whisper)
        batched_kwargs = pipeline.transcribe.call_args.kwargs
        assert batched_kwargs["clip_timestamps"] == [{"start": 0, "end": 20}, {"start": 61, "end": 62}]
        assert batched_kwargs["batch_size"] == 8
        assert batched_kwargs["vad_filter"] is False
        assert batched_kwargs["word_timestamps"] is True

        # The 35s uninterrupted segment is transcribed sequentially, offset from its own start
        assert whisper.transcribe.call_count == 1
        assert len(whisper.transcribe.call_args.kwargs["audio"]) == 35 * 16000

        # Batched timestamps are absolute; everything is returned in time order
        assert [(s.text, s.start_sec) for s in segments_out] == [("first", 0.5), ("long speech", 25.0), ("later", 61.2)]
