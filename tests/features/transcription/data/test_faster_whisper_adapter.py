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
    _clips_to_redecode,
    _plan_batched_clips,
    _uncovered_speech,
    _widen_to_min_duration
)
from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
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


def _said(*word_spans) -> TranscriptSegment:
    words = [TranscriptWord(start_sec=s, end_sec=e, word="w") for s, e in word_spans]
    return TranscriptSegment(start_sec=word_spans[0][0], end_sec=word_spans[-1][1], text="w", words=words)


class TestSpeechCoverage:
    @pytest.mark.parametrize("speech, words, uncovered", [
        ((0.0, 3.0), [(1.0, 1.4)], False),          # a word inside
        ((0.0, 3.0), [(3.1, 3.5)], False),          # a word just past the edge: timing drift
        ((0.0, 3.0), [(4.0, 4.5)], True),           # nothing within tolerance
        ((0.0, 0.6), [(4.0, 4.5)], False),          # too short to count as a dropped utterance
    ])
    def test_uncovered_speech(self, speech, words, uncovered):
        result = _uncovered_speech(_vad(speech).segments, [_said(*words)])
        assert (len(result) == 1) is uncovered

    def test_a_segment_without_word_timings_covers_its_own_span(self):
        segment = TranscriptSegment(start_sec=0.0, end_sec=3.0, text="yes")
        assert _uncovered_speech(_vad((0.5, 2.5)).segments, [segment]) == []

    def test_only_clips_holding_wordless_speech_are_redecoded(self):
        clips = [(0.0, 25.0), (30.0, 40.0)]
        speech = _vad((0.0, 8.0), (9.0, 17.0), (18.0, 25.0), (30.0, 40.0)).segments
        # The batched decoder returned only the last sentence of the first clip
        transcribed = [_said((19.0, 24.0)), _said((31.0, 39.0))]

        assert _clips_to_redecode(clips, speech, transcribed) == [(0.0, 25.0)]


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
    def test_detect_language_judges_only_speech_up_to_thirty_seconds(self, mock_torchaudio, manifest):
        """Silence between VAD segments is left out, and Whisper sees at most one 30 s window of speech."""
        audio = torch.arange(16000 * 60, dtype=torch.float32).unsqueeze(0)
        mock_torchaudio.load.return_value = (audio, 16000)
        whisper = MagicMock()
        whisper.detect_language.return_value = ("sr", 0.91, [("sr", 0.91)])

        detection = FasterWhisperAdapter().detect_language(
            LoadedModel(manifest=manifest, model_instance=whisper), "/mock/path.wav", _vad((1.0, 2.0), (10.0, 50.0))
        )

        assert detection == ("sr", 0.91)
        kwargs = whisper.detect_language.call_args.kwargs
        assert kwargs["language_detection_segments"] == 1
        assert len(kwargs["audio"]) == 16000 * 30
        assert kwargs["audio"][0] == 16000.0             # starts at the first speech, not at 0 s
        assert kwargs["audio"][16000] == 16000.0 * 10    # the 8 s of silence after it is skipped

    @patch(f"{ADAPTER}.torchaudio")
    def test_detect_language_without_speech_returns_none(self, mock_torchaudio, manifest):
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000), 16000)
        whisper = MagicMock()

        detection = FasterWhisperAdapter().detect_language(
            LoadedModel(manifest=manifest, model_instance=whisper), "/mock/path.wav", _vad()
        )

        assert detection is None
        whisper.detect_language.assert_not_called()

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
            [_raw_segment(61.2, 61.8, text="later"), _raw_segment(0.5, 19.0, text="first")], MagicMock()
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

    @patch.object(FasterWhisperAdapter, "_get_batched_pipeline")
    @patch(f"{ADAPTER}.torchaudio")
    def test_clip_whose_speech_came_back_wordless_is_redecoded_sequentially(
            self, mock_torchaudio, mock_get_pipeline, manifest, mock_settings
    ):
        """
        The batched decoder can return only the last sentence of a clip joining several utterances. Those
        utterances used to be lost without a trace; the clip is now decoded again and its batched output replaced.
        """
        mock_settings.WHISPER_BATCH_SIZE = 8
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 60), 16000)
        last_only = _raw_segment(19.0, 24.0, text="third", words=[MagicMock(start=19.5, end=23.5, word="third",
                                                                            probability=0.9)])
        separate = _raw_segment(40.5, 44.0, text="fourth", words=[MagicMock(start=41.0, end=43.0, word="fourth",
                                                                           probability=0.9)])
        pipeline = MagicMock()
        pipeline.transcribe.return_value = ([last_only, separate], MagicMock())
        mock_get_pipeline.return_value = pipeline
        whisper = MagicMock()
        whisper.transcribe.return_value = ([
            _raw_segment(0.5, 7.5, text="first", words=[MagicMock(start=1.0, end=7.0, word="first", probability=0.9)]),
            _raw_segment(9.5, 16.5, text="second", words=[MagicMock(start=10.0, end=16.0, word="second",
                                                                    probability=0.9)]),
            _raw_segment(18.5, 24.5, text="third", words=[MagicMock(start=19.0, end=24.0, word="third",
                                                                    probability=0.9)]),
        ], MagicMock())

        segments_out = FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((0.5, 8.0), (9.0, 17.0), (18.0, 25.0), (40.0, 45.0)),
            language="en",
            hotwords="strike"
        )

        # Only the clip that lost speech is decoded again, from its own start, with the same options
        assert whisper.transcribe.call_count == 1
        kwargs = whisper.transcribe.call_args.kwargs
        assert len(kwargs["audio"]) == int(24.5 * 16000)
        assert (kwargs["language"], kwargs["hotwords"], kwargs["word_timestamps"]) == ("en", "strike", True)
        # Its batched "third" is replaced, not duplicated; the other clip's output is kept
        assert [(s.text, s.start_sec) for s in segments_out] == [
            ("first", 1.0), ("second", 10.0), ("third", 19.0), ("fourth", 40.5)]

    @patch.object(FasterWhisperAdapter, "_get_batched_pipeline")
    @patch(f"{ADAPTER}.torchaudio")
    def test_fully_transcribed_clips_are_not_decoded_twice(
            self, mock_torchaudio, mock_get_pipeline, manifest, mock_settings
    ):
        mock_settings.WHISPER_BATCH_SIZE = 8
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 30), 16000)
        pipeline = MagicMock()
        pipeline.transcribe.return_value = ([
            _raw_segment(0.5, 7.5, words=[MagicMock(start=1.0, end=7.0, word="a", probability=0.9)]),
            _raw_segment(9.5, 16.5, words=[MagicMock(start=10.0, end=16.0, word="b", probability=0.9)]),
        ], MagicMock())
        mock_get_pipeline.return_value = pipeline
        whisper = MagicMock()

        FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((0.5, 8.0), (9.0, 17.0)),
            language="en"
        )

        whisper.transcribe.assert_not_called()

    @patch(f"{ADAPTER}.torchaudio")
    def test_speech_still_wordless_after_its_span_is_decoded_on_its_own(self, mock_torchaudio, manifest, mock_settings):
        """A sequentially decoded span can still lose its last sentence (FLEURS en @584 s); decoded alone, it's there."""
        mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 30), 16000)
        whisper = MagicMock()
        whisper.transcribe.side_effect = [
            ([_raw_segment(0.5, 7.5, text="first", words=[MagicMock(start=1.0, end=7.0, word="first",
                                                                    probability=0.9)])], MagicMock()),
            ([_raw_segment(0.0, 4.5, text="last", words=[MagicMock(start=0.2, end=4.4, word="last",
                                                                   probability=0.9)])], MagicMock()),
        ]

        segments_out = FasterWhisperAdapter().transcribe(
            model=LoadedModel(manifest=manifest, model_instance=whisper),
            audio_path="/mock/path.wav",
            vad_map=_vad((0.5, 8.0), (9.0, 14.0)),
            language="en"
        )

        assert whisper.transcribe.call_count == 2
        assert len(whisper.transcribe.call_args_list[1].kwargs["audio"]) == 5 * 16000
        # One merged span decoded from 0.5 s, then the lost segment alone from 9.0 s
        assert [(s.text, s.start_sec) for s in segments_out] == [("first", 1.0), ("last", 9.0)]
