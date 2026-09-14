# File: tests/features/vad/data/test_nemo_adapter.py
import uuid
import pytest
from unittest.mock import patch, MagicMock
import torch

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.data.nemo_adapter import NeMoVadAdapter
from app.features.vad.domain.entities import VadSegment
from app.features.vad.exceptions.model import VadModelFilePathError

ADAPTER = "app.features.vad.data.nemo_adapter"


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.VAD,
        model_path=settings.VAD_MODEL_PATH,
        required_vram_mb=1200.0
    )


@pytest.fixture
def vad_settings():
    with patch(f"{ADAPTER}.settings") as mocked, patch(f"{ADAPTER}.torch.cuda.is_available", return_value=False):
        mocked.VAD_MODEL_PATH = settings.VAD_MODEL_PATH
        mocked.VAD_BATCH_SIZE = 512
        mocked.VAD_PAD_SEC = 0.3
        yield mocked


def _model_with_speech_pattern(is_speech):
    """Forward mock returning speech/non-speech logits for each window of every batch, in order."""
    model = MagicMock()
    counter = {"window": 0}

    def forward(input_signal, input_signal_length):
        rows = []
        for _ in range(input_signal.shape[0]):
            speech = is_speech(counter["window"])
            rows.append([0.0, 9.9] if speech else [9.9, 0.0])
            counter["window"] += 1
        return torch.tensor(rows)

    model.forward.side_effect = forward
    return model


@patch(f"{ADAPTER}.EncDecClassificationModel")
@patch("os.path.exists")
def test_load_success(mock_exists, mock_nemo_model, manifest):
    mock_exists.return_value = True

    mock_instance = MagicMock()
    mock_nemo_model.restore_from.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    loaded_model = NeMoVadAdapter().load(manifest)

    assert loaded_model.manifest == manifest
    assert loaded_model.model_instance == mock_instance
    mock_nemo_model.restore_from.assert_called_once_with(restore_path=settings.VAD_MODEL_PATH)
    mock_instance.to.assert_called_once_with("cuda")
    mock_instance.eval.assert_called_once()


@patch("os.path.exists")
def test_load_missing_file_raises_error(mock_exists, manifest):
    mock_exists.return_value = False

    with pytest.raises(VadModelFilePathError):
        NeMoVadAdapter().load(manifest)


@patch(f"{ADAPTER}.torchaudio")
def test_generate_vad_map_continuous_speech_is_padded_and_runs_to_the_end(mock_torchaudio, manifest, vad_settings):
    mock_torchaudio.load.return_value = (torch.zeros(1, 16000), 16000)
    model = _model_with_speech_pattern(lambda idx: True)
    source_id = uuid.uuid4()

    vad_map = NeMoVadAdapter().generate_vad_map(LoadedModel(manifest=manifest, model_instance=model), "fake.wav", source_id)

    assert vad_map.source_id == source_id
    assert len(vad_map.segments) == 1
    # First window centre (0.315s) minus 0.3s padding; trailing speech runs to the end of the audio
    assert vad_map.segments[0].start_sec == pytest.approx(0.015)
    assert vad_map.segments[0].end_sec == pytest.approx(1.0)
    assert vad_map.segments[0].confidence > 0.9


@patch(f"{ADAPTER}.torchaudio")
def test_segment_end_uses_window_centre_plus_padding(mock_torchaudio, manifest, vad_settings):
    """Stamping windows at their start ended segments up to ~0.5s before speech stopped."""
    mock_torchaudio.load.return_value = (torch.zeros(1, 16000 * 3), 16000)
    model = _model_with_speech_pattern(lambda idx: idx < 10)

    vad_map = NeMoVadAdapter().generate_vad_map(LoadedModel(manifest=manifest, model_instance=model), "fake.wav", uuid.uuid4())

    assert len(vad_map.segments) == 1
    # Speech stops at window 10, whose centre is 10 * 0.08 + 0.315 = 1.115s; padded by 0.3s
    assert vad_map.segments[0].end_sec == pytest.approx(1.415)


@patch(f"{ADAPTER}.torchaudio")
def test_windows_are_scored_in_batches(mock_torchaudio, manifest, vad_settings):
    vad_settings.VAD_BATCH_SIZE = 2
    mock_torchaudio.load.return_value = (torch.zeros(1, 16000), 16000)  # 5 windows
    model = _model_with_speech_pattern(lambda idx: True)

    NeMoVadAdapter().generate_vad_map(LoadedModel(manifest=manifest, model_instance=model), "fake.wav", uuid.uuid4())

    batch_shapes = [c.kwargs["input_signal"].shape for c in model.forward.call_args_list]
    assert batch_shapes == [torch.Size([2, 10080]), torch.Size([2, 10080]), torch.Size([1, 10080])]
    lengths = model.forward.call_args_list[0].kwargs["input_signal_length"]
    assert lengths.tolist() == [10080, 10080]


@patch(f"{ADAPTER}.torchaudio")
def test_audio_shorter_than_one_window_is_still_scored(mock_torchaudio, manifest, vad_settings):
    mock_torchaudio.load.return_value = (torch.zeros(1, 8000), 16000)
    model = _model_with_speech_pattern(lambda idx: True)

    vad_map = NeMoVadAdapter().generate_vad_map(LoadedModel(manifest=manifest, model_instance=model), "fake.wav", uuid.uuid4())

    assert model.forward.call_count == 1
    assert len(vad_map.segments) == 1
    assert vad_map.segments[0].end_sec == pytest.approx(0.5)


def test_pad_and_merge_joins_segments_whose_padding_overlaps():
    segments = [
        VadSegment(start_sec=1.0, end_sec=2.0, confidence=0.8),
        VadSegment(start_sec=2.4, end_sec=3.0, confidence=0.6),
        VadSegment(start_sec=9.9, end_sec=10.0, confidence=0.9),
    ]

    merged = NeMoVadAdapter._pad_and_merge(segments, total_duration=10.0, pad_sec=0.3)

    assert [(round(s.start_sec, 3), round(s.end_sec, 3)) for s in merged] == [(0.7, 3.3), (9.6, 10.0)]
    assert merged[0].confidence == pytest.approx((0.8 * 1.6 + 0.6 * 1.2) / 2.8)
