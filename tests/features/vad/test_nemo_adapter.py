# File: tests/features/vad/test_nemo_adapter.py
import uuid
import pytest
from unittest.mock import patch, MagicMock
import torch

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.data.nemo_adapter import NeMoVadAdapter
from app.features.vad.exceptions.model import VadModelFilePathError


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.VAD,
        model_path=settings.VAD_MODEL_PATH,
        required_vram_mb=1200.0
    )


@patch("app.features.vad.data.nemo_adapter.EncDecClassificationModel")
@patch("os.path.exists")
def test_load_success(mock_exists, mock_nemo_model, manifest):
    mock_exists.return_value = True

    mock_instance = MagicMock()
    mock_nemo_model.restore_from.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    adapter = NeMoVadAdapter()
    loaded_model = adapter.load(manifest)

    assert loaded_model.manifest == manifest
    assert loaded_model.model_instance == mock_instance
    mock_nemo_model.restore_from.assert_called_once_with(restore_path=settings.VAD_MODEL_PATH)
    mock_instance.to.assert_called_once_with("cuda")
    mock_instance.eval.assert_called_once()


@patch("os.path.exists")
def test_load_missing_file_raises_error(mock_exists, manifest):
    mock_exists.return_value = False
    adapter = NeMoVadAdapter()

    with pytest.raises(VadModelFilePathError):
        adapter.load(manifest)


@patch("app.features.vad.data.nemo_adapter.torchaudio")
def test_generate_vad_map_success(mock_torchaudio, manifest):
    # Setup Synthetic Audio (1 second of 16kHz audio)
    sample_rate = 16000
    dummy_waveform = torch.zeros(1, sample_rate)
    mock_torchaudio.load.return_value = (dummy_waveform, sample_rate)

    # Setup Mock Model
    mock_instance = MagicMock()
    # Mock the forward pass to always return a high probability of speech
    # Logits shape: [batch, classes]. Class 1 is speech.
    mock_instance.forward.return_value = torch.tensor([[0.1, 9.9]])

    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)
    adapter = NeMoVadAdapter()

    source_id = uuid.uuid4()

    # Act
    vad_map = adapter.generate_vad_map(loaded_model, "fake/path.wav", source_id)

    # Assert
    assert vad_map.source_id == source_id
    assert len(vad_map.segments) == 1
    # Check that it detected speech for the whole 1-second duration
    assert vad_map.segments[0].start_sec == 0.0
    assert vad_map.segments[0].end_sec > 0.0
    assert vad_map.segments[0].confidence > 0.9