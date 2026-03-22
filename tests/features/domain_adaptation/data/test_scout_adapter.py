# Canvas: test_scout_whisper_adapter_v6
# Path: tests/features/domain_adaptation/data/test_scout_whisper_adapter.py

import uuid
import pytest
import torch
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_adaptation.data.scout_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.domain.entities import TimeSlice


class TestScoutWhisperAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.DOMAIN_SCOUT,
            model_path="/mock/path/faster-whisper-small",
            required_vram_mb=1000.0
        )

    @patch.object(ScoutWhisperAdapter, '_get_whisper_model')
    @patch("os.path.exists")
    def test_load_success(self, mock_exists, mock_get_whisper, manifest):
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_get_whisper.return_value = mock_instance

        adapter = ScoutWhisperAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == mock_instance
        mock_get_whisper.assert_called_once_with(manifest.model_path)

    @patch("app.features.domain_adaptation.data.scout_adapter.torchaudio")
    def test_transcribe_ram_chunking(self, mock_torchaudio, manifest):
        sample_rate = 16000
        dummy_waveform = torch.zeros(1, sample_rate * 5)
        mock_torchaudio.load.return_value = (dummy_waveform, sample_rate)

        mock_whisper_instance = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "Financial terminology."

        def dummy_transcribe_generator():
            yield mock_segment

        mock_whisper_instance.transcribe.return_value = (dummy_transcribe_generator(), MagicMock())
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        source_id = uuid.uuid4()
        slices = [TimeSlice(start_sec=1.0, end_sec=2.0)]

        adapter = ScoutWhisperAdapter()
        result = adapter.transcribe_samples(loaded_model, "/mock/path.wav", slices, source_id)

        assert result.source_id == source_id
        assert "Financial terminology." in result.combined_text

        mock_whisper_instance.transcribe.assert_called_once()
        call_kwargs = mock_whisper_instance.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 1
        assert call_kwargs["vad_filter"] is False