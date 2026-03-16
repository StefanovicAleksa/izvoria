# Canvas: test_scout_whisper_adapter_v3
# Path: tests/features/domain_adaptation/test_scout_whisper_adapter.py

import uuid
from unittest.mock import MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_adaptation.data.scout_whisper_adapter import ScoutWhisperAdapter
from app.features.domain_adaptation.domain.entities import AudioSample


def test_transcribe_samples(mocker):
    adapter = ScoutWhisperAdapter()
    manifest = ModelManifest(model_type=ModelType.DOMAIN_SCOUT, model_path="mock/path", required_vram_mb=200.0)
    mock_whisper = MagicMock()

    mock_segment = MagicMock()
    mock_segment.text = "Hello financial world."
    mock_whisper.transcribe.return_value = ([mock_segment], None)

    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper)
    source_id = uuid.uuid4()
    samples = [AudioSample(start_sec=0.0, end_sec=20.0, file_path="mock.wav")]

    mocker.patch("os.path.exists", return_value=True)

    result = adapter.transcribe_samples(loaded_model, samples, source_id)

    assert result.source_id == source_id
    assert "Hello financial world." in result.combined_text
    mock_whisper.transcribe.assert_called_once()
    assert mock_whisper.transcribe.call_args[1]["vad_filter"] is False