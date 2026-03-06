# Canvas: test_t5_adapter_v1
# Path: tests/features/asr_error_correction/test_t5_adapter.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.asr_error_correction.data.t5_adapter import T5AnomalyAdapter
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.ASR_CORRECTION_PHASE_2,
        model_path="/fake/t5",
        required_vram_mb=2000.0
    )


def test_detect_anomalies_success(manifest):
    adapter = T5AnomalyAdapter()

    # Mock pipeline returning a completely different string (triggers similarity < 0.95)
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"generated_text": "I need an MRI."}]
    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipeline)

    sentence_id = uuid.uuid4()
    transcript = ReconstructedTranscript(
        source_id=uuid.uuid4(),
        sentences=[
            ReconstructedSentence(sentence_id=sentence_id, text="I need an m r i.", speaker_id="S1", start_sec=0.0,
                                  end_sec=1.0, segments=[])
        ]
    )

    anomalies = adapter.detect_anomalies(loaded_model, transcript)

    assert len(anomalies) == 1
    assert anomalies[0].sentence_id == sentence_id
    assert anomalies[0].original_text == "I need an m r i."