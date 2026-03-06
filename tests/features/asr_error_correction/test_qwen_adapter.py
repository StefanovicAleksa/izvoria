# Canvas: test_qwen_adapter_v1
# Path: tests/features/asr_error_correction/test_qwen_adapter.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.asr_error_correction.data.qwen_adapter import QwenCorrectionAdapter
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.asr_error_correction.domain.entities import DocumentGlossary, GlossaryItem, AnomalyTag


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.ASR_CORRECTION_PHASE_3,
        model_path="/fake/qwen.gguf",
        required_vram_mb=5000.0
    )


def test_apply_corrections_success(manifest):
    adapter = QwenCorrectionAdapter()
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"text": "I like an MRI scan."}]}
    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

    sentence_id = uuid.uuid4()
    transcript = ReconstructedTranscript(
        source_id=uuid.uuid4(),
        sentences=[
            ReconstructedSentence(sentence_id=sentence_id, text="I like an m r i scan.", speaker_id="S1", start_sec=0.0,
                                  end_sec=1.0, segments=[])
        ]
    )

    anomalies = [
        AnomalyTag(
            sentence_id=sentence_id,
            original_start_sec=0.0,
            original_end_sec=1.0,
            original_text="I like an m r i scan.",
            speaker_id="S1",
            confidence_of_error=0.9,
            anomaly_reason="T5"
        )
    ]
    glossary = DocumentGlossary(
        source_id=transcript.source_id,
        domain_category="Medical",
        items=[GlossaryItem(term="MRI", context="Magnetic scan")]
    )

    result = adapter.apply_corrections(loaded_model, transcript, anomalies, glossary)

    assert len(result.sentences) == 1
    assert result.sentences[0].text == "I like an MRI scan."
    assert len(result.audit_trail) == 1
    assert "MRI" in result.audit_trail[0].applied_glossary_terms