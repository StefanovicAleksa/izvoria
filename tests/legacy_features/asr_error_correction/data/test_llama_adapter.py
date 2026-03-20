# Canvas: test_llama_adapter_v2
# Path: tests/features/asr_error_correction/test_llama_adapter.py

import pytest
import uuid
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.legacy_features.asr_error_correction.data.llama_adapter import LlamaGlossaryAdapter
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.ASR_CORRECTION_PHASE_1,
        model_path="/fake/llama.gguf",
        required_vram_mb=3500.0
    )


@patch("os.path.exists")
@patch("app.legacy_features.asr_error_correction.data.llama_adapter.Llama")
def test_load_success(mock_llama, mock_exists, manifest):
    mock_exists.return_value = True
    mock_llama_instance = MagicMock()
    mock_llama.return_value = mock_llama_instance

    adapter = LlamaGlossaryAdapter()
    loaded = adapter.load(manifest)

    assert loaded.model_instance == mock_llama_instance
    assert loaded.manifest == manifest


def test_extract_glossary_success(manifest):
    adapter = LlamaGlossaryAdapter()
    mock_llm = MagicMock()
    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {"message": {"content": '{"domain_category": "Medical", "items": [{"term": "MRI", "context": "Scan"}]}'}}
        ]
    }

    transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=[])

    glossary = adapter.extract_glossary(loaded_model, transcript)

    assert glossary.domain_category == "Medical"
    assert len(glossary.items) == 1
    assert glossary.items[0].term == "MRI"