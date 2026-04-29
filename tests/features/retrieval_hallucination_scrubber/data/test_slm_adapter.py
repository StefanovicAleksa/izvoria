# Canvas: test_retrieval_scrubber_slm_adapter_v3
# Path: tests/features/retrieval_hallucination_scrubber/data/test_slm_adapter.py

import pytest
import uuid
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter


class TestQwenRetrievalScrubberAdapter:
    @pytest.fixture
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.RETRIEVAL_SCRUBBER,
            model_path="/fake/qwen.gguf",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenRetrievalScrubberAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_qwen: MagicMock, mock_exists: MagicMock, manifest: ModelManifest) -> None:
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        adapter = QwenRetrievalScrubberAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    def test_evaluate_relevance_rejection(self, manifest: ModelManifest) -> None:
        adapter = QwenRetrievalScrubberAdapter()
        mock_llm = MagicMock()

        json_payload = '{"is_relevant": false, "discard_reason": "Discusses cafeteria, not MRI."}'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        chunk_id = uuid.uuid4()
        audit = adapter.evaluate_relevance(loaded_model, chunk_id, "What does the MRI show?", "We serve pizza on Friday.")

        assert audit.chunk_id == chunk_id
        assert audit.is_relevant is False
        assert audit.discard_reason == "Discusses cafeteria, not MRI."

        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"

    def test_evaluate_relevance_acceptance(self, manifest: ModelManifest) -> None:
        adapter = QwenRetrievalScrubberAdapter()
        mock_llm = MagicMock()

        json_payload = '{"is_relevant": true, "discard_reason": null}'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        audit = adapter.evaluate_relevance(loaded_model, uuid.uuid4(), "What does the MRI show?", "The MRI indicates a tumor.")

        assert audit.is_relevant is True
        assert audit.discard_reason is None

    def test_evaluate_relevance_salvages_conversational_filler(self, manifest: ModelManifest) -> None:
        adapter = QwenRetrievalScrubberAdapter()
        mock_llm = MagicMock()

        json_payload = 'Sure! Here is your JSON:\n{"is_relevant": false, "discard_reason": "Not related."}\nHave a great day!'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        chunk_id = uuid.uuid4()
        audit = adapter.evaluate_relevance(loaded_model, chunk_id, "prompt", "text")

        assert audit.chunk_id == chunk_id
        assert audit.is_relevant is False
        assert audit.discard_reason == "Not related."

    @patch("app.features.retrieval_hallucination_scrubber.data.slm_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc: MagicMock, manifest: ModelManifest) -> None:
        adapter = QwenRetrievalScrubberAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()