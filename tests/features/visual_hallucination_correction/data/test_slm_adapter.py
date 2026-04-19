# Canvas: test_hallucination_slm_adapter_v2
# Path: tests/features/visual_hallucination_correction/data/test_slm_adapter.py

import pytest
import uuid
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter


class TestQwenHallucinationAdapter:
    @pytest.fixture
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.VISUAL_HALLUCINATION_CORRECTION,
            model_path="/fake/qwen.gguf",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenHallucinationAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_qwen: MagicMock, mock_exists: MagicMock, manifest: ModelManifest) -> None:
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        adapter = QwenHallucinationAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    def test_evaluate_visual_match_rejection(self, manifest: ModelManifest) -> None:
        adapter = QwenHallucinationAdapter()
        mock_llm = MagicMock()

        json_payload = '{"is_match": false, "discard_reason": "Irrelevant background of a brick wall."}'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        chunk_id = uuid.uuid4()
        audit = adapter.evaluate_visual_match(loaded_model, chunk_id, "text", "description")

        assert audit.chunk_id == chunk_id
        assert audit.is_visual_accepted is False
        assert audit.discard_reason == "Irrelevant background of a brick wall."

        # Verify JSON object mode was enforced
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_object"

    def test_evaluate_visual_match_acceptance(self, manifest: ModelManifest) -> None:
        adapter = QwenHallucinationAdapter()
        mock_llm = MagicMock()

        json_payload = '{"is_match": true, "discard_reason": null}'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        audit = adapter.evaluate_visual_match(loaded_model, uuid.uuid4(), "Look at this scan", "An X-ray of a lung.")

        assert audit.is_visual_accepted is True
        assert audit.discard_reason is None

    def test_evaluate_visual_match_salvages_conversational_filler(self, manifest: ModelManifest) -> None:
        """
        NEW: Verifies the regex safely extracts the JSON payload even if the SLM ignores
        the system prompt and generates conversational filler around the JSON.
        """
        adapter = QwenHallucinationAdapter()
        mock_llm = MagicMock()

        # Malformed payload with conversational wrapper
        json_payload = 'Here is the analysis:\n{"is_match": false, "discard_reason": "Totally wrong."}\nHope this helps!'
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json_payload}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        chunk_id = uuid.uuid4()
        audit = adapter.evaluate_visual_match(loaded_model, chunk_id, "text", "description")

        # Assertion ensures the Regex fallback successfully bypassed the JSONDecodeError
        assert audit.chunk_id == chunk_id
        assert audit.is_visual_accepted is False
        assert audit.discard_reason == "Totally wrong."

    @patch("app.features.visual_hallucination_correction.data.slm_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc: MagicMock, manifest: ModelManifest) -> None:
        adapter = QwenHallucinationAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()