# Canvas: test_gemma_adapter_v4
# Path: tests/features/qa_generation/data/test_gemma_adapter.py

import json
import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.qa_generation.data.gemma_adapter import GemmaGenerationAdapter
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError


class TestGemmaGenerationAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.QA_GENERATION,
            model_path="/mock/path/gemma-4-e4b-it-Q4_K_M.gguf",
            required_vram_mb=4500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(GemmaGenerationAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_model, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_model.return_value = mock_instance

        adapter = GemmaGenerationAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_model.assert_called_once_with(manifest.model_path)

    @patch("app.features.qa_generation.data.gemma_adapter.gc.collect")
    def test_unload_clears_vram(self, mock_gc, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        loaded = LoadedModel(
            manifest=manifest,
            model_instance=mock_llm
        )

        result = adapter.unload(loaded)

        assert result is True
        assert loaded.model_instance is None
        mock_llm.close.assert_called_once()
        mock_gc.assert_called()

    def test_generate_answer_success(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "The patient has a fracture."}}]
        }

        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        response = adapter.generate_answer(loaded, "sys_prompt", "context", "query")

        assert response == "The patient has a fracture."
        mock_llm.create_chat_completion.assert_called_once()

    def test_critique_answer_sufficient(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({"verdict": "sufficient", "gap_description": None})}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        critique = adapter.critique_answer(loaded, "query", "draft answer", "evidence")

        assert critique.verdict == CritiqueVerdict.SUFFICIENT
        assert critique.gap_description is None

    def test_critique_answer_needs_more_evidence(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(
                {"verdict": "needs_more_evidence", "gap_description": "Claim X is unsupported."}
            )}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        critique = adapter.critique_answer(loaded, "query", "draft answer", "evidence")

        assert critique.verdict == CritiqueVerdict.NEEDS_MORE_EVIDENCE
        assert critique.gap_description == "Claim X is unsupported."

    def test_critique_answer_raises_on_unrecognized_verdict(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({"verdict": "maybe", "gap_description": None})}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        with pytest.raises(JSONCritiqueParseError):
            adapter.critique_answer(loaded, "query", "draft answer", "evidence")