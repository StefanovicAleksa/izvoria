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
    def test_count_tokens_uses_model_tokenizer_without_bos(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.tokenize.return_value = [1, 2, 3]
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.count_tokens(loaded, "héllo") == 3
        mock_llm.tokenize.assert_called_once_with("héllo".encode("utf-8"), add_bos=False, special=True)

    @patch("app.features.qa_generation.data.gemma_adapter.settings")
    def test_context_budget_reserves_prompt_answer_and_margin(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1000
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 8192
        mock_llm.tokenize.return_value = [0] * 300
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_context_token_budget(loaded, "What happened?", include_critique=False)

        assert budget == 8192 - 300 - 1000 - adapter.TEMPLATE_MARGIN_TOKENS

    @patch("app.features.qa_generation.data.gemma_adapter.settings")
    def test_context_budget_with_critique_also_fits_draft_and_verdict(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1000
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 8192
        # Answer prompt 300 tokens, critique prompt 400 tokens
        mock_llm.tokenize.side_effect = [[0] * 300, [0] * 400]
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_context_token_budget(loaded, "Q", include_critique=True)

        assert budget == 8192 - 400 - 1000 - adapter.CRITIQUE_MAX_TOKENS - adapter.TEMPLATE_MARGIN_TOKENS

    def test_context_budget_never_negative(self, manifest):
        adapter = GemmaGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 512
        mock_llm.tokenize.return_value = [0] * 600
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.get_context_token_budget(loaded, "Q", include_critique=True) == 0
