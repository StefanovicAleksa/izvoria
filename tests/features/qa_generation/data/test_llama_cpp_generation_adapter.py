# Path: tests/features/qa_generation/data/test_llama_cpp_generation_adapter.py

import json
import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.qa_generation.data.llama_cpp_generation_adapter import LlamaCppGenerationAdapter
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.exceptions.base import QAGenerationError
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError

MODULE = "app.features.qa_generation.data.llama_cpp_generation_adapter"


def _reply(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


class TestLlamaCppGenerationAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.QA_GENERATION,
            model_path="/mock/path/Qwen3.5-9B-Q4_K_M.gguf",
            required_vram_mb=10000.0
        )

    @pytest.fixture
    def loaded(self, manifest):
        return LoadedModel(manifest=manifest, model_instance=MagicMock())

    @patch("os.path.exists", return_value=True)
    @patch.object(LlamaCppGenerationAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_model, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_model.return_value = mock_instance

        adapter = LlamaCppGenerationAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_model.assert_called_once_with(manifest.model_path)

    @patch(f"{MODULE}.build_llama")
    @patch(f"{MODULE}.settings")
    def test_model_uses_generation_context_and_kv_cache_settings(self, mock_settings, mock_build):
        mock_settings.QA_GENERATION_N_CTX = 131072
        mock_settings.QA_LLM_KV_CACHE_TYPE = "q8_0"

        assert LlamaCppGenerationAdapter._get_llama_model("/abs/model.gguf") == mock_build.return_value
        mock_build.assert_called_once_with("/abs/model.gguf", 131072, "q8_0")

    @patch(f"{MODULE}.gc.collect")
    def test_unload_clears_vram(self, mock_gc, manifest):
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded)

        assert result is True
        assert loaded.model_instance is None
        mock_llm.close.assert_called_once()
        mock_gc.assert_called()

    @patch(f"{MODULE}.settings")
    @patch(f"{MODULE}.chat_completion")
    def test_generate_answer_strips_the_empty_think_block(self, mock_chat, mock_settings, loaded):
        mock_settings.QA_LLM_ENABLE_THINKING = False
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1024
        mock_chat.return_value = _reply("<think>\n\n</think>\n\nThe patient has a fracture [Doc 1].")

        response = LlamaCppGenerationAdapter().generate_answer(loaded, "sys_prompt", "context", "query")

        assert response == "The patient has a fracture [Doc 1]."
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        assert args[0] is loaded.model_instance
        assert args[1][0] == {"role": "system", "content": "sys_prompt"}
        assert "context" in args[1][1]["content"] and "query" in args[1][1]["content"]
        assert kwargs == {"enable_thinking": False, "temperature": 0.1, "max_tokens": 1024}

    @patch(f"{MODULE}.chat_completion")
    def test_generate_conversation_answer_sends_the_messages_as_given(self, mock_chat, loaded):
        mock_chat.return_value = _reply("Then he left [Doc 3].", finish_reason="length")
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]

        assert LlamaCppGenerationAdapter().generate_conversation_answer(loaded, messages) == "Then he left [Doc 3]."
        assert mock_chat.call_args[0][1] is messages

    @patch(f"{MODULE}.chat_completion", side_effect=RuntimeError("llama crashed"))
    def test_generate_conversation_answer_maps_errors(self, _, loaded):
        with pytest.raises(QAGenerationError):
            LlamaCppGenerationAdapter().generate_conversation_answer(loaded, [])

    @patch(f"{MODULE}.chat_completion")
    def test_critique_answer_sufficient(self, mock_chat, loaded):
        mock_chat.return_value = _reply(json.dumps({"verdict": "sufficient", "gap_description": None}))

        critique = LlamaCppGenerationAdapter().critique_answer(loaded, "query", "draft answer", "evidence")

        assert critique.verdict == CritiqueVerdict.SUFFICIENT
        assert critique.gap_description is None
        assert mock_chat.call_args.kwargs["response_format"] == {"type": "json_object"}

    @patch(f"{MODULE}.chat_completion")
    def test_critique_answer_needs_more_evidence(self, mock_chat, loaded):
        mock_chat.return_value = _reply(json.dumps(
            {"verdict": "needs_more_evidence", "gap_description": "Claim X is unsupported."}
        ))

        critique = LlamaCppGenerationAdapter().critique_answer(loaded, "query", "draft answer", "evidence")

        assert critique.verdict == CritiqueVerdict.NEEDS_MORE_EVIDENCE
        assert critique.gap_description == "Claim X is unsupported."

    @patch(f"{MODULE}.chat_completion")
    def test_critique_answer_raises_on_unrecognized_verdict(self, mock_chat, loaded):
        mock_chat.return_value = _reply(json.dumps({"verdict": "maybe", "gap_description": None}))

        with pytest.raises(JSONCritiqueParseError):
            LlamaCppGenerationAdapter().critique_answer(loaded, "query", "draft answer", "evidence")

    def test_count_tokens_uses_model_tokenizer_without_bos(self, manifest):
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.tokenize.return_value = [1, 2, 3]
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.count_tokens(loaded, "héllo") == 3
        mock_llm.tokenize.assert_called_once_with("héllo".encode("utf-8"), add_bos=False, special=True)

    @patch(f"{MODULE}.settings")
    def test_context_budget_reserves_prompt_answer_and_margin(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1000
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 8192
        mock_llm.tokenize.return_value = [0] * 300
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_context_token_budget(loaded, "What happened?", include_critique=False)

        assert budget == 8192 - 300 - 1000 - adapter.TEMPLATE_MARGIN_TOKENS

    @patch(f"{MODULE}.settings")
    def test_context_budget_with_critique_also_fits_draft_and_verdict(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1000
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 8192
        # Answer prompt 300 tokens, critique prompt 400 tokens
        mock_llm.tokenize.side_effect = [[0] * 300, [0] * 400]
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_context_token_budget(loaded, "Q", include_critique=True)

        assert budget == 8192 - 400 - 1000 - adapter.CRITIQUE_MAX_TOKENS - adapter.TEMPLATE_MARGIN_TOKENS

    def test_context_budget_never_negative(self, manifest):
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 512
        mock_llm.tokenize.return_value = [0] * 600
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert adapter.get_context_token_budget(loaded, "Q", include_critique=True) == 0

    @patch(f"{MODULE}.settings")
    def test_conversation_budget_reserves_system_prompt_answer_and_margin(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1024
        adapter = LlamaCppGenerationAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 131072
        mock_llm.tokenize.return_value = [0] * 500
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_conversation_token_budget(loaded, "system prompt")

        assert budget == 131072 - 500 - 1024 - adapter.TEMPLATE_MARGIN_TOKENS
        mock_llm.tokenize.assert_called_once_with(b"system prompt", add_bos=False, special=True)

    @patch(f"{MODULE}.settings")
    def test_conversation_budget_never_negative(self, mock_settings, manifest):
        mock_settings.QA_GENERATION_MAX_ANSWER_TOKENS = 1024
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 512
        mock_llm.tokenize.return_value = [0] * 100
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert LlamaCppGenerationAdapter().get_conversation_token_budget(loaded, "s") == 0
