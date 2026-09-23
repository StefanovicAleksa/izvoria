# Path: tests/features/qa_reasoning/data/test_qwen_reasoner_adapter.py

import pytest
import json
import uuid
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.qa_reasoning.data.qwen_reasoner_adapter import QwenReasonerAdapter
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.qa_reasoning.exceptions.pipeline import JSONReasonerParseError


MODULE = "app.features.qa_reasoning.data.qwen_reasoner_adapter"


class TestQwenReasonerAdapter:
    @pytest.fixture(autouse=True)
    def chat_completion(self):
        """
        The adapter calls llama_cpp_chat.chat_completion(llm, messages, enable_thinking=..., **kw); route it to the
        mocked llm.create_chat_completion so each test can stub and inspect a single call on the model mock.
        """
        def _route(llm, messages, enable_thinking, **kwargs):
            return llm.create_chat_completion(messages=messages, **kwargs)

        with patch(f"{MODULE}.chat_completion", side_effect=_route) as mock_chat:
            yield mock_chat

    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.QA_REASONER,
            model_path="/mock/path/Qwen3.5-9B-Q4_K_M.gguf",
            required_vram_mb=10000.0
        )

    @patch(f"{MODULE}.build_llama")
    @patch(f"{MODULE}.settings")
    def test_model_uses_reasoner_context_and_kv_cache_settings(self, mock_settings, mock_build):
        mock_settings.QA_REASONER_N_CTX = 65536
        mock_settings.QA_LLM_KV_CACHE_TYPE = "q8_0"

        assert QwenReasonerAdapter._get_llama_model("/abs/model.gguf") == mock_build.return_value
        mock_build.assert_called_once_with("/abs/model.gguf", 65536, "q8_0")

    @patch(f"{MODULE}.settings")
    def test_rewrite_passes_the_thinking_switch_and_strips_think_blocks(self, mock_settings, chat_completion,
                                                                        manifest):
        mock_settings.QA_LLM_ENABLE_THINKING = False
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '<think>\n\n</think>\n\n{"standalone_query": "Q2"}'}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        assert QwenReasonerAdapter().rewrite_query(loaded, "Q", "prior") == "Q2"
        assert chat_completion.call_args.args[0] is mock_llm
        assert chat_completion.call_args.kwargs["enable_thinking"] is False

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenReasonerAdapter, "_get_llama_model")
    def test_load_success(self, mock_get_model, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_model.return_value = mock_instance

        adapter = QwenReasonerAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_model.assert_called_once()

    @patch("app.features.qa_reasoning.data.qwen_reasoner_adapter.gc.collect")
    def test_unload_clears_vram(self, mock_gc, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded)

        assert result is True
        assert loaded.model_instance is None
        mock_llm.close.assert_called_once()
        mock_gc.assert_called()

    def test_reason_answer_ready(self, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        chunk_id = uuid.uuid4()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "chain_of_thought": "thinking...",
                "interim_analysis": "The evidence supports X.",
                "evidence_links": [
                    {"evidence_id": "E2", "claim": "X happened"},
                    {"evidence_id": "E9", "claim": "Y happened"},  # not in the pool: dropped
                ],
                "decision": "answer_ready",
                "refined_query": None,
                "clarification_question": None
            })}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        decision = adapter.reason(loaded, "What happened?", "evidence block", "history block", "", "",
                                  {"E1": uuid.uuid4(), "E2": chunk_id})

        assert decision.decision == ReasonerDecisionType.ANSWER_READY
        assert decision.interim_analysis == "The evidence supports X."
        assert [(link.chunk_id, link.claim) for link in decision.evidence_links] == [(chunk_id, "X happened")]

    def test_reason_refine_query(self, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "chain_of_thought": "need more",
                "interim_analysis": "Not enough yet.",
                "evidence_links": [],
                "decision": "refine_query",
                "refined_query": "narrower search",
                "clarification_question": None
            })}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        decision = adapter.reason(loaded, "What happened?", "evidence", "history", "", "", {})

        assert decision.decision == ReasonerDecisionType.REFINE_QUERY
        assert decision.refined_query == "narrower search"

    def test_reason_fails_closed_on_unrecognized_decision(self, manifest):
        """No branch among the three is a safe default, so an unrecognized decision must raise, not default."""
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "chain_of_thought": "x", "interim_analysis": "y", "evidence_links": [],
                "decision": "give_up", "refined_query": None, "clarification_question": None
            })}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        with pytest.raises(JSONReasonerParseError):
            adapter.reason(loaded, "Q", "evidence", "history", "", "", {})

    def test_reason_skips_malformed_evidence_link_but_keeps_decision(self, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "chain_of_thought": "x", "interim_analysis": "y",
                "evidence_links": [{"evidence_id": "E1"}],  # missing "claim"
                "decision": "answer_ready", "refined_query": None, "clarification_question": None
            })}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        decision = adapter.reason(loaded, "Q", "evidence", "history", "", "", {})

        assert decision.decision == ReasonerDecisionType.ANSWER_READY
        assert decision.evidence_links == []

    @pytest.mark.parametrize("raw_tag", ["E1", "[E1]", " e1 "])
    def test_reason_accepts_tag_spelling_variants(self, manifest, raw_tag):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        chunk_id = uuid.uuid4()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "chain_of_thought": "x", "interim_analysis": "y",
                "evidence_links": [{"evidence_id": raw_tag, "claim": "c"}],
                "decision": "answer_ready", "refined_query": None, "clarification_question": None
            })}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        decision = adapter.reason(loaded, "Q", "evidence", "history", "", "", {"E1": chunk_id})

        assert [link.chunk_id for link in decision.evidence_links] == [chunk_id]

    def test_evidence_budget_reserves_prompt_reply_and_margin(self, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.n_ctx.return_value = 8192
        mock_llm.tokenize.return_value = [0] * 900
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        budget = adapter.get_evidence_token_budget(loaded, "Q", "history", "prior", "gap")

        assert budget == 8192 - 900 - adapter.REASON_MAX_TOKENS - adapter.TEMPLATE_MARGIN_TOKENS
        counted = mock_llm.tokenize.call_args.args[0].decode("utf-8")
        assert "history" in counted and "prior" in counted and "gap" in counted

    def test_rewrite_query_returns_standalone_query(self, manifest):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": json.dumps({"standalone_query": " Why did the pound reverse? "})}}]
        }
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        rewritten = adapter.rewrite_query(loaded, "Why did it reverse?", "[PRIOR TURN]\nQ: pound?\nA: It rose.")

        assert rewritten == "Why did the pound reverse?"
        call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == adapter.REWRITE_MAX_TOKENS
        assert "Why did it reverse?" in call_kwargs["messages"][1]["content"]
        assert "It rose." in call_kwargs["messages"][1]["content"]

    @pytest.mark.parametrize("content", ['{"standalone_query": ""}', '{"other": "x"}', '["x"]'])
    def test_rewrite_query_raises_parse_error_without_a_query(self, manifest, content):
        adapter = QwenReasonerAdapter()
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": content}}]}
        loaded = LoadedModel(manifest=manifest, model_instance=mock_llm)

        with pytest.raises(JSONReasonerParseError):
            adapter.rewrite_query(loaded, "Q", "prior")


class TestPromptForDocuments:
    def test_the_document_prompt_only_extends_the_opening_line(self):
        from app.features.qa_reasoning.data.prompts import (
            QUERY_REWRITE_SYSTEM_PROMPT, REASONER_DOCUMENTS_SYSTEM_PROMPT, REASONER_SYSTEM_PROMPT,
            reasoner_system_prompt,
        )

        assert reasoner_system_prompt(False) is REASONER_SYSTEM_PROMPT
        assert reasoner_system_prompt(True) is REASONER_DOCUMENTS_SYSTEM_PROMPT
        assert "and documents" in REASONER_DOCUMENTS_SYSTEM_PROMPT and "and documents" not in REASONER_SYSTEM_PROMPT
        assert "transcript search engine" in QUERY_REWRITE_SYSTEM_PROMPT
