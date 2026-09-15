# Canvas: test_llama_adapter_v5
# Path: tests/features/domain_adaptation/data/test_llama_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_adaptation.exceptions.model import ModelLoadError

CANDIDATES = {
    "CENTRAL_BANK_MACRO": "inflation, CPI",
    "ORDER_FLOW_EXECUTION": "order flow, DOM",
    "EUROPEAN_CENTRAL_BANK": "ECB, Lagarde",
    "BANK_OF_JAPAN": "BOJ, Ueda",
}


def _llm_returning(content: str) -> MagicMock:
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": content}}]}
    return llm


class TestLlamaAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path="/mock/path/llama.gguf",
            required_vram_mb=3500.0
        )

    @patch.object(LlamaAdapter, '_get_llama_model')
    @patch("os.path.exists")
    def test_load_success(self, mock_exists, mock_get_llama, manifest):
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_get_llama.return_value = mock_instance

        adapter = LlamaAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == mock_instance
        mock_get_llama.assert_called_once_with(manifest.model_path)

    @patch("os.path.exists")
    def test_load_missing_file_raises_error(self, mock_exists, manifest):
        mock_exists.return_value = False
        adapter = LlamaAdapter()

        with pytest.raises(ModelLoadError):
            adapter.load(manifest)

    @patch("app.features.domain_adaptation.data.llama_adapter.settings")
    @patch.object(LlamaAdapter, '_get_grammar')
    def test_predict_subdomains_keeps_high_scoring_topics_best_first(self, mock_get_grammar, mock_settings, manifest):
        mock_settings.DOMAIN_ROUTER_MIN_SCORE = 5
        llm = _llm_returning('{"CENTRAL_BANK_MACRO": 8, "ORDER_FLOW_EXECUTION": 4, "EUROPEAN_CENTRAL_BANK": 9, "BANK_OF_JAPAN": 0}')

        predicted = LlamaAdapter().predict_subdomains(LoadedModel(manifest=manifest, model_instance=llm), "Scout text", CANDIDATES)

        assert predicted == ["EUROPEAN_CENTRAL_BANK", "CENTRAL_BANK_MACRO"]

        grammar_text = mock_get_grammar.call_args[0][0]
        for label in CANDIDATES:
            assert f'"\\"{label}\\":" space score' in grammar_text
        assert 'score ::= "10" | [0-9]' in grammar_text

        call_kwargs = llm.create_chat_completion.call_args.kwargs
        assert call_kwargs["grammar"] == mock_get_grammar.return_value
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 64 + 32 * len(CANDIDATES)
        system_prompt = call_kwargs["messages"][0]["content"]
        assert "- EUROPEAN_CENTRAL_BANK: ECB, Lagarde" in system_prompt
        assert "score from 0 (not discussed at all) to 10" in system_prompt
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Scout text"}

    @patch("app.features.domain_adaptation.data.llama_adapter.settings")
    @patch.object(LlamaAdapter, '_get_grammar')
    def test_predict_subdomains_selects_nothing_when_no_topic_is_discussed(self, mock_get_grammar, mock_settings, manifest):
        mock_settings.DOMAIN_ROUTER_MIN_SCORE = 5
        llm = _llm_returning('{"CENTRAL_BANK_MACRO": 0, "ORDER_FLOW_EXECUTION": 0, "EUROPEAN_CENTRAL_BANK": 0, "BANK_OF_JAPAN": 0}')

        predicted = LlamaAdapter().predict_subdomains(LoadedModel(manifest=manifest, model_instance=llm), "Patient notes", CANDIDATES)

        assert predicted == []

    @patch.object(LlamaAdapter, '_get_grammar')
    def test_predict_subdomains_graceful_degradation_on_json_error(self, mock_get_grammar, manifest):
        llm = _llm_returning('{"CENTRAL_BANK_MACRO": 8, "ORDER_FL')

        predicted = LlamaAdapter().predict_subdomains(LoadedModel(manifest=manifest, model_instance=llm), "Dirty text", CANDIDATES)

        assert predicted == []

    @patch.object(LlamaAdapter, '_get_grammar')
    def test_predict_subdomains_without_candidates_skips_the_model(self, mock_get_grammar, manifest):
        llm = MagicMock()

        assert LlamaAdapter().predict_subdomains(LoadedModel(manifest=manifest, model_instance=llm), "text", {}) == []
        llm.create_chat_completion.assert_not_called()
        mock_get_grammar.assert_not_called()
