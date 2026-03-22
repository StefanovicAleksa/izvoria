# Canvas: test_llama_adapter_v4
# Path: tests/features/domain_adaptation/data/test_llama_adapter.py

import pytest
import json
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
from app.features.domain_adaptation.exceptions.model import ModelLoadError


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

    @patch.object(LlamaAdapter, '_get_grammar')
    def test_predict_subdomains_uses_gbnf_grammar(self, mock_get_grammar, manifest):
        adapter = LlamaAdapter()
        mock_llm = MagicMock()
        mock_grammar_instance = MagicMock()

        mock_get_grammar.return_value = mock_grammar_instance

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '["OPTIONS_TRADING", "M_AND_A"]'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        predicted = adapter.predict_subdomains(
            loaded_model,
            "Dirty scout text",
            ["OPTIONS_TRADING", "M_AND_A"]
        )

        assert len(predicted) == 2
        assert "OPTIONS_TRADING" in predicted

        mock_get_grammar.assert_called_once_with(adapter.JSON_ARRAY_GRAMMAR)
        mock_llm.create_chat_completion.assert_called_once()
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["grammar"] == mock_grammar_instance

    @patch.object(LlamaAdapter, '_get_grammar')
    @patch("app.features.domain_adaptation.data.llama_adapter.json.loads")
    def test_predict_subdomains_graceful_degradation_on_json_error(self, mock_json_loads, mock_get_grammar, manifest):
        adapter = LlamaAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '["TRUNCATED_JSON...'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)
        mock_json_loads.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        predicted = adapter.predict_subdomains(loaded_model, "Dirty text", ["OPTIONS_TRADING"])

        assert predicted == []