# Canvas: test_llama_adapter_v3
# Path: tests/features/domain_adaptation/data/test_llama_adapter.py

import pytest
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

    # CRITICAL FIX: Patching the local module namespace where Llama is actually imported
    @patch("app.features.domain_adaptation.data.llama_adapter.Llama")
    @patch("os.path.exists")
    def test_load_success(self, mock_exists, mock_llama_cls, manifest):
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_llama_cls.return_value = mock_instance

        adapter = LlamaAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == mock_instance
        mock_llama_cls.assert_called_once_with(
            model_path=manifest.model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )

    @patch("os.path.exists")
    def test_load_missing_file_raises_error(self, mock_exists, manifest):
        mock_exists.return_value = False
        adapter = LlamaAdapter()

        with pytest.raises(ModelLoadError):
            adapter.load(manifest)

    @patch("app.features.domain_adaptation.data.llama_adapter.LlamaGrammar.from_string")
    def test_predict_subdomains_uses_gbnf_grammar(self, mock_grammar_cls, manifest):
        adapter = LlamaAdapter()
        mock_llm = MagicMock()
        mock_grammar_instance = MagicMock()

        mock_grammar_cls.return_value = mock_grammar_instance

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

        mock_grammar_cls.assert_called_once_with(adapter.JSON_ARRAY_GRAMMAR)
        mock_llm.create_chat_completion.assert_called_once_with(
            messages=[
                {"role": "system", "content": (
                    "You are an expert legal/medical taxonomy router.\n"
                    "Analyze the following dirty transcript and identify the sub-domains being discussed.\n"
                    "You MUST choose from the following exact labels: OPTIONS_TRADING, M_AND_A.\n"
                    "Output ONLY a raw JSON array of the identified strings."
                )},
                {"role": "user", "content": "Dirty scout text"}
            ],
            max_tokens=128,
            temperature=0.0,
            grammar=mock_grammar_instance
        )