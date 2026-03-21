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

    # FIX: Patch the external library since the adapter now lazy-loads it
    @patch("llama_cpp.Llama")
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

    def test_predict_subdomains_clean_json(self, manifest):
        adapter = LlamaAdapter()
        mock_llm = MagicMock()

        # FIX: Update mock to match the `create_chat_completion` return structure
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
        mock_llm.create_chat_completion.assert_called_once()

    def test_predict_subdomains_markdown_json_fallback(self, manifest):
        adapter = LlamaAdapter()
        mock_llm = MagicMock()

        # FIX: Update mock to match the `create_chat_completion` return structure
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '```json\n["CORPORATE_FINANCE"]\n```'}}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        predicted = adapter.predict_subdomains(loaded_model, "Dirty text", ["CORPORATE_FINANCE"])

        assert len(predicted) == 1
        assert predicted[0] == "CORPORATE_FINANCE"