# Canvas: test_rag_retrieval_embedding_adapter_v1
# Path: tests/features/rag_retrieval/data/test_embedding_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter


class TestHuggingFacePromptAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path="/mock/path/bge-m3",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(HuggingFacePromptAdapter, "_get_embedding_model")
    def test_load_success_cuda(self, mock_get_model, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_model.return_value = mock_instance

        adapter = HuggingFacePromptAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.model_instance == mock_instance
        mock_get_model.assert_called_once_with(manifest.model_path)

    @patch("app.features.rag_retrieval.data.embedding_adapter.gc.collect")
    @patch("app.features.rag_retrieval.data.embedding_adapter.torch.cuda.is_available", return_value=True)
    @patch("app.features.rag_retrieval.data.embedding_adapter.torch.cuda.empty_cache")
    def test_unload_success(self, mock_empty_cache, mock_cuda, mock_gc, manifest):
        adapter = HuggingFacePromptAdapter()
        mock_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)

        adapter.unload(loaded_model)

        assert not hasattr(loaded_model, "model_instance")
        mock_gc.assert_called()
        mock_empty_cache.assert_called()

    def test_embed_prompt_fails_if_not_loaded(self, manifest):
        adapter = HuggingFacePromptAdapter()
        loaded_model = LoadedModel(manifest=manifest, model_instance=None)

        with pytest.raises(Exception, match="model is not loaded"):
            adapter.embed_prompt(loaded_model, "search query")

    def test_embed_prompt_fails_if_empty(self, manifest):
        adapter = HuggingFacePromptAdapter()
        mock_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)

        with pytest.raises(ValueError, match="search prompt is empty"):
            adapter.embed_prompt(loaded_model, "   ")

    def test_embed_prompt_success(self, manifest):
        adapter = HuggingFacePromptAdapter()
        mock_instance = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_instance.encode.return_value = mock_embedding

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)
        result = adapter.embed_prompt(loaded_model, "patient migraine symptoms")

        assert result == [0.1, 0.2, 0.3]
        mock_instance.encode.assert_called_once()