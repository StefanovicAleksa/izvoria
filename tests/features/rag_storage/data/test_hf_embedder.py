# Path: tests/features/rag_storage/data/test_hf_embedder.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter


class TestHuggingFaceEmbeddingAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.RAG_EMBEDDING,
            model_path="/mock/path/bge-m3",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(HuggingFaceEmbeddingAdapter, "_get_embedding_model")
    def test_load_success_cuda(self, mock_get_model, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_model.return_value = mock_instance

        adapter = HuggingFaceEmbeddingAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.model_instance == mock_instance
        mock_get_model.assert_called_once_with(manifest.model_path)

    @patch("app.features.rag_storage.data.hf_embedder.gc.collect")
    @patch("app.features.rag_storage.data.hf_embedder.torch.cuda.is_available", return_value=True)
    @patch("app.features.rag_storage.data.hf_embedder.torch.cuda.empty_cache")
    def test_unload_success(self, mock_empty_cache, mock_cuda, mock_gc, manifest):
        adapter = HuggingFaceEmbeddingAdapter()
        mock_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)

        adapter.unload(loaded_model)

        assert not hasattr(loaded_model, "model_instance")
        mock_gc.assert_called()
        mock_empty_cache.assert_called()

    def test_embed_batch_fails_if_not_loaded(self, manifest):
        adapter = HuggingFaceEmbeddingAdapter()
        loaded_model = LoadedModel(manifest=manifest, model_instance=None)
        with pytest.raises(Exception, match="Cannot embed texts"):
            adapter.embed_batch(loaded_model, ["test"])

    def test_embed_batch_success(self, manifest):
        adapter = HuggingFaceEmbeddingAdapter()
        mock_instance = MagicMock()

        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_instance.encode.return_value = mock_embeddings

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)
        result = adapter.embed_batch(loaded_model, ["text 1", "text 2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_instance.encode.assert_called_once()