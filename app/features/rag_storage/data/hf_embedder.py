# Path: app/features/rag_storage/data/hf_embedder.py

import os
import gc
import logging
from typing import List, Any
import torch

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.utils.torchcodec_guard import block_torchcodec_import
from app.features.rag_storage.domain.interfaces import IEmbeddingAdapter
from app.features.rag_storage.utils.error_mapper import map_storage_error

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingAdapter(IEmbeddingAdapter):

    @staticmethod
    def _get_embedding_model(model_path: str) -> Any:
        # Scoped torchcodec block: see app/core/model_manager/utils/torchcodec_guard.py
        with block_torchcodec_import():
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            return SentenceTransformer(
                model_path,
                device=device,
                model_kwargs=model_kwargs,
                local_files_only=True
            )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Embedding model directory missing: {manifest.model_path}")

            logger.info(f"Loading embedding model from {manifest.model_path}...")
            model = self._get_embedding_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_storage_error(e, context="Loading HuggingFace embedding model") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_storage_error(e, context="Unloading HuggingFace embedding model") from e

    def embed_batch(self, model: LoadedModel, texts: List[str]) -> List[List[float]]:
        if not getattr(model, "model_instance", None):
            raise Exception("Cannot embed texts: The embedding model is not loaded.")

        if not texts:
            return []

        try:
            embedder: Any = model.model_instance
            embeddings = embedder.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            raise map_storage_error(e, context="Executing embed_batch inference") from e
