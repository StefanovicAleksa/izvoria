# Canvas: rag_retrieval_embedding_adapter_v2
# Path: app/features/rag_retrieval/data/embedding_adapter.py

import os
import gc
import logging
from typing import List, Any
import torch

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.utils.torchcodec_guard import block_torchcodec_import
from app.features.rag_retrieval.domain.interfaces import IPromptEmbeddingAdapter
from app.features.rag_retrieval.utils.error_mapper import map_retrieval_error

logger = logging.getLogger(__name__)

class HuggingFacePromptAdapter(IPromptEmbeddingAdapter):

    @staticmethod
    def _get_embedding_model(model_path: str) -> Any:
        # Scoped torchcodec block (same fix as batch ingestion): see torchcodec_guard.py
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
                raise FileNotFoundError(f"Embedding model missing at: {manifest.model_path}")

            logger.info(f"Loading single-prompt embedder from {manifest.model_path}...")
            model = self._get_embedding_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_retrieval_error(e, context="Loading Prompt Embedding Model") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_retrieval_error(e, context="Unloading Prompt Embedding Model") from e

    def embed_prompt(self, model: LoadedModel, text: str) -> List[float]:
        if not getattr(model, "model_instance", None):
            raise Exception("Cannot embed prompt: The model is not loaded in VRAM.")

        if not text.strip():
            raise ValueError("Provided search prompt is empty.")

        try:
            embedder: Any = model.model_instance
            # Encode a single string, returning a flat list of floats
            embedding = embedder.encode(
                text,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            raise map_retrieval_error(e, context="Executing prompt vectorization") from e
