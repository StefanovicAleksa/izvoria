# Path: app/features/rag_reranking/data/cross_encoder_adapter.py

import os
import gc
import logging
from typing import Any, List

import torch

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.utils.torchcodec_guard import block_torchcodec_import
from app.features.rag_reranking.domain.interfaces import IRerankerAdapter
from app.features.rag_reranking.exceptions.model import RerankInferenceError
from app.features.rag_reranking.utils.error_mapper import map_rerank_error

logger = logging.getLogger(__name__)


class CrossEncoderRerankerAdapter(IRerankerAdapter):
    """
    Adapter for a sentence-transformers CrossEncoder (e.g. bge-reranker-v2-m3), fp16 on CUDA.
    """

    # Query + contextualized chunk (~250 words of transcript plus generated context) fits well within this.
    MAX_SEQ_LENGTH = 1024
    BATCH_SIZE = 16

    @staticmethod
    def _get_cross_encoder(model_path: str) -> Any:
        # Scoped torchcodec block: see app/core/model_manager/utils/torchcodec_guard.py
        with block_torchcodec_import():
            from sentence_transformers import CrossEncoder

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
            return CrossEncoder(
                model_path,
                device=device,
                local_files_only=True,
                model_kwargs=model_kwargs,
                max_length=CrossEncoderRerankerAdapter.MAX_SEQ_LENGTH
            )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Reranker model directory missing: {manifest.model_path}")

            logger.info(f"Loading cross-encoder reranker from {manifest.model_path}...")
            model = self._get_cross_encoder(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_rerank_error(e, context=f"Loading cross-encoder reranker: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_rerank_error(e, context="Unloading cross-encoder reranker") from e

    def score(self, model: LoadedModel, query: str, texts: List[str]) -> List[float]:
        if not getattr(model, "model_instance", None):
            raise RerankInferenceError("Cannot rerank: the cross-encoder is not loaded.")

        if not texts:
            return []

        try:
            encoder: Any = model.model_instance
            scores = encoder.predict(
                [(query, text) for text in texts],
                batch_size=self.BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return [float(score) for score in scores]
        except Exception as e:
            raise map_rerank_error(e, context="Scoring query/chunk pairs") from e
