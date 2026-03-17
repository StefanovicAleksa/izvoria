# Canvas: sat_adapter_v7
# Path: app/features/sentence_reconstruction/data/sat_adapter.py

import os
import logging
from typing import List
import torch
from wtpsplit import SaT

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.features.sentence_reconstruction.domain.interfaces import ISentenceAdapter

# Explicit exception imports
from app.features.sentence_reconstruction.exceptions import (
    SaTModelLoadError,
    SaTOOMError,
    SaTExecutionError
)

logger = logging.getLogger(__name__)


class SaTAdapter(ISentenceAdapter):
    """
    Adapter for the Segment Any Text (SaT) engine.
    Conforms strictly to the Orchestrator's IModelLoader interface.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        logger.info(f"Initializing SaT model from local manifest path: {manifest.model_path}")

        try:
            # Enforce offline mode via environment variable since SaT() doesn't accept hub_kwargs
            os.environ["HF_HUB_OFFLINE"] = "1"

            model_instance = SaT(manifest.model_path)

            if torch.cuda.is_available():
                model_instance.half().to("cuda")

            return LoadedModel(
                manifest=manifest,
                model_instance=model_instance
            )
        except OSError as e:
            logger.error(f"Failed to find SaT model {manifest.model_path} on disk: {str(e)}")
            raise SaTModelLoadError(f"Model files missing or unreachable locally: {str(e)}") from e
        except torch.cuda.OutOfMemoryError as e:
            logger.critical("CUDA OOM during SaT load.")
            raise SaTOOMError("Insufficient VRAM to load SaT.") from e
        except Exception as e:
            logger.error(f"Unexpected error loading SaT: {str(e)}")
            raise SaTExecutionError(f"Initialization failed: {str(e)}") from e

    def unload(self, model: LoadedModel) -> bool:
        logger.info("Releasing SaT model VRAM...")
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to unload SaT model: {str(e)}")
            return False

    def segment_text(self, model: LoadedModel, text: str) -> List[str]:
        try:
            sat_instance: SaT = model.model_instance
            sentences = sat_instance.split(text)
            return [s.strip() for s in sentences if s.strip()]
        except torch.cuda.OutOfMemoryError as e:
            logger.critical("CUDA OOM during sentence splitting.")
            raise SaTOOMError("GPU ran out of memory processing text.") from e
        except Exception as e:
            logger.error(f"SaT splitting failed: {str(e)}")
            raise SaTExecutionError(f"Inference failed: {str(e)}") from e