# Canvas: deberta_adapter_v3
# Path: app/features/hallucination_scrubber/data/deberta_adapter.py

import os
import gc
from typing import List, Dict, Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.hallucination_scrubber.domain.interfaces import IZeroShotAdapter
from app.features.hallucination_scrubber.utils.error_mapper import map_scrubber_error


class DebertaAdapter(IZeroShotAdapter):
    """
    Executes DeBERTa-v3-base-mnli using the HuggingFace Transformers pipeline.
    This is an Encoder-only classification model, meaning it physically cannot
    generate new text or hallucinate—it only outputs probability scores.
    """

    @staticmethod
    def _get_pipeline(model_path: str) -> Any:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1

        return pipeline(
            "zero-shot-classification",
            model=model_path,
            device=device,
            local_files_only=True
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"DeBERTa model directory missing: {manifest.model_path}")

            classifier = self._get_pipeline(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=classifier)
        except Exception as e:
            raise map_scrubber_error(e, context="Load DeBERTa Zero-Shot Adapter") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
                model.model_instance = None

            gc.collect()

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        except Exception as e:
            raise map_scrubber_error(e, context="Unload DeBERTa Zero-Shot Adapter") from e

    def classify(self, model: LoadedModel, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """Evaluates a single string against the candidate labels."""
        try:
            classifier: Any = model.model_instance

            # CRITICAL FIX: multi_label=True uses independent Sigmoid evaluation.
            # This prevents Softmax dilution across overlapping candidate labels.
            result = classifier(text, candidate_labels, multi_label=True)

            return dict(zip(result["labels"], result["scores"]))
        except Exception as e:
            raise map_scrubber_error(e, context="DeBERTa Inference (Single)") from e

    def classify_batch(self, model: LoadedModel, texts: List[str], candidate_labels: List[str]) -> List[
        Dict[str, float]]:
        """Evaluates an array of strings simultaneously on the GPU."""
        try:
            classifier: Any = model.model_instance

            # CRITICAL FIX: multi_label=True
            results = classifier(texts, candidate_labels, multi_label=True)

            batch_scores = []
            for res in results:
                batch_scores.append(dict(zip(res["labels"], res["scores"])))

            return batch_scores
        except Exception as e:
            raise map_scrubber_error(e, context="DeBERTa Inference (Batch)") from e