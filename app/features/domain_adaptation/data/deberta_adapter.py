# Canvas: deberta_adapter_v3
# Path: app/features/domain_adaptation/data/deberta_adapter.py

import os
from typing import List
from transformers import pipeline

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import IClassifierAdapter
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

class DebertaClassifierAdapter(IClassifierAdapter):
    """
    Adapter for DeBERTa v3 XSmall, executed on the GPU.
    """
    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Classifier model directory missing: {manifest.model_path}")

            # device=0 forces the pipeline to load onto the first NVIDIA GPU
            hf_pipeline = pipeline(
                "zero-shot-classification",
                model=manifest.model_path,
                tokenizer=manifest.model_path,
                device=0,
                local_files_only=True
            )
            return LoadedModel(manifest=manifest, model_instance=hf_pipeline)
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Load GPU DeBERTa Classifier") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload GPU DeBERTa Classifier") from e

    def classify_domain(
        self, model: LoadedModel, text: str, candidate_labels: List[str]
    ) -> str:
        try:
            classifier_pipeline = model.model_instance
            result = classifier_pipeline(
                text,
                candidate_labels,
                multi_label=False,
                truncation=True,
                max_length=512
            )
            return result["labels"][0]
        except Exception as e:
            raise map_domain_adaptation_error(e, context="GPU Zero-Shot Classification") from e