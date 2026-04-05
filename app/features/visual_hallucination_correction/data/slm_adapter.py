# Canvas: visual_hallucination_slm_adapter_v1
# Path: app/features/visual_hallucination_correction/data/slm_adapter.py

import os
import gc
import json
import uuid
from typing import Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_hallucination_correction.domain.interfaces import IVisualHallucinationAdapter
from app.features.visual_hallucination_correction.domain.entities import VisualHallucinationAudit
from app.features.visual_hallucination_correction.data.prompts import (
    HALLUCINATION_AUDIT_SYSTEM_PROMPT,
    HALLUCINATION_AUDIT_USER_TEMPLATE
)
from app.features.visual_hallucination_correction.utils.error_mapper import map_hallucination_error


class QwenHallucinationAdapter(IVisualHallucinationAdapter):
    """
    Adapter for executing Qwen 2.5 3B via llama.cpp to evaluate text/visual alignment.
    Inherits from IModelLoader to support strict Orchestrator VRAM management.
    """

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        """
        FACTORY METHOD: Defers llama-cpp import to prevent C-extension collisions.
        """
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Offload fully to GPU
            n_ctx=4096,       # Standard context window is sufficient for chunk comparisons
            flash_attn=True,
            verbose=False     # Suppress spammy C++ logs in the Enterprise console
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"SLM model file missing: {manifest.model_path}")

            llm = self._get_llama_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_hallucination_error(e, context=f"Load Hallucination Adapter: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_hallucination_error(e, context="Unload Hallucination Adapter") from e

    def evaluate_visual_match(
        self,
        model: LoadedModel,
        chunk_id: uuid.UUID,
        text_chunk: str,
        visual_description: str
    ) -> VisualHallucinationAudit:
        try:
            llm: Any = model.model_instance

            user_content = HALLUCINATION_AUDIT_USER_TEMPLATE.format(
                text_chunk=text_chunk,
                visual_description=visual_description
            )

            messages = [
                {"role": "system", "content": HALLUCINATION_AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            # Enforce strict JSON output at the C++ inference level
            response_format: Any = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_match": {"type": "boolean"},
                        "discard_reason": {"type": ["string", "null"]}
                    },
                    "required": ["is_match", "discard_reason"]
                }
            }

            response = llm.create_chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=0.1,  # Near-zero creativity for strict auditing
                max_tokens=150
            )

            raw_content = response["choices"][0]["message"]["content"]
            parsed_data = json.loads(raw_content)

            return VisualHallucinationAudit(
                chunk_id=chunk_id,
                is_visual_accepted=parsed_data.get("is_match", False),
                discard_reason=parsed_data.get("discard_reason")
            )

        except Exception as e:
            raise map_hallucination_error(e, context=f"Evaluate visual match inference for chunk {chunk_id}") from e