# Canvas: visual_synthesis_slm_adapter_v1
# Path: app/features/visual_context_synthesis/data/slm_adapter.py

import os
import gc
from typing import Any, List, Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_context_synthesis.domain.interfaces import IVisualSLMAdapter
from app.features.visual_context_synthesis.data.prompts import (
    VISUAL_SYNTHESIS_SYSTEM_PROMPT,
    VISUAL_SYNTHESIS_USER_TEMPLATE
)
from app.features.visual_context_synthesis.utils.error_mapper import map_visual_synthesis_error


class VisualSLMAdapter(IVisualSLMAdapter):
    """
    Adapter for Llama 3.2 3B (GGUF) via llama.cpp.
    Synthesizes the fuzzy-grouped visual timeline into a RAG-optimized paragraph.
    """

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Full GPU offloading
            n_ctx=4096,       # Synthesis tasks are relatively short
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"SLM model weights missing: {manifest.model_path}")

            llm = self._get_llama_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_visual_synthesis_error(e, context=f"Loading SLM: {manifest.model_path}")

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
            raise map_visual_synthesis_error(e, context="Unloading SLM")

    def synthesize_context(
        self,
        model: LoadedModel,
        visual_events: List[Dict[str, Any]]
    ) -> str:
        """
        Formats the timeline and executes the LLM synthesis.
        """
        try:
            llm: Any = model.model_instance

            # 1. Format the timeline for the prompt
            timeline_str = ""
            for event in visual_events:
                timeline_str += (
                    f"- [{event['start_sec']:.2f}s - {event['end_sec']:.2f}s]: "
                    f"{event['representative_description']}\n"
                )

            user_content = VISUAL_SYNTHESIS_USER_TEMPLATE.format(
                visual_timeline_events=timeline_str
            )

            # 2. Inference
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": VISUAL_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=200,
                temperature=0.0 # Deterministic
            )

            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            raise map_visual_synthesis_error(e, context="Visual SLM Synthesis Inference")