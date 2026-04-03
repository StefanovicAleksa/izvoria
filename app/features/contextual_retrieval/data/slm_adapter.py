# Canvas: contextual_slm_adapter_v5
# Path: app/features/contextual_retrieval/data/slm_adapter.py

import os
import gc
from typing import Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.contextual_retrieval.domain.interfaces import ISLMContextAdapter
from app.features.contextual_retrieval.data.prompts import (
    CONTEXTUAL_SYSTEM_PROMPT,
    CONTEXTUAL_USER_PROMPT_TEMPLATE
)
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error


class SLMContextAdapter(ISLMContextAdapter):
    """
    Adapter for executing Small Language Models (e.g., Llama 3.2 1B/3B or Qwen 2.5) via llama.cpp.
    Optimized for KV Cache reuse to rapidly generate context for hundreds of chunks.
    """

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        """
        FACTORY METHOD & LAZY IMPORT
        Prevents C-extension collisions with PyTorch during application startup.
        """
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # Offload all layers to the GPU
            n_ctx=32768,  # Large context window to hold the full document (adjust based on model specs)
            verbose=False  # Suppress spammy C++ token generation logs
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"SLM model file missing: {manifest.model_path}")

            llm = self._get_llama_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_contextual_error(e, context=f"Load SLM Adapter: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            # FIX: Explicitly close the C++ object before deletion
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_contextual_error(e, context="Unload SLM Adapter") from e

    def generate_context(
            self,
            model: LoadedModel,
            full_document_text: str,
            chunk_text: str
    ) -> str:
        try:
            llm: Any = model.model_instance

            user_content = CONTEXTUAL_USER_PROMPT_TEMPLATE.format(
                full_document_text=full_document_text,
                chunk_text=chunk_text
            )

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": CONTEXTUAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=150,  # Cap the output to ensure it remains a "short succinct context"
                temperature=0.0,  # Zero creativity for deterministic, factual context
            )

            output_text = response['choices'][0]['message']['content'].strip()

            # Defensive cleaning in case the SLM ignores the "no conversational filler" rule
            if output_text.lower().startswith("here is the context"):
                output_text = output_text.split(":", 1)[-1].strip()

            return output_text

        except Exception as e:
            raise map_contextual_error(e, context="SLM Context Generation Inference") from e