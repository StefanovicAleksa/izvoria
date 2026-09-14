# Canvas: contextual_slm_adapter_v7
# Path: app/features/contextual_retrieval/data/slm_adapter.py

import os
import gc
import re
import logging
from typing import Any

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.contextual_retrieval.domain.interfaces import ISLMContextAdapter
from app.features.contextual_retrieval.data.prompts import (
    CONTEXTUAL_SYSTEM_PROMPT,
    CONTEXTUAL_USER_PROMPT_TEMPLATE
)
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error

logger = logging.getLogger(__name__)

# A speaker label the SLM echoed at the start of its answer: "[Speaker_0 @ 00:12:31]" (colon optional)
# or "Speaker_0:". "Speaker_0 describes..." is real prose and is left untouched.
_LEADING_SPEAKER_LABEL = re.compile(r"^\s*(?:\[[^\]\n]*@\s*\d{1,2}:\d{2}:\d{2}\]\s*:?|Speaker_\w+\s*:)\s*")


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
            # Large context window to hold the document (or the section of it) the chunk is situated in.
            # ContextualizerAPI sizes document sections against this same setting.
            n_ctx=settings.CONTEXTUAL_RETRIEVAL_N_CTX,
            flash_attn=True,  # CRITICAL FIX: Hardware-accelerated attention prevents OOM on 32k contexts
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

    def count_tokens(self, model: LoadedModel, text: str) -> int:
        try:
            llm: Any = model.model_instance
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception as e:
            raise map_contextual_error(e, context="SLM token counting for chunk windowing") from e

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

            while True:
                stripped = _LEADING_SPEAKER_LABEL.sub("", output_text, count=1).strip()
                if stripped == output_text:
                    break
                output_text = stripped

            if not output_text:
                logger.warning("Context SLM returned only a speaker label; storing an empty context for this chunk.")

            return output_text

        except Exception as e:
            raise map_contextual_error(e, context="SLM Context Generation Inference") from e
