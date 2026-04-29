# Canvas: retrieval_scrubber_slm_adapter_v2
# Path: app/features/retrieval_hallucination_scrubber/data/slm_adapter.py

import os
import gc
import json
import uuid
import re
from typing import Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.retrieval_hallucination_scrubber.domain.interfaces import IRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit
from app.features.retrieval_hallucination_scrubber.data.prompts import (
    RETRIEVAL_SCRUBBER_SYSTEM_PROMPT,
    RETRIEVAL_SCRUBBER_USER_TEMPLATE
)
from app.features.retrieval_hallucination_scrubber.utils.error_mapper import map_scrubber_error


class QwenRetrievalScrubberAdapter(IRetrievalScrubberAdapter):
    """
    Adapter for executing Qwen 2.5 3B via llama.cpp.
    Utilizes regex fallback parsing to ensure stability over strict grammar constraints.
    """

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"SLM model file missing: {manifest.model_path}")
            llm = self._get_llama_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_scrubber_error(e, context=f"Load Scrubber Adapter: {manifest.model_path}") from e

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
            raise map_scrubber_error(e, context="Unload Scrubber Adapter") from e

    def evaluate_relevance(
        self,
        model: LoadedModel,
        chunk_id: uuid.UUID,
        user_prompt: str,
        text_chunk: str
    ) -> ScrubAudit:
        try:
            llm: Any = model.model_instance

            user_content = RETRIEVAL_SCRUBBER_USER_TEMPLATE.format(
                user_prompt=user_prompt,
                text_chunk=text_chunk
            )

            messages = [
                {"role": "system", "content": RETRIEVAL_SCRUBBER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            response = llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,  # Zero temperature for maximum determinism
                max_tokens=150
            )

            raw_content = response["choices"][0]["message"]["content"]

            # Robust JSON extraction to prevent parsing errors
            json_match = re.search(r'\{.*}', raw_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
            else:
                parsed_data = json.loads(raw_content)

            return ScrubAudit(
                chunk_id=chunk_id,
                is_relevant=parsed_data.get("is_relevant", True),  # Default to True on partial parse failure
                discard_reason=parsed_data.get("discard_reason")
            )

        except Exception as e:
            raise map_scrubber_error(e, context=f"Evaluate relevance inference for chunk {chunk_id}") from e