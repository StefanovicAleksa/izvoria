# Canvas: retrieval_scrubber_slm_adapter_v2
# Path: app/features/retrieval_hallucination_scrubber/data/slm_adapter.py

import os
import gc
import json
import uuid
import re
from typing import Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.utils.llama_cpp_chat import build_llama, chat_completion, strip_reasoning
from app.features.retrieval_hallucination_scrubber.domain.interfaces import IRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit
from app.features.retrieval_hallucination_scrubber.data.prompts import (
    RETRIEVAL_SCRUBBER_SYSTEM_PROMPT,
    RETRIEVAL_SCRUBBER_USER_TEMPLATE
)
from app.features.retrieval_hallucination_scrubber.exceptions.pipeline import JSONScrubParseError
from app.features.retrieval_hallucination_scrubber.utils.error_mapper import map_scrubber_error


class QwenRetrievalScrubberAdapter(IRetrievalScrubberAdapter):
    """
    Adapter for executing the scrubber's instruct GGUF via llama.cpp: Qwen2.5-7B in-process on the appliance, or the
    shared Qwen3.5-9B behind llama-server when hosted (LLM_BACKEND=llama_server), where several chunks are judged
    in parallel slots. Utilizes regex fallback parsing to ensure stability over strict grammar constraints.
    """

    N_CTX = 4096
    # f16 is what llama.cpp allocates when no KV type is given, which is what this adapter always used
    KV_CACHE_TYPE = "f16"

    @classmethod
    def _get_llama_model(cls, model_path: str) -> Any:
        return build_llama(model_path, cls.N_CTX, cls.KV_CACHE_TYPE)

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

            response = chat_completion(
                llm,
                messages,
                enable_thinking=False,  # a verdict, not a reasoned answer; Qwen2.5's template ignores the flag
                response_format={"type": "json_object"},
                temperature=0.0,  # Zero temperature for maximum determinism
                max_tokens=150
            )

            raw_content = strip_reasoning(response["choices"][0]["message"].get("content") or "")

            # Robust JSON extraction to prevent parsing errors
            json_match = re.search(r'\{.*}', raw_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
            else:
                parsed_data = json.loads(raw_content)

            if not isinstance(parsed_data, dict):
                raise JSONScrubParseError(f"Scrubber verdict is not a JSON object: {raw_content[:200]!r}")

            return ScrubAudit(
                chunk_id=chunk_id,
                is_relevant=parsed_data.get("is_relevant", True),  # Default to True on partial parse failure
                discard_reason=parsed_data.get("discard_reason")
            )

        except Exception as e:
            raise map_scrubber_error(e, context=f"Evaluate relevance inference for chunk {chunk_id}") from e