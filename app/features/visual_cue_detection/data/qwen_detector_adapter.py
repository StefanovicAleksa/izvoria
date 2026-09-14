# Canvas: qwen_detector_adapter_v3
# Path: app/features/visual_cue_detection/data/qwen_detector_adapter.py

import json
import gc
import torch
from typing import Any
from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.visual_cue_detection.domain.interfaces import IVisualCueDetectorAdapter
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.visual_cue_detection.data.prompts import VISUAL_CUE_SYSTEM_PROMPT, CUE_DETECTION_SCHEMA
from app.features.visual_cue_detection.utils.error_mapper import map_detection_error


class QwenDetectorAdapter(IVisualCueDetectorAdapter):
    """
    Adapter for running Qwen 2.5 (GGUF) via llama.cpp to detect visual cues.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        llm = Llama(
            model_path=manifest.model_path,
            n_gpu_layers=-1,
            flash_attn=True,
            n_ctx=2048,
            verbose=False
        )
        return LoadedModel(manifest=manifest, model_instance=llm)

    def unload(self, model: LoadedModel) -> bool:
        """Purges the model from VRAM and flushes the CUDA cache."""
        if model.model_instance:
            # FIX: Explicitly close the C++ object before deletion
            if hasattr(model.model_instance, "close"):
                model.model_instance.close()
            del model.model_instance
            model.model_instance = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def detect_cues(self, model: LoadedModel, chunk: ContextualizedChunk) -> ContextualizedChunk:
        llm: Llama = model.model_instance
        user_message = f"Context: {chunk.context_string}\n\nChunk Text: {chunk.original_text}"

        try:
            messages = [
                ChatCompletionRequestSystemMessage(role="system", content=VISUAL_CUE_SYSTEM_PROMPT),
                ChatCompletionRequestUserMessage(role="user", content=user_message)
            ]

            response_format: Any = {
                "type": "json_object",
                "schema": CUE_DETECTION_SCHEMA
            }

            response = llm.create_chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=0.0,  # A verdict must be reproducible across reruns
                max_tokens=250
            )

            raw_content = response["choices"][0]["message"]["content"]
            parsed_data = json.loads(raw_content)

            needs_visual_flag = parsed_data.get("needs_visual", False)

            return chunk.model_copy(update={"needs_visual": needs_visual_flag})

        except Exception as e:
            raise map_detection_error(e, context=f"Detecting cues for chunk {chunk.chunk_id}") from e