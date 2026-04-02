# Canvas: llama_detector_adapter_v6
# Path: app/features/visual_cue_detection/data/llama_detector_adapter.py

import json
from typing import Any
from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage
)

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.visual_cue_detection.domain.interfaces import IVisualCueDetectorAdapter
from app.features.visual_cue_detection.data.prompts import VISUAL_CUE_SYSTEM_PROMPT, CUE_DETECTION_SCHEMA
from app.features.visual_cue_detection.utils.error_mapper import map_detection_error


class LlamaDetectorAdapter(IVisualCueDetectorAdapter):
    """
    NLP Data Adapter that uses a local SLM to detect visual dependencies in text.
    Enforces strict JSON schema output to guarantee pipeline stability.
    """

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
                temperature=0.1,
                max_tokens=250  # Increased to allow room for the CoT 'reasoning' generation
            )

            raw_content = response["choices"][0]["message"]["content"]
            parsed_data = json.loads(raw_content)

            # The reasoning string is generated to improve accuracy, but we only need the boolean flag
            needs_visual_flag = parsed_data.get("needs_visual", False)

            return chunk.model_copy(update={"needs_visual": needs_visual_flag})

        except Exception as e:
            raise map_detection_error(e, context=f"Detecting cues for chunk {chunk.chunk_id}") from e