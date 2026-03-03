# Canvas: nlp_router_phi3_adapter_v2
# Path: app/features/nlp_router/data/phi3_adapter.py

import os
import json
from llama_cpp import (
    Llama,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestResponseFormat
)

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.nlp_router.domain.interfaces import ILLMRouterAdapter
from app.features.nlp_router.domain.entities import TranscriptChunk, RouterReasoning
from app.features.nlp_router.data.prompts import ROUTER_SYSTEM_PROMPT, build_user_prompt
from app.features.nlp_router.utils.error_mapper import map_nlp_router_error


class Phi3RouterAdapter(ILLMRouterAdapter):
    """
    Executes local, strictly-typed LLM inference using llama.cpp.
    Optimized for the Phi-3.5-mini-instruct GGUF model.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        if not os.path.exists(manifest.model_path):
            raise map_nlp_router_error(FileNotFoundError(manifest.model_path))

        try:
            # n_gpu_layers=-1 offloads all 32 layers of Phi-3.5 to the RTX 3060 VRAM.
            # n_ctx=4096 is optimal: large enough for a 5-min transcript, small enough to save memory.
            llm = Llama(
                model_path=manifest.model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False  # Suppress excessive C++ stdout spam
            )
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_nlp_router_error(e, context="Model Load Initialization")

    def unload(self, model: LoadedModel) -> bool:
        if hasattr(model, "model_instance"):
            # Deleting the object triggers the C++ destructor, instantly freeing the VRAM
            del model.model_instance
        return True

    def extract_visual_cues(self, model: LoadedModel, chunk: TranscriptChunk) -> RouterReasoning:
        llm: Llama = model.model_instance

        # 1. Construct Strictly Typed Messages
        system_msg = ChatCompletionRequestSystemMessage(
            role="system",
            content=ROUTER_SYSTEM_PROMPT
        )
        user_msg = ChatCompletionRequestUserMessage(
            role="user",
            content=build_user_prompt(chunk.text_block)
        )
        messages = [system_msg, user_msg]

        # 2. Construct Strictly Typed Response Format
        response_format = ChatCompletionRequestResponseFormat(type="json_object")

        try:
            # 3. Execute inference with JSON constraints
            response = llm.create_chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=0.1,  # Ultra-low temperature for strict, deterministic logic
                max_tokens=2048  # We only need a short reasoning string and a few timestamps
            )

            # Extract the string payload from the ChatML response format
            raw_json_str = str(response["choices"][0]["message"]["content"])

            # Parse the string into a raw Python dictionary
            parsed_data = json.loads(raw_json_str)

            # Strictly validate the dictionary against our Domain Entity rules
            return RouterReasoning.model_validate(parsed_data)

        except Exception as e:
            # The Error Mapper will catch JSONDecodeErrors or ValidationErrors if the LLM hallucinated
            context_msg = f"Chunk: {chunk.start_sec}s - {chunk.end_sec}s"
            raise map_nlp_router_error(e, context=context_msg)