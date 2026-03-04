# Canvas: phi3_adapter_v5
# Path: app/features/nlp_router/data/phi3_adapter.py

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.nlp_router.domain.interfaces import ILLMRouterAdapter
from app.features.nlp_router.domain.entities import TranscriptChunk, RouterReasoning
from app.features.nlp_router.data.prompts import ROUTER_SYSTEM_PROMPT, build_user_prompt
from app.features.nlp_router.utils.error_mapper import map_nlp_router_error


class Phi3RouterAdapter(ILLMRouterAdapter):
    """
    Executes local LLM inference using native Hugging Face Transformers.
    Optimized for the unquantized Phi-3.5-mini-instruct model in PyTorch/Safetensors format.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        if not os.path.exists(manifest.model_path):
            raise map_nlp_router_error(FileNotFoundError(manifest.model_path))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                manifest.model_path,
                local_files_only=True
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                manifest.model_path,
                device_map="cuda",
                torch_dtype=torch.bfloat16,  # Use bfloat16 to fit in 12GB VRAM efficiently
                local_files_only=True
            )

            payload = {
                "model": hf_model,
                "tokenizer": tokenizer
            }
            return LoadedModel(manifest=manifest, model_instance=payload)

        except Exception as e:
            raise map_nlp_router_error(e, context="Model Load Initialization")

    def unload(self, model: LoadedModel) -> bool:
        if hasattr(model, "model_instance"):
            del model.model_instance["model"]
            del model.model_instance["tokenizer"]
            del model.model_instance

        # Explicitly clear the PyTorch CUDA allocator cache to free the VRAM for other models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    def extract_visual_cues(self, model: LoadedModel, chunk: TranscriptChunk) -> RouterReasoning:
        payload = model.model_instance
        hf_model = payload["model"]
        tokenizer = payload["tokenizer"]

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(chunk.text_block)}
        ]

        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(hf_model.device)

            outputs = hf_model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=0.1,  # Ultra-low for deterministic logic
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Extract only the newly generated tokens
            generated_tokens = outputs[0][inputs.shape[1]:]
            raw_output_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Strip Markdown JSON wrappers if the LLM hallucinated them
            start_idx = raw_output_str.find('{')
            end_idx = raw_output_str.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                raw_json_str = raw_output_str[start_idx:end_idx + 1]
            else:
                raw_json_str = raw_output_str  # Fallback to raw string

            parsed_data = json.loads(raw_json_str)
            return RouterReasoning.model_validate(parsed_data)

        except Exception as e:
            context_msg = f"Chunk: {chunk.start_sec}s - {chunk.end_sec}s"
            raise map_nlp_router_error(e, context=context_msg)