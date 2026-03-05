# Canvas: phi3_adapter_v6
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
    Includes built-in retry logic to handle non-deterministic LLM output.
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
                torch_dtype=torch.bfloat16,
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
        except Exception as e:
            raise map_nlp_router_error(e, context="Tokenization Error")

        last_error = None
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Dynamically increase temperature on retries to break out of hallucination loops
                temp = 0.1 if attempt == 0 else 0.3 + (attempt * 0.2)

                outputs = hf_model.generate(
                    inputs,
                    max_new_tokens=2048,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                generated_tokens = outputs[0][inputs.shape[1]:]
                raw_output_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                start_idx = raw_output_str.find('{')
                end_idx = raw_output_str.rfind('}')

                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    raw_json_str = raw_output_str[start_idx:end_idx + 1]
                else:
                    raw_json_str = raw_output_str

                parsed_data = json.loads(raw_json_str)
                return RouterReasoning.model_validate(parsed_data)

            except Exception as e:
                last_error = e
                # Explicitly clear VRAM of the failed generation tensors before retrying
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        # If we exhausted all retries, bubble it up to the API layer for graceful degradation
        context_msg = f"Chunk: {chunk.start_sec}s - {chunk.end_sec}s | Failed after {max_retries} retries."
        raise map_nlp_router_error(last_error, context=context_msg)