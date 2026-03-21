# Canvas: llama_adapter_v5
# Path: app/features/domain_adaptation/data/llama_adapter.py

import os
import json
import re
import gc
from typing import List, Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import ILlamaAdapter
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class LlamaAdapter(ILlamaAdapter):
    """
    Executes the quantized Llama 3.2 3B model via llama.cpp.
    Reads the dirty scout transcript and outputs JSON sub-domains.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            from llama_cpp import Llama

            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Llama model file missing: {manifest.model_path}")

            llm = Llama(
                model_path=manifest.model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Load Llama Adapter") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance

            # 1. Force Python Sweep
            gc.collect()

            # 2. Force PyTorch to release greedy cached VRAM to the OS so CTranslate2 can use it
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload Llama Adapter") from e

    def predict_subdomains(
            self, model: LoadedModel, scout_text: str, candidate_labels: List[str]
    ) -> List[str]:
        try:
            llm: Any = model.model_instance

            system_prompt = (
                "You are an expert legal/medical taxonomy router.\n"
                "Analyze the following dirty transcript and identify the sub-domains being discussed.\n"
                f"You MUST choose from the following exact labels: {', '.join(candidate_labels)}.\n"
                "Output ONLY a raw JSON array of the identified strings, e.g., [\"M_AND_A\", \"IP_LAW\"]."
            )

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scout_text}
                ],
                max_tokens=64,
                temperature=0.1
            )

            output_text = response['choices'][0]['message']['content'].strip()
            return self._parse_json_array(output_text)

        except Exception as e:
            raise map_domain_adaptation_error(e, context="Llama Adapter Inference") from e

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        try:
            match = re.search(r'\[.*]', text, re.DOTALL)
            if match:
                clean_json = match.group(0)
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    return parsed

            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []