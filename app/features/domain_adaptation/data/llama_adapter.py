# Canvas: llama_adapter_v1
# Path: app/features/domain_adaptation/data/llama_adapter.py

import os
import json
import re
from typing import List

# Assuming llama_cpp is your engine for .gguf models
from llama_cpp import Llama

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
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Llama model file missing: {manifest.model_path}")

            # n_gpu_layers=-1 offloads all layers to the GTX 3060
            # n_ctx=4096 is plenty for 10 minutes of dense text
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
            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload Llama Adapter") from e

    def predict_subdomains(
            self, model: LoadedModel, scout_text: str, candidate_labels: List[str]
    ) -> List[str]:
        try:
            llm: Llama = model.model_instance

            system_prompt = (
                "You are an expert legal/medical taxonomy router. "
                "Analyze the following dirty transcript and identify the sub-domains being discussed. "
                f"You MUST choose from the following exact labels: {', '.join(candidate_labels)}. "
                "Output ONLY a raw JSON array of the identified strings, e.g., [\"M_AND_A\", \"IP_LAW\"]."
            )

            # Llama 3 instructional format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{scout_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            response = llm(
                prompt,
                max_tokens=64,
                temperature=0.1,  # Low temp for deterministic JSON output
                stop=["<|eot_id|>"]
            )

            output_text = response['choices'][0]['text'].strip()
            return self._parse_json_array(output_text)

        except Exception as e:
            raise map_domain_adaptation_error(e, context="Llama Adapter Inference") from e

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        """Safely extracts a JSON array, even if Llama wraps it in markdown blocks."""
        try:
            # Strip potential markdown formatting (e.g., ```json [...] ```)
            match = re.search(r'\[.*]', text, re.DOTALL)
            if match:
                clean_json = match.group(0)
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    return parsed

            # Fallback
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError as e:
            raise map_domain_adaptation_error(e, context=f"Failed to parse Llama output: {text}") from e