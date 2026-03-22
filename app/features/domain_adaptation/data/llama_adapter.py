# Canvas: llama_adapter_v6
# Path: app/features/domain_adaptation/data/llama_adapter.py

import os
import json
import gc
from typing import List, Any

from llama_cpp import Llama, LlamaGrammar
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import ILlamaAdapter
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class LlamaAdapter(ILlamaAdapter):
    """
    Executes the quantized Llama 3.2 3B model via llama.cpp.
    Strictly constrained using GBNF Grammar to output ONLY a valid JSON Array.
    """

    # GBNF Grammar defining a strict JSON string array: e.g., ["A", "B"]
    JSON_ARRAY_GRAMMAR = r'''
        root ::= "[" space elements "]" space
        elements ::= string ("," space string)* | ""
        string ::= "\"" [^"]* "\""
        space ::= " "*
    '''

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
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

            gc.collect()
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

            # Load the grammar constraint
            grammar = LlamaGrammar.from_string(self.JSON_ARRAY_GRAMMAR)

            system_prompt = (
                "You are an expert legal/medical taxonomy router.\n"
                "Analyze the following dirty transcript and identify the sub-domains being discussed.\n"
                f"You MUST choose from the following exact labels: {', '.join(candidate_labels)}.\n"
                "Output ONLY a raw JSON array of the identified strings."
            )

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scout_text}
                ],
                max_tokens=128,
                temperature=0.0,  # Zero temperature + Grammar = Total Determinism
                grammar=grammar  # Mathematical enforcement
            )

            output_text = response['choices'][0]['message']['content'].strip()

            # Because of GBNF, we can safely trust json.loads without regex hacks
            parsed = json.loads(output_text)
            return parsed if isinstance(parsed, list) else []

        except Exception as e:
            raise map_domain_adaptation_error(e, context="Llama Adapter Inference") from e