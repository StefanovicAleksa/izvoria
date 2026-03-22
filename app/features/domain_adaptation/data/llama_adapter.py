# Canvas: llama_adapter_v9
# Path: app/features/domain_adaptation/data/llama_adapter.py

import os
import json
import gc
from typing import List, Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import ILlamaAdapter
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class LlamaAdapter(ILlamaAdapter):
    """
    Executes the quantized Llama 3.2 3B model via llama.cpp.
    Strictly constrained using GBNF Grammar to output ONLY a valid JSON Array.
    """

    JSON_ARRAY_GRAMMAR = r'''
        root ::= "[" space elements "]" space
        elements ::= string ("," space string)* | ""
        string ::= "\"" [^"]* "\""
        space ::= " "*
    '''

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        """
        FACTORY METHOD & LAZY IMPORT
        Prevents llama-cpp-python from injecting its C-extensions into the global
        namespace during App Startup, which causes CUDA symbol collisions with PyTorch.
        """
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )

    @staticmethod
    def _get_grammar(grammar_str: str) -> Any:
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(grammar_str)

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Llama model file missing: {manifest.model_path}")

            llm = self._get_llama_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Load Llama Adapter") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload Llama Adapter") from e

    def predict_subdomains(
            self, model: LoadedModel, scout_text: str, candidate_labels: List[str]
    ) -> List[str]:
        try:
            llm: Any = model.model_instance
            grammar = self._get_grammar(self.JSON_ARRAY_GRAMMAR)

            system_prompt = (
                "You are an expert taxonomy router. Analyze the transcript and identify the sub-domains.\n"
                f"Labels: {', '.join(candidate_labels)}.\n"
                "Output ONLY a raw JSON array."
            )

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scout_text}
                ],
                max_tokens=1024,
                temperature=0.0,
                grammar=grammar
            )

            output_text = response['choices'][0]['message']['content'].strip()

            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]

            parsed = json.loads(output_text.strip())
            return parsed if isinstance(parsed, list) else []

        except json.JSONDecodeError as e:
            print(f"\n[WARNING] Llama produced invalid JSON. Degrading gracefully.\nReason: {str(e)}")
            return []
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Llama Adapter Inference") from e