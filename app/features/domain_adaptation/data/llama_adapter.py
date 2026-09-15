# Canvas: llama_adapter_v11
# Path: app/features/domain_adaptation/data/llama_adapter.py

import os
import json
import gc
import logging
from typing import Any, Dict, List

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import ILlamaAdapter
from app.features.domain_adaptation.data.prompts import ROUTER_SYSTEM_PROMPT_TEMPLATE
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

logger = logging.getLogger(__name__)


class LlamaAdapter(ILlamaAdapter):
    """
    Executes the quantized Llama 3.2 3B model via llama.cpp.
    Scores every candidate sub-domain, constrained by a GBNF grammar that only allows the real labels.
    """

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

    @staticmethod
    def _score_grammar(labels: List[str]) -> str:
        """A JSON object with every label exactly once, in order, each scored with an integer from 0 to 10."""
        pairs = ' "," space '.join(f'"\\"{label}\\":" space score' for label in labels)
        return f'root ::= "{{" space {pairs} space "}}" space\nscore ::= "10" | [0-9]\nspace ::= " "*\n'

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
            # FIX: Explicitly close the C++ object before deletion
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload Llama Adapter") from e

    def predict_subdomains(
            self, model: LoadedModel, scout_text: str, candidate_labels: Dict[str, str]
    ) -> List[str]:
        """
        Rates every candidate sub-domain 0-10 and keeps those scoring at least DOMAIN_ROUTER_MIN_SCORE, best first.
        Asked to list "the sub-domains", the model copied every label in order; a score per topic lets it say
        "not discussed".
        """
        if not candidate_labels:
            return []

        try:
            llm: Any = model.model_instance
            labels = list(candidate_labels)
            topic_descriptions = "\n".join(f"- {label}: {description}" for label, description in candidate_labels.items())

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT_TEMPLATE.format(topic_descriptions=topic_descriptions)},
                    {"role": "user", "content": scout_text}
                ],
                max_tokens=64 + 32 * len(labels),
                temperature=0.0,
                grammar=self._get_grammar(self._score_grammar(labels))
            )

            scores = json.loads(response['choices'][0]['message']['content'].strip())
            if not isinstance(scores, dict):
                return []

            logger.info(f"Sub-domain router scores: {scores}")
            selected = [
                (label, scores[label]) for label in labels
                if isinstance(scores.get(label), (int, float)) and scores[label] >= settings.DOMAIN_ROUTER_MIN_SCORE
            ]
            selected.sort(key=lambda item: item[1], reverse=True)
            return [label for label, _ in selected]

        except json.JSONDecodeError as e:
            logger.warning(f"Sub-domain router produced invalid JSON; no sub-domains selected. Reason: {e}")
            return []
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Llama Adapter Inference") from e
