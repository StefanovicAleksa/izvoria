# Path: app/features/domain_phonetic_correction/data/qwen_adapter.py

import os
import gc
import logging
from typing import List, Any, Dict

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticPatchAdapter
from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate
from app.features.domain_phonetic_correction.data.prompts import (
    PHONETIC_VERIFY_SYSTEM_PROMPT,
    PHONETIC_VERIFY_USER_TEMPLATE
)
from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error

logger = logging.getLogger(__name__)

# Characters of the sentence kept on each side of a candidate, so run-on sentences stay inside the context window
CONTEXT_CHARS = 600
_CHOICE_GRAMMAR = 'root ::= "A" | "B"\n'


class QwenDomainAdapter(IDomainPhoneticPatchAdapter):
    @staticmethod
    def _get_qwen_model(model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            flash_attn=True,
            verbose=False
        )

    @staticmethod
    def _get_grammar(grammar_str: str) -> Any:
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(grammar_str)

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model file missing: {manifest.model_path}")
            llm = self._get_qwen_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_phonetic_error(e, context=f"Load Qwen 2.5 Model: {manifest.model_path}") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance") and model.model_instance is not None:
                if hasattr(model.model_instance, "close"):
                    model.model_instance.close()
                del model.model_instance
                model.model_instance = None
            gc.collect()
            return True
        except Exception as e:
            raise map_phonetic_error(e, context="Unload Qwen 2.5 Model") from e

    def verify_candidates(
            self,
            model: LoadedModel,
            sentence_text: str,
            previous_text: str,
            candidates: List[PhoneticCandidate],
            domain: str
    ) -> List[bool]:
        """
        A candidate is confirmed only if the model picks the version with the term both when it is shown first and
        when it is shown second. Either order alone let a position bias accept ordinary words ("bond" -> "Bund").
        """
        if not candidates:
            return []

        try:
            llm: Any = model.model_instance
            grammar = self._get_grammar(_CHOICE_GRAMMAR)
            system_content = PHONETIC_VERIFY_SYSTEM_PROMPT.format(domain=domain)
            verdicts: List[bool] = []

            for candidate in candidates:
                heard_version = self._marked_window(sentence_text, candidate, candidate.heard)
                term_version = self._marked_window(sentence_text, candidate, candidate.replacement)
                term_first = self._choose(llm, grammar, system_content, previous_text, term_version, heard_version) == "A"
                term_second = self._choose(llm, grammar, system_content, previous_text, heard_version, term_version) == "B"
                verdicts.append(term_first and term_second)

            return verdicts

        except Exception as e:
            raise map_phonetic_error(e, context="Verify Phonetic Candidates") from e

    @staticmethod
    def _choose(llm: Any, grammar: Any, system_content: str, previous_text: str, version_a: str, version_b: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": PHONETIC_VERIFY_USER_TEMPLATE.format(
                previous_text=previous_text or "(none)", version_a=version_a, version_b=version_b
            )}
        ]
        response = llm.create_chat_completion(messages=messages, temperature=0.0, max_tokens=1, grammar=grammar)
        return response["choices"][0]["message"].get("content", "").replace('\x00', '').strip()

    @staticmethod
    def _marked_window(text: str, candidate: PhoneticCandidate, written: str) -> str:
        """The sentence around the candidate, with the candidate span replaced by `written` in [brackets]."""
        start = max(0, candidate.start - CONTEXT_CHARS)
        end = min(len(text), candidate.end + CONTEXT_CHARS)
        prefix = ("…" if start > 0 else "") + text[start:candidate.start]
        suffix = text[candidate.end:end] + ("…" if end < len(text) else "")
        return f"{prefix}[{written}]{suffix}"
