# Path: app/features/qa_generation/data/llama_cpp_generation_adapter.py

import os
import gc
import json
import re
import logging
from typing import Any, List, Dict, Optional
from pathlib import Path

from app.core.language.service.language_api import get_language_pack
from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.utils.llama_cpp_chat import build_llama, chat_completion, strip_reasoning
from app.features.qa_generation.domain.interfaces import IQAGenerationAdapter
from app.features.qa_generation.domain.entities import AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import (
    QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, CRITIQUE_SYSTEM_PROMPT, CRITIQUE_USER_TEMPLATE
)
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.qa_generation.utils.error_mapper import map_qa_error

logger = logging.getLogger(__name__)


class LlamaCppGenerationAdapter(IQAGenerationAdapter):
    """
    Adapter for executing an instruct GGUF (Qwen3.5-9B) via llama.cpp for the Final QA Generation phase.
    """

    CRITIQUE_MAX_TOKENS = 200
    # Chat-template role markers and rounding in the per-document estimate
    TEMPLATE_MARGIN_TOKENS = 64

    @staticmethod
    def _get_llama_model(absolute_model_path: str) -> Any:
        return build_llama(absolute_model_path, settings.QA_GENERATION_N_CTX, settings.QA_LLM_KV_CACHE_TYPE)

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            # Force absolute path resolution to prevent C++ binding path errors
            absolute_path = str(Path(manifest.model_path).resolve())

            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"QA Generation model file missing: {absolute_path}")

            llm = self._get_llama_model(absolute_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_qa_error(e, context=f"Load QA Generation Model: {manifest.model_path}") from e

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
            raise map_qa_error(e, context="Unload QA Generation Model") from e

    def count_tokens(self, model: LoadedModel, text: str) -> int:
        try:
            llm: Any = model.model_instance
            return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        except Exception as e:
            raise map_qa_error(e, context="Counting tokens") from e

    def get_context_token_budget(self, model: LoadedModel, user_query: str, include_critique: bool,
                                 system_prompt: Optional[str] = None) -> int:
        """
        Tokens left for the context block. The answer prompt holds the block plus the answer being generated;
        the critique prompt holds the same block plus the full draft plus its own verdict, so with a critique
        the block must fit the smaller of the two.
        """
        try:
            llm: Any = model.model_instance
            n_ctx = llm.n_ctx()
            answer_tokens = settings.QA_GENERATION_MAX_ANSWER_TOKENS
            pack = get_language_pack()
            answer_prompt = system_prompt if system_prompt is not None else pack.localize_prompt(QA_SYSTEM_PROMPT)
            fixed = self.count_tokens(model, answer_prompt + QA_USER_TEMPLATE.format(
                context_block="", user_query=user_query))
            budget = n_ctx - fixed - answer_tokens - self.TEMPLATE_MARGIN_TOKENS

            if include_critique:
                critique_fixed = self.count_tokens(model, pack.localize_prompt(CRITIQUE_SYSTEM_PROMPT) + CRITIQUE_USER_TEMPLATE.format(
                    user_query=user_query, evidence_block="", draft_answer=""))
                critique_budget = (n_ctx - critique_fixed - answer_tokens - self.CRITIQUE_MAX_TOKENS
                                   - self.TEMPLATE_MARGIN_TOKENS)
                budget = min(budget, critique_budget)

            return max(budget, 0)
        except Exception as e:
            raise map_qa_error(e, context="Computing the context token budget") from e

    def get_conversation_token_budget(self, model: LoadedModel, system_prompt: str) -> int:
        """Tokens left for every conversation message once the system prompt and the answer are reserved."""
        try:
            llm: Any = model.model_instance
            fixed = self.count_tokens(model, system_prompt)
            budget = (llm.n_ctx() - fixed - settings.QA_GENERATION_MAX_ANSWER_TOKENS
                      - self.TEMPLATE_MARGIN_TOKENS)
            return max(budget, 0)
        except Exception as e:
            raise map_qa_error(e, context="Computing the conversation token budget") from e

    @staticmethod
    def _complete_answer(llm: Any, messages: List[Dict[str, str]]) -> str:
        response = chat_completion(
            llm,
            messages,
            enable_thinking=settings.QA_LLM_ENABLE_THINKING,
            temperature=0.1,
            max_tokens=settings.QA_GENERATION_MAX_ANSWER_TOKENS
        )

        choice = response["choices"][0]
        raw_content = choice["message"].get("content") or ""

        if choice.get("finish_reason", "stop") == "length":
            logger.warning("QA Generation LLM hit max_tokens limit. Response may be truncated.")

        return strip_reasoning(raw_content)

    def generate_answer(
            self,
            model: LoadedModel,
            system_prompt: str,
            context_block: str,
            user_query: str
    ) -> str:
        try:
            user_content = QA_USER_TEMPLATE.format(
                context_block=context_block,
                user_query=user_query
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            return self._complete_answer(model.model_instance, messages)

        except Exception as e:
            raise map_qa_error(e, context="Executing Autoregressive QA Generation") from e

    def generate_conversation_answer(self, model: LoadedModel, messages: List[Dict[str, str]]) -> str:
        try:
            return self._complete_answer(model.model_instance, messages)
        except Exception as e:
            raise map_qa_error(e, context="Executing Conversational QA Generation") from e

    def critique_answer(
            self,
            model: LoadedModel,
            user_query: str,
            draft_answer: str,
            evidence_block: str
    ) -> AnswerCritique:
        try:
            llm: Any = model.model_instance

            user_content = CRITIQUE_USER_TEMPLATE.format(
                user_query=user_query,
                evidence_block=evidence_block,
                draft_answer=draft_answer
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": get_language_pack().localize_prompt(CRITIQUE_SYSTEM_PROMPT)},
                {"role": "user", "content": user_content}
            ]

            response = chat_completion(
                llm,
                messages,
                enable_thinking=settings.QA_LLM_ENABLE_THINKING,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=self.CRITIQUE_MAX_TOKENS
            )

            raw_content = strip_reasoning(response["choices"][0]["message"]["content"] or "")

            json_match = re.search(r'\{.*}', raw_content, re.DOTALL)
            parsed_data = json.loads(json_match.group(0)) if json_match else json.loads(raw_content)

            if not isinstance(parsed_data, dict):
                raise JSONCritiqueParseError(f"Critique verdict is not a JSON object: {raw_content[:200]!r}")

            try:
                verdict = CritiqueVerdict(parsed_data.get("verdict"))
            except ValueError:
                raise JSONCritiqueParseError(
                    f"Critique returned an unrecognized verdict {parsed_data.get('verdict')!r}: {raw_content[:200]!r}"
                )

            return AnswerCritique(verdict=verdict, gap_description=parsed_data.get("gap_description"))

        except Exception as e:
            raise map_qa_error(e, context="Executing Self-Critique of QA Draft") from e
