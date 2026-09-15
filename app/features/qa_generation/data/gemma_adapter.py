# Canvas: gemma_adapter_v11
# Path: app/features/qa_generation/data/gemma_adapter.py

import os
import gc
import json
import re
import logging
from typing import Any, List, Dict
from pathlib import Path

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.qa_generation.domain.interfaces import IQAGenerationAdapter
from app.features.qa_generation.domain.entities import AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.qa_generation.utils.error_mapper import map_qa_error

logger = logging.getLogger(__name__)


class GemmaGenerationAdapter(IQAGenerationAdapter):
    """
    Adapter for executing Gemma Instruct GGUF via llama.cpp for the Final QA Generation phase.
    """

    @staticmethod
    def _get_llama_model(absolute_model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=absolute_model_path,
            n_gpu_layers=-1,
            n_ctx=8192,  # 8K context for RAG
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            # Force absolute path resolution to prevent C++ binding path errors
            absolute_path = str(Path(manifest.model_path).resolve())

            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"QA Generation model file missing: {absolute_path}")

            llm = self._get_llama_model(absolute_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_qa_error(e, context=f"Load Gemma Model: {manifest.model_path}") from e

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
            raise map_qa_error(e, context="Unload Gemma Model") from e

    def generate_answer(
            self,
            model: LoadedModel,
            system_prompt: str,
            context_block: str,
            user_query: str
    ) -> str:
        try:
            llm: Any = model.model_instance

            from app.features.qa_generation.data.prompts import QA_USER_TEMPLATE
            user_content = QA_USER_TEMPLATE.format(
                context_block=context_block,
                user_query=user_query
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )

            choice = response["choices"][0]
            raw_content = choice["message"].get("content", "")
            finish_reason = choice.get("finish_reason", "stop")

            if finish_reason == "length":
                logger.warning("QA Generation LLM hit max_tokens limit. Response may be truncated.")

            # Strip C++ null terminators safely
            raw_content = raw_content.replace('\x00', '')

            # Gemma 4 outputs empty thought tags when reasoning is disabled. Strip them to keep the UI clean.
            cleaned_content = re.sub(r'<\|channel>thought\n<channel\|>', '', raw_content).strip()

            return cleaned_content

        except Exception as e:
            raise map_qa_error(e, context="Executing Autoregressive QA Generation") from e

    def critique_answer(
            self,
            model: LoadedModel,
            user_query: str,
            draft_answer: str,
            evidence_block: str
    ) -> AnswerCritique:
        try:
            llm: Any = model.model_instance

            from app.features.qa_generation.data.prompts import CRITIQUE_SYSTEM_PROMPT, CRITIQUE_USER_TEMPLATE
            user_content = CRITIQUE_USER_TEMPLATE.format(
                user_query=user_query,
                evidence_block=evidence_block,
                draft_answer=draft_answer
            )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            response = llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200
            )

            raw_content = response["choices"][0]["message"]["content"]

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