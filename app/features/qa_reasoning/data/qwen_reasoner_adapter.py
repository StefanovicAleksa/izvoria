# Path: app/features/qa_reasoning/data/qwen_reasoner_adapter.py

import os
import gc
import json
import re
import logging
from typing import Any, Dict, List
from pathlib import Path

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.qa_reasoning.domain.interfaces import IReasonerAdapter
from app.features.qa_reasoning.domain.entities import ReasonerDecision, EvidenceLink
from app.features.qa_reasoning.domain.enums import ReasonerDecisionType
from app.features.qa_reasoning.data.prompts import REASONER_SYSTEM_PROMPT, REASONER_USER_TEMPLATE
from app.features.qa_reasoning.exceptions.pipeline import JSONReasonerParseError
from app.features.qa_reasoning.utils.error_mapper import map_reasoner_error

logger = logging.getLogger(__name__)


class QwenReasonerAdapter(IReasonerAdapter):
    """
    Adapter for executing Qwen2.5-7B-Instruct via llama.cpp as the Deep Research Reasoner.
    Fails closed on an unparseable/invalid `decision` field: unlike the retrieval scrubber's
    fail-open default, there is no safe default among three semantically different
    control-flow branches (see JSONReasonerParseError).
    """

    @staticmethod
    def _get_llama_model(model_path: str) -> Any:
        from llama_cpp import Llama
        return Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=8192,
            flash_attn=True,
            verbose=False
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            absolute_path = str(Path(manifest.model_path).resolve())
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(f"Reasoner model file missing: {absolute_path}")
            llm = self._get_llama_model(absolute_path)
            return LoadedModel(manifest=manifest, model_instance=llm)
        except Exception as e:
            raise map_reasoner_error(e, context=f"Load Reasoner Adapter: {manifest.model_path}") from e

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
            raise map_reasoner_error(e, context="Unload Reasoner Adapter") from e

    def reason(
        self,
        model: LoadedModel,
        original_question: str,
        evidence_block: str,
        decision_history_block: str,
        prior_session_context: str,
        extra_context: str,
    ) -> ReasonerDecision:
        try:
            llm: Any = model.model_instance

            user_content = REASONER_USER_TEMPLATE.format(
                original_question=original_question,
                evidence_block=evidence_block,
                decision_history_block=decision_history_block,
                prior_session_context=prior_session_context or "(None — this is a new conversation.)",
                extra_context=extra_context or "(None.)",
            )

            messages = [
                {"role": "system", "content": REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            response = llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                # Deterministic: the decision must follow from the evidence actually gathered,
                # not from decoding randomness flip-flopping between near-identical inputs.
                temperature=0.0,
                max_tokens=1000
            )

            raw_content = response["choices"][0]["message"]["content"]

            json_match = re.search(r'\{.*}', raw_content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
            else:
                parsed_data = json.loads(raw_content)

            if not isinstance(parsed_data, dict):
                raise JSONReasonerParseError(f"Reasoner verdict is not a JSON object: {raw_content[:200]!r}")

            return self._to_decision(parsed_data, raw_content)

        except Exception as e:
            raise map_reasoner_error(e, context="Executing Reasoner turn") from e

    @staticmethod
    def _to_decision(parsed_data: Dict[str, Any], raw_content: str) -> ReasonerDecision:
        decision_raw = parsed_data.get("decision")
        try:
            decision = ReasonerDecisionType(decision_raw)
        except ValueError:
            # Fail CLOSED: no branch among the three is a safe default to silently pick.
            raise JSONReasonerParseError(
                f"Reasoner returned an unrecognized decision {decision_raw!r}: {raw_content[:200]!r}"
            )

        evidence_links: List[EvidenceLink] = []
        for raw_link in parsed_data.get("evidence_links", []) or []:
            try:
                evidence_links.append(EvidenceLink(chunk_id=raw_link["chunk_id"], claim=raw_link["claim"]))
            except Exception as link_err:
                # Fail open for this sub-field only: one malformed evidence link doesn't invalidate
                # the whole turn's decision.
                logger.warning(f"Skipping unparseable evidence_link entry: {raw_link!r} ({link_err})")

        return ReasonerDecision(
            decision=decision,
            chain_of_thought=parsed_data.get("chain_of_thought", ""),
            interim_analysis=parsed_data.get("interim_analysis", ""),
            evidence_links=evidence_links,
            refined_query=parsed_data.get("refined_query"),
            clarification_question=parsed_data.get("clarification_question"),
        )
