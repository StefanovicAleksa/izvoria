# Path: app/features/qa_reasoning/service/reasoning_api.py

import logging
from typing import List, Optional

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.qa_generation.utils.context_packer import has_document_chunks
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_reasoning.domain.interfaces import IReasonerAdapter
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.utils.evidence_formatter import build_evidence_block, build_decision_history_block
from app.features.qa_reasoning.exceptions.pipeline import JSONReasonerParseError
from app.features.qa_reasoning.utils.error_mapper import map_reasoner_error

logger = logging.getLogger(__name__)


class ReasoningAPI:
    """
    Service for running one Reasoner turn. Each call acquires the model once,
    runs a single reasoning turn, and releases — the multi-iteration loop lives in
    LiveReasoningPipeline, not here, since every pipeline iteration is a fresh call.
    """

    def __init__(
            self,
            adapter: IReasonerAdapter,
            orchestrator: ModelManagerOrchestrator
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator
        self._manifest = ModelManifest(
            model_type=ModelType.QA_REASONER,
            model_path=settings.QA_REASONER_MODEL_PATH,
            required_vram_mb=settings.QA_REASONER_VRAM_REQUIREMENT_MB
        )

    def _acquire(self) -> LoadedModel:
        return self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

    def rewrite_query(self, question: str, prior_session_context: Optional[str], release_after: bool = True) -> str:
        """
        Turns a follow-up ("what did they say after that?") into a standalone search query using the prior
        turn. The rewrite is only for searching; the Reasoner and the answer model keep the user's own words.
        Without prior context there is nothing to resolve, and an unusable rewrite falls back to the question.
        """
        if not prior_session_context:
            return question

        try:
            loaded_model = self._acquire()
            try:
                rewritten = self._adapter.rewrite_query(loaded_model, question, prior_session_context)
            except JSONReasonerParseError as parse_err:
                logger.warning(f"Query rewrite unparseable; searching with the original question. {parse_err}")
                return question
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            logger.info(f"Rewrote follow-up '{question}' as standalone query '{rewritten}'")
            return rewritten

        except Exception as e:
            raise map_reasoner_error(e, context=f"Rewriting follow-up question: '{question}'") from e

    def reason(
            self,
            original_question: str,
            evidence_pool: List[RetrievedChunk],
            decision_history: List[ReasonerDecision],
            prior_session_context: Optional[str] = None,
            extra_context: Optional[str] = None,
            release_after: bool = True,
    ) -> ReasonerDecision:
        logger.info(f"Executing Reasoner turn for question: '{original_question}'")

        try:
            loaded_model = self._acquire()

            try:
                history_block = build_decision_history_block(decision_history)
                has_documents = has_document_chunks(evidence_pool)
                budget = self._adapter.get_evidence_token_budget(
                    loaded_model, original_question, history_block, prior_session_context or "", extra_context or "",
                    has_documents=has_documents
                )
                evidence = build_evidence_block(
                    evidence_pool, budget, lambda text: self._adapter.count_tokens(loaded_model, text)
                )
                decision = self._adapter.reason(
                    model=loaded_model,
                    original_question=original_question,
                    evidence_block=evidence.formatted_text,
                    decision_history_block=history_block,
                    prior_session_context=prior_session_context or "",
                    extra_context=extra_context or "",
                    evidence_tags=evidence.tag_mapping,
                    has_documents=has_documents,
                )
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            logger.info(f"Reasoner turn complete. Decision: {decision.decision.value}")
            return decision

        except Exception as e:
            raise map_reasoner_error(e, context=f"Reasoning turn for question: '{original_question}'") from e
