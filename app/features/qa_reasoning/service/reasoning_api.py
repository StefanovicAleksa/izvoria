# Path: app/features/qa_reasoning/service/reasoning_api.py

import logging
from typing import List, Optional

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_reasoning.domain.interfaces import IReasonerAdapter
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.qa_reasoning.utils.evidence_formatter import build_evidence_block, build_decision_history_block
from app.features.qa_reasoning.utils.error_mapper import map_reasoner_error

logger = logging.getLogger(__name__)


class ReasoningAPI:
    """
    Service for running one Deep Research Reasoner turn. Each call acquires the model once,
    runs a single reasoning turn, and releases — the multi-iteration loop lives in
    LiveDeepResearchPipeline, not here, since every pipeline iteration is a fresh call.
    """

    def __init__(
            self,
            adapter: IReasonerAdapter,
            orchestrator: ModelManagerOrchestrator
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator

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
            manifest = ModelManifest(
                model_type=ModelType.QA_REASONER,
                model_path=settings.QA_REASONER_MODEL_PATH,
                required_vram_mb=settings.QA_REASONER_VRAM_REQUIREMENT_MB
            )

            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter, wait_for_clear=True)

            try:
                decision = self._adapter.reason(
                    model=loaded_model,
                    original_question=original_question,
                    evidence_block=build_evidence_block(evidence_pool),
                    decision_history_block=build_decision_history_block(decision_history),
                    prior_session_context=prior_session_context or "",
                    extra_context=extra_context or "",
                )
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            logger.info(f"Reasoner turn complete. Decision: {decision.decision.value}")
            return decision

        except Exception as e:
            raise map_reasoner_error(e, context=f"Reasoning turn for question: '{original_question}'") from e
