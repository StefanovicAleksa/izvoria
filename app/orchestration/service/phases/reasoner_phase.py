# Path: app/orchestration/service/phases/reasoner_phase.py

import logging
import concurrent.futures
import uuid
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.core.model_manager.service.async_model_service import AsyncModelService
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.exceptions.base import ModelManagerError

from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI
from app.features.qa_reasoning.domain.entities import ReasonerDecision
from app.features.rag_retrieval.domain.entities import RetrievedChunk

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ReasonerJobResult

logger = logging.getLogger(__name__)


class ReasonerPhase:
    """
    Wraps one Deep Research Reasoner turn with job tracking and model lifecycle. The
    multi-iteration loop lives in LiveDeepResearchPipeline, not here — this Phase, like every
    other Phase, wraps exactly one logical step.
    """

    def __init__(
            self,
            job_manager: JobManager,
            reasoning_api: ReasoningAPI,
            async_model_service: AsyncModelService,
            loader: IModelLoader
    ):
        self._job_manager = job_manager
        self._reasoning_api = reasoning_api
        self._async_service = async_model_service
        self._loader = loader
        self._preload_future: Optional[concurrent.futures.Future] = None

    def preload_model(self) -> None:
        logger.info("Triggering background preload for Deep Research Reasoner LLM...")
        manifest = ModelManifest(
            model_type=ModelType.QA_REASONER,
            model_path=settings.QA_REASONER_MODEL_PATH,
            required_vram_mb=settings.QA_REASONER_VRAM_REQUIREMENT_MB
        )
        self._preload_future = self._async_service.trigger_preload(
            manifest=manifest,
            loader=self._loader
        )

    def discard_preload(self) -> None:
        """Releases a preload that execute() never resolved (e.g. the pipeline halted early)."""
        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            self._async_service.discard_preload(future, self._loader)

    def rewrite_query(self, search_id: uuid.UUID, question: str, prior_session_context: Optional[str]) -> str:
        """
        Standalone search query for a follow-up question. A failed rewrite only costs search quality, so it
        falls back to the question itself rather than halting the pipeline.
        """
        if not prior_session_context:
            return question

        request = JobRequest(
            job_type=JobType.DEEP_RESEARCH_REASONING,
            source_id=None,
            input_params={"search_id": str(search_id), "step": "query_rewrite"}
        )
        try:
            return self._job_manager.run_job(
                request,
                self._reasoning_api.rewrite_query,
                question=question,
                prior_session_context=prior_session_context,
                release_after=True
            )
        except Exception as e:
            logger.error(f"Query rewrite failed for Search ID {search_id}; searching with the question as asked: {e}")
            return question

    def execute(
            self,
            reasoning_requests: List[Tuple[
                uuid.UUID, str, List[RetrievedChunk], List[ReasonerDecision], Optional[str], Optional[str]
            ]]
    ) -> PhaseBatchResult:
        """
        Takes a list of tuples containing:
        (search_id, original_question, evidence_pool, decision_history, prior_session_context, extra_context).
        In practice LiveDeepResearchPipeline always calls this with exactly one item per turn.
        """
        logger.info(f"Executing Reasoner Phase for {len(reasoning_requests)} turns...")

        results: List[Tuple[ReasonerJobResult, Optional[ReasonerDecision]]] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0
        loaded_model: Optional[LoadedModel] = None

        if self._preload_future:
            future, self._preload_future = self._preload_future, None
            try:
                loaded_model = self._async_service.resolve_preload(future)
                logger.info("Successfully resolved preloaded Reasoner LLM.")
            except ModelManagerError as e:
                logger.error(f"Failed to load Reasoner LLM: {e}")
                return PhaseBatchResult(
                    phase=PipelinePhase.DEEP_RESEARCH_REASONING,
                    status=PhaseStatus.FAILED,
                    errors=[f"Critical failure preloading Reasoner LLM: {str(e)}"]
                )

        try:
            for search_id, question, evidence_pool, history, prior_context, extra_context in reasoning_requests:
                request = JobRequest(
                    job_type=JobType.DEEP_RESEARCH_REASONING,
                    source_id=None,
                    input_params={"search_id": str(search_id)}
                )

                try:
                    logger.info(f"Processing Reasoner turn for Search ID: {search_id}")

                    decision: ReasonerDecision = self._job_manager.run_job(
                        request,
                        self._reasoning_api.reason,
                        original_question=question,
                        evidence_pool=evidence_pool,
                        decision_history=history,
                        prior_session_context=prior_context,
                        extra_context=extra_context,
                        release_after=loaded_model is None
                    )

                    job_result = ReasonerJobResult(search_id=search_id, success=True, decision=decision.decision.value)
                    results.append((job_result, decision))
                    success_count += 1

                except Exception as e:
                    err_msg = str(e)
                    logger.error(f"Job failed for Search ID {search_id}: {err_msg}")

                    job_result = ReasonerJobResult(search_id=search_id, success=False, error_message=err_msg)
                    results.append((job_result, None))
                    errors.append(f"Failed Reasoner turn for Search ID {search_id}: {err_msg}")
                    failed_count += 1

        finally:
            if loaded_model:
                logger.info("Unloading Reasoner model to free VRAM...")
                self._async_service.trigger_unload(
                    loaded_model=loaded_model,
                    loader=self._loader,
                    flush=True
                )

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.DEEP_RESEARCH_REASONING,
            status=status,
            total_processed=len(reasoning_requests),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )
