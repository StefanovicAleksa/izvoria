# Canvas: context_unification_phase_v1
# Path: app/orchestration/service/phases/context_unification_phase.py

import logging
from typing import List

from app.core.jobs.service.manager import JobManager
from app.core.jobs.domain.entities import JobRequest
from app.core.jobs.domain.enums import JobType

from app.features.context_unification.service.unification_api import UnificationAPI
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.visual_context_synthesis.domain.interfaces import IVisualSynthesisRepository

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ContextualRetrievalJobResult, \
    ContextUnificationJobResult

logger = logging.getLogger(__name__)


class ContextUnificationPhase:
    """
    Phase 15: The final deterministic data preparation phase before RAG vector indexing.
    This phase iterates over ALL Contextualized Transcripts (from Phase 9).
    If a corresponding Visual Timeline (Phase 14) exists, it fuses them.
    If no visual timeline exists (e.g., pure audio), it safely wraps the audio-only payload.
    """

    def __init__(
            self,
            job_manager: JobManager,
            unification_api: UnificationAPI,
            contextual_repo: IContextualRepository,
            visual_synthesis_repo: IVisualSynthesisRepository
    ):
        self._job_manager = job_manager
        self._unification_api = unification_api
        self._contextual_repo = contextual_repo
        self._visual_synthesis_repo = visual_synthesis_repo

    def execute(self, contextual_results: List[ContextualRetrievalJobResult]) -> PhaseBatchResult:
        logger.info(f"Executing Context Unification Phase for {len(contextual_results)} transcripts...")
        results: List[ContextUnificationJobResult] = []
        errors: List[str] = []
        success_count = 0
        failed_count = 0

        for item in contextual_results:
            if not item.success:
                continue

            source_id = item.source_id

            # 1. Fetch Phase 9 Audio Context
            audio_transcript = self._contextual_repo.get_by_source_id(source_id)
            if not audio_transcript:
                err_msg = f"Cannot unify context: Audio Transcript for Source {source_id} missing."
                logger.error(err_msg)
                results.append(ContextUnificationJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(err_msg)
                failed_count += 1
                continue

            # 2. Fetch Phase 14 Visual Context (Can safely be None)
            visual_timeline = self._visual_synthesis_repo.get_by_source_id(source_id)

            request = JobRequest(
                job_type=JobType.CONTEXT_UNIFICATION,
                source_id=source_id,
                input_params={"video_source_id": str(source_id)}
            )

            try:
                logger.info(f"Processing Context Unification for Source: {source_id}")

                unified_transcript = self._job_manager.run_job(
                    request,
                    self._unification_api.execute,
                    audio_transcript=audio_transcript,
                    visual_timeline=visual_timeline
                )

                results.append(ContextUnificationJobResult(
                    source_id=source_id,
                    success=True,
                    unified_transcript_id=unified_transcript.id
                ))
                success_count += 1

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Job failed for Source {source_id}: {err_msg}")
                results.append(ContextUnificationJobResult(source_id=source_id, success=False, error_message=err_msg))
                errors.append(f"Failed Context Unification for Source {source_id}: {err_msg}")
                failed_count += 1

        status = PhaseStatus.COMPLETED
        if success_count == 0 and failed_count > 0:
            status = PhaseStatus.FAILED
        elif success_count > 0 and failed_count > 0:
            status = PhaseStatus.PARTIAL_SUCCESS

        return PhaseBatchResult(
            phase=PipelinePhase.CONTEXT_UNIFICATION,
            status=status,
            total_processed=len(contextual_results),
            successful_count=success_count,
            failed_count=failed_count,
            items=results,
            errors=errors
        )