# Canvas: correction_api_v1
# Path: app/features/visual_hallucination_correction/service/correction_api.py

import logging
import uuid
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.contextual_retrieval.domain.interfaces import IContextualRepository
from app.features.visual_analysis.domain.interfaces import IVisualAnalysisRepository

from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)
from app.features.visual_hallucination_correction.domain.interfaces import (
    IVisualHallucinationRepository,
    IVisualHallucinationAdapter
)
from app.features.visual_hallucination_correction.utils.error_mapper import map_hallucination_error

logger = logging.getLogger(__name__)


class VisualHallucinationCorrectionAPI:
    """
    Orchestrates the Phase 13 Gatekeeper logic.
    Evaluates Phase 12 VLM descriptions against Phase 9 Text Chunks to filter out
    hallucinations and irrelevant imagery before final RAG ingestion.
    """

    def __init__(
            self,
            repository: IVisualHallucinationRepository,
            adapter: IVisualHallucinationAdapter,
            contextual_repo: IContextualRepository,
            visual_analysis_repo: IVisualAnalysisRepository,
            orchestrator: ModelManagerOrchestrator
    ):
        self._repo = repository
        self._adapter = adapter
        self._contextual_repo = contextual_repo
        self._visual_analysis_repo = visual_analysis_repo
        self._orchestrator = orchestrator

        # Strict Domain Manifest Isolation
        self._manifest = ModelManifest(
            model_type=ModelType.VISUAL_HALLUCINATION_CORRECTION,
            model_path=settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH,
            required_vram_mb=settings.VISUAL_HALLUCINATION_CORRECTION_VRAM_REQUIREMENT_MB
        )

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    @property
    def adapter(self) -> IVisualHallucinationAdapter:
        return self._adapter

    def execute(
            self,
            source_id: uuid.UUID,
            release_after: bool = True
    ) -> VisualHallucinationEvaluatedTranscript:
        try:
            # 1. Idempotency Check: Don't re-run inference if we survived a previous crash
            existing_record = self._repo.get_by_source_id(source_id)
            if existing_record:
                logger.info(f"Visual Hallucination Audit already exists for Source {source_id}. Skipping.")
                return existing_record

            # 2. Fetch the upstream prerequisites
            contextual_transcript = self._contextual_repo.get_by_source_id(source_id)
            if not contextual_transcript or not contextual_transcript.chunks:
                return self._save_empty_result(source_id)

            audits: List[VisualHallucinationAudit] = []
            loaded_model = None

            try:
                # 3. Request the Model Manager to secure the GPU lock and load Qwen 2.5 3B
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)

                # 4. Iterate and evaluate only the chunks that actually generated visual data
                for chunk in contextual_transcript.chunks:
                    if not chunk.needs_visual:
                        continue

                    visual_analysis = self._visual_analysis_repo.get_by_chunk_id(chunk.chunk_id)

                    # If there are no physical frames or the VLM failed to analyze them, skip
                    if not visual_analysis or not visual_analysis.analyses:
                        continue

                    logger.info(f"  -> Auditing Visual Context for Chunk {chunk.chunk_id}...")

                    # Combine all frame descriptions for this chunk into a single evidence block
                    combined_visual_evidence = "\n".join([a.description for a in visual_analysis.analyses])

                    # 5. Execute the SLM Gatekeeper Inference
                    audit_record = self._adapter.evaluate_visual_match(
                        model=loaded_model,
                        chunk_id=chunk.chunk_id,
                        text_chunk=chunk.original_text,
                        visual_description=combined_visual_evidence
                    )

                    audits.append(audit_record)

                    if audit_record.is_visual_accepted:
                        logger.debug(f"Visual evidence ACCEPTED for chunk {chunk.chunk_id}.")
                    else:
                        logger.warning(
                            f"Visual evidence REJECTED for chunk {chunk.chunk_id}: {audit_record.discard_reason}")

            finally:
                # 6. Release VRAM explicitly for the next pipeline phase
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            # 7. Aggregate and Persist the Ledger
            final_evaluated_transcript = VisualHallucinationEvaluatedTranscript(
                source_id=source_id,
                audits=audits
            )

            return self._repo.save(final_evaluated_transcript)

        except Exception as e:
            raise map_hallucination_error(
                e,
                context=f"VisualHallucinationCorrectionAPI.execute for Source {source_id}"
            ) from e

    def _save_empty_result(self, source_id: uuid.UUID) -> VisualHallucinationEvaluatedTranscript:
        """Saves an empty ledger record if there were no valid visual chunks to audit."""
        empty_transcript = VisualHallucinationEvaluatedTranscript(
            source_id=source_id,
            audits=[]
        )
        return self._repo.save(empty_transcript)