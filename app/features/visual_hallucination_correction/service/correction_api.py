import logging
import uuid
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.domain.interfaces import IVisualHallucinationAdapter
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)
from app.features.visual_hallucination_correction.utils.error_mapper import map_hallucination_error

logger = logging.getLogger(__name__)

class VisualHallucinationCorrectionAPI:
    def __init__(
        self,
        repository: VisualHallucinationRepository,
        adapter: IVisualHallucinationAdapter,
        contextual_repo: ContextualRepository,
        visual_analysis_repo: VisualAnalysisRepository,
        orchestrator: ModelManagerOrchestrator
    ):
        self._repo = repository
        self._adapter = adapter
        self._contextual_repo = contextual_repo
        self._visual_repo = visual_analysis_repo
        self._orchestrator = orchestrator

    def execute(self, source_id: uuid.UUID, release_after: bool = True) -> VisualHallucinationEvaluatedTranscript:
        """Audits transcript chunks against visual descriptions while enforcing SLM context limits."""
        try:
            transcript = self._contextual_repo.get_by_source_id(source_id)
            if not transcript or not transcript.chunks:
                return VisualHallucinationEvaluatedTranscript(source_id=source_id, audits=[])

            # Idempotency Check
            existing_eval = self._repo.get_by_source_id(source_id)
            if existing_eval:
                return existing_eval

            manifest = ModelManifest(
                model_type=ModelType.VISUAL_HALLUCINATION_CORRECTION,
                model_path=settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH,
                required_vram_mb=settings.VISUAL_HALLUCINATION_CORRECTION_VRAM_REQUIREMENT_MB
            )

            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)
            audits: List[VisualHallucinationAudit] = []

            try:
                for chunk in transcript.chunks:
                    if not getattr(chunk, 'needs_visual', False):
                        continue

                    visual_record = self._visual_repo.get_by_chunk_id(chunk.chunk_id)
                    if not visual_record or not visual_record.analyses:
                        continue

                    # Compile redundant frame descriptions using numerical index
                    combined_visual_evidence = "\n".join([
                        f"- Frame {idx + 1}: {a.description}"
                        for idx, a in enumerate(visual_record.analyses)
                    ])

                    # --- FIX: Context Window Truncation Safeguard ---
                    MAX_CHARS = 10000
                    if len(combined_visual_evidence) > MAX_CHARS:
                        logger.warning(f"Truncating visual context for chunk {chunk.chunk_id} to 10k chars.")
                        combined_visual_evidence = combined_visual_evidence[:MAX_CHARS] + "\n... [TRUNCATED TO PREVENT OOM]"
                    # ------------------------------------------------

                    audit = self._adapter.evaluate_visual_match(
                        model=loaded_model,
                        chunk_id=chunk.chunk_id,
                        text_chunk=chunk.original_text,
                        visual_description=combined_visual_evidence
                    )
                    audits.append(audit)

                evaluated_transcript = VisualHallucinationEvaluatedTranscript(
                    source_id=source_id,
                    audits=audits
                )

                if audits:
                    self._repo.save(evaluated_transcript)

                return evaluated_transcript

            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

        except Exception as e:
            raise map_hallucination_error(e, context=f"CorrectionAPI.execute for {source_id}") from e