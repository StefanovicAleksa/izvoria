# Canvas: visual_synthesis_api_v6
# Path: app/features/visual_context_synthesis/service/synthesis_api.py

import logging
from typing import List, Optional

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_analysis.domain.interfaces import IVisualAnalysisRepository
from app.features.visual_hallucination_correction.domain.interfaces import IVisualHallucinationRepository

from app.features.visual_context_synthesis.domain.entities import (
    VisualTimeline,
    VisualTimelineChunk,
    VisualContextFrame
)
from app.features.visual_context_synthesis.domain.interfaces import (
    IVisualSynthesisRepository,
    IVisualSLMAdapter
)
from app.features.visual_context_synthesis.utils.fuzzy_grouper import build_fuzzy_visual_timeline
from app.features.visual_context_synthesis.utils.error_mapper import map_visual_synthesis_error

logger = logging.getLogger(__name__)


class VisualSynthesisAPI:
    """
    Service Layer for Phase 14: Visual Context Synthesis.
    """

    def __init__(
            self,
            repository: IVisualSynthesisRepository,
            orchestrator: ModelManagerOrchestrator,
            slm_adapter: IVisualSLMAdapter,
            visual_analysis_repo: IVisualAnalysisRepository,
            hallucination_repo: IVisualHallucinationRepository
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._slm_adapter = slm_adapter
        self._visual_analysis_repo = visual_analysis_repo
        self._hallucination_repo = hallucination_repo

        self._manifest = ModelManifest(
            model_type=ModelType.VISUAL_CONTEXT_SYNTHESIS,
            model_path=settings.VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH,
            required_vram_mb=settings.VISUAL_CONTEXT_SYNTHESIS_VRAM_REQUIREMENT_MB
        )

    def execute(
            self,
            transcript: ContextualizedTranscript,
            release_after: bool = False
    ) -> VisualTimeline:
        try:
            source_id = transcript.source_id

            existing = self._repo.get_by_source_id(source_id)
            if existing:
                logger.info(f"Visual Timeline already exists for Source {source_id}. Skipping.")
                return existing

            hallucination_ledger = self._hallucination_repo.get_by_source_id(source_id)

            accepted_chunk_ids = set()
            if hallucination_ledger:
                for audit in hallucination_ledger.audits:
                    if audit.is_visual_accepted:
                        accepted_chunk_ids.add(audit.chunk_id)

            visual_chunks: List[VisualTimelineChunk] = []
            loaded_model: Optional[LoadedModel] = None

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._slm_adapter)

                for text_chunk in transcript.chunks:
                    if text_chunk.chunk_id not in accepted_chunk_ids:
                        visual_chunks.append(VisualTimelineChunk(chunk_id=text_chunk.chunk_id))
                        continue

                    visual_analysis = self._visual_analysis_repo.get_by_chunk_id(text_chunk.chunk_id)
                    if not visual_analysis or not visual_analysis.analyses:
                        visual_chunks.append(VisualTimelineChunk(chunk_id=text_chunk.chunk_id))
                        continue

                    accepted_frame_ids = {a.frame_id for a in visual_analysis.analyses}

                    # FIX: Pass the chunk's start_sec so frames without explicit timestamps inherit it
                    chunk_start_time = getattr(text_chunk, "start_sec", 0.0)

                    timeline_events = build_fuzzy_visual_timeline(
                        visual_analyses=visual_analysis.analyses,
                        accepted_frame_ids=accepted_frame_ids,
                        base_timestamp_sec=chunk_start_time
                    )

                    if not timeline_events:
                        visual_chunks.append(VisualTimelineChunk(chunk_id=text_chunk.chunk_id))
                        continue

                    visual_narrative = self._slm_adapter.synthesize_context(
                        model=loaded_model,
                        visual_events=timeline_events
                    )

                    evidence_frames = []
                    for event in timeline_events:
                        for fid in event["frame_ids"]:
                            evidence_frames.append(VisualContextFrame(
                                frame_id=fid,
                                timestamp_sec=event["start_sec"]
                            ))

                    visual_chunks.append(VisualTimelineChunk(
                        chunk_id=text_chunk.chunk_id,
                        visual_context_string=visual_narrative,
                        visual_evidence=evidence_frames
                    ))

            finally:
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._slm_adapter, flush=True)

            final_timeline = VisualTimeline(
                source_id=source_id,
                visual_chunks=visual_chunks
            )
            return self._repo.save(final_timeline)

        except Exception as e:
            raise map_visual_synthesis_error(e, context=f"VisualSynthesisAPI.execute for {transcript.source_id}")