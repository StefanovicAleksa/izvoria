# Canvas: context_unification_api_v1
# Path: app/features/context_unification/service/unification_api.py

import logging
from typing import Optional

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_context_synthesis.domain.entities import VisualTimeline

from app.features.context_unification.domain.interfaces import IContextUnificationRepository
from app.features.context_unification.domain.entities import UnifiedTranscript
from app.features.context_unification.exceptions.process import TranscriptMismatchError
from app.features.context_unification.utils.assembler import assemble_unified_chunks
from app.features.context_unification.utils.error_mapper import map_unification_error

logger = logging.getLogger(__name__)


class UnificationAPI:
    """
    Service Layer for Phase 15: Context Unification.
    Fuses the Audio Context and Visual Context deterministically with zero VRAM cost.
    """

    def __init__(self, repository: IContextUnificationRepository):
        self._repo = repository

    def execute(
            self,
            audio_transcript: ContextualizedTranscript,
            visual_timeline: Optional[VisualTimeline] = None
    ) -> UnifiedTranscript:
        try:
            source_id = audio_transcript.source_id

            # 1. Idempotency Check: Don't reconstruct if it already exists
            existing_record = self._repo.get_by_source_id(source_id)
            if existing_record:
                logger.info(f"Unified Transcript already exists for Source {source_id}. Skipping assembly.")
                return existing_record

            # 2. Data Validation Guard
            if visual_timeline and visual_timeline.source_id != source_id:
                raise TranscriptMismatchError(
                    f"CRITICAL: Attempted to fuse Audio (Source: {source_id}) "
                    f"with Visuals from a different file (Source: {visual_timeline.source_id})."
                )

            # 3. Deterministic Assembly (Zero-VRAM Python logic)
            logger.info(f"Assembling unified chunks for Source {source_id}...")
            unified_chunks = assemble_unified_chunks(
                audio_transcript=audio_transcript,
                visual_timeline=visual_timeline
            )

            # 4. Construct Final Domain Aggregate
            final_transcript = UnifiedTranscript(
                source_id=source_id,
                chunks=unified_chunks
            )

            # 5. Persist to PostgreSQL
            logger.info(f"Saving fully assembled Unified Transcript for Source {source_id}...")
            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_unification_error(e, context=f"UnificationAPI.execute for Source {audio_transcript.source_id}")