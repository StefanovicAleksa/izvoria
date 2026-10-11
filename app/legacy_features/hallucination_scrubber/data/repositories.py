# Canvas: scrubber_repositories_v1
# Path: app/features/hallucination_scrubber/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.transcription.domain.entities import TranscriptSegment
from app.legacy_features.hallucination_scrubber.domain.interfaces import IHallucinationRepository
from app.legacy_features.hallucination_scrubber.domain.entities import (
    HallucinationScrubbedTranscript,
    AppliedScrubbingRecord
)
from app.legacy_features.hallucination_scrubber.data.models import HallucinationScrubbedTranscriptModel
from app.legacy_features.hallucination_scrubber.utils.error_mapper import map_scrubber_error


class HallucinationRepository(IHallucinationRepository):
    """
    PostgreSQL implementation for persisting Hallucination Scrubbed Transcripts.
    Handles serialization and deserialization between Domain Entities and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: HallucinationScrubbedTranscript) -> HallucinationScrubbedTranscript:
        try:
            # Serialize the complex Pydantic models to JSON-compliant dictionaries
            serialized_segments = [seg.model_dump(mode='json') for seg in transcript.segments]
            serialized_audit = [audit.model_dump(mode='json') for audit in transcript.audit_trail]

            db_model = HallucinationScrubbedTranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                segments=serialized_segments,
                audit_trail=serialized_audit,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                # Izvoria standard: we merge to handle re-processing/upserts cleanly
                session.merge(db_model)

            return transcript

        except Exception as e:
            raise map_scrubber_error(e, context="HallucinationRepository.save_transcript") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[HallucinationScrubbedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(HallucinationScrubbedTranscriptModel).where(
                    HallucinationScrubbedTranscriptModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Deserialize JSONB dictionaries back into strict Pydantic Domain Entities
                domain_segments = [TranscriptSegment(**s) for s in db_model.segments]
                domain_audit = [AppliedScrubbingRecord(**a) for a in db_model.audit_trail]

                return HallucinationScrubbedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    segments=domain_segments,
                    audit_trail=domain_audit,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_scrubber_error(e, context="HallucinationRepository.get_transcript_by_source_id") from e