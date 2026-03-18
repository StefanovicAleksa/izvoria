# Canvas: phonetic_repositories_v2
# Path: app/features/phonetic_correction/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence

from app.features.phonetic_correction.domain.interfaces import IPhoneticRepository
from app.features.phonetic_correction.domain.entities import (
    PhoneticCorrectedTranscript,
    AppliedCorrectionRecord
)
from app.features.phonetic_correction.data.models import PhoneticCorrectedTranscriptModel
from app.features.phonetic_correction.utils.error_mapper import map_phonetic_error


class PhoneticRepository(IPhoneticRepository):
    """
    PostgreSQL implementation of the Phonetic Correction Repository.
    Handles the serialization of complex Pydantic aggregates into JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: PhoneticCorrectedTranscript) -> PhoneticCorrectedTranscript:
        try:
            # Safely serialize the Pydantic models for Postgres JSONB insertion
            serialized_sentences = [s.model_dump(mode='json') for s in transcript.sentences]
            serialized_audit = [a.model_dump(mode='json') for a in transcript.audit_trail]

            db_model = PhoneticCorrectedTranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                sentences=serialized_sentences,
                audit_trail=serialized_audit,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return transcript

        except Exception as e:
            # FIXED: Context string now includes "transcript" to trigger TranscriptSaveError in error mapper
            raise map_phonetic_error(e, context="PhoneticRepository.save_transcript") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[PhoneticCorrectedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(PhoneticCorrectedTranscriptModel).where(
                    PhoneticCorrectedTranscriptModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Safely deserialize the JSONB back into strong Pydantic Domain Entities
                domain_sentences = [ReconstructedSentence(**s) for s in db_model.sentences]
                domain_audit = [AppliedCorrectionRecord(**a) for a in db_model.audit_trail]

                return PhoneticCorrectedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    sentences=domain_sentences,
                    audit_trail=domain_audit,
                    created_at=db_model.created_at
                )

        except Exception as e:
            # FIXED: Context string now includes "transcript"
            raise map_phonetic_error(e, context="PhoneticRepository.get_transcript_by_source_id") from e