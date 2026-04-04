# Canvas: domain_phonetic_repositories_v5
# Path: app/features/domain_phonetic_correction/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticRepository
from app.features.domain_phonetic_correction.domain.entities import (
    DomainPhoneticCorrectedTranscript,
    AppliedDomainCorrectionRecord
)
from app.features.domain_phonetic_correction.data.models import DomainPhoneticCorrectedTranscriptModel
from app.features.domain_phonetic_correction.utils.error_mapper import map_phonetic_error

class DomainPhoneticRepository(IDomainPhoneticRepository):
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: DomainPhoneticCorrectedTranscript) -> DomainPhoneticCorrectedTranscript:
        try:
            serialized_sentences = [s.model_dump(mode='json') for s in transcript.sentences]
            serialized_audit = [a.model_dump(mode='json') for a in transcript.audit_trail]

            db_model = DomainPhoneticCorrectedTranscriptModel(
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
            # FIXED: Added "transcript" to the context string to trigger TranscriptSaveError mapping
            raise map_phonetic_error(e, context="DomainPhoneticRepository.save_transcript") from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainPhoneticCorrectedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(DomainPhoneticCorrectedTranscriptModel).where(
                    DomainPhoneticCorrectedTranscriptModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                domain_sentences = [ReconstructedSentence(**s) for s in db_model.sentences]
                domain_audit = [AppliedDomainCorrectionRecord(**a) for a in db_model.audit_trail]

                return DomainPhoneticCorrectedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    sentences=domain_sentences,
                    audit_trail=domain_audit,
                    created_at=db_model.created_at
                )

        except Exception as e:
            # FIXED: Added "transcript" to the context string
            raise map_phonetic_error(e, context="DomainPhoneticRepository.get_transcript_by_source_id") from e