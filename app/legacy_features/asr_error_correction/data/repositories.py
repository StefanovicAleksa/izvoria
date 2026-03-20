import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.legacy_features.asr_error_correction.domain.interfaces import ICorrectionRepository
from app.legacy_features.asr_error_correction.domain.entities import (
    DocumentGlossary,
    GlossaryItem,
    DomainCorrectedTranscript,
    CorrectionAuditRecord
)
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.legacy_features.asr_error_correction.data.models import DocumentGlossaryModel, DomainCorrectedTranscriptModel
from app.legacy_features.asr_error_correction.utils.error_mapper import map_asr_correction_error


class CorrectionRepository(ICorrectionRepository):
    """
    PostgreSQL implementation of the ASR Correction Repository.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save_glossary(self, glossary: DocumentGlossary) -> DocumentGlossary:
        try:
            # Serialize the Pydantic Domain models for JSONB
            serialized_items = [item.model_dump(mode='json') for item in glossary.items]

            db_model = DocumentGlossaryModel(
                id=glossary.id,
                source_id=glossary.source_id,
                domain_category=glossary.domain_category,
                items=serialized_items,
                created_at=glossary.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return glossary

        except Exception as e:
            raise map_asr_correction_error(e, context="save_glossary") from e

    def get_glossary(self, source_id: uuid.UUID) -> Optional[DocumentGlossary]:
        try:
            with self._db.session() as session:
                stmt = select(DocumentGlossaryModel).where(DocumentGlossaryModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Deserialize while the DB session is open
                domain_items = [GlossaryItem(**item) for item in db_model.items]

                return DocumentGlossary(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    domain_category=db_model.domain_category,
                    items=domain_items,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_asr_correction_error(e, context="get_glossary") from e

    def save_corrected_transcript(self, transcript: DomainCorrectedTranscript) -> DomainCorrectedTranscript:
        try:
            # Serialize sentences and audit records for JSONB
            serialized_sentences = [s.model_dump(mode='json') for s in transcript.sentences]
            serialized_audit = [a.model_dump(mode='json') for a in transcript.audit_trail]

            db_model = DomainCorrectedTranscriptModel(
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
            raise map_asr_correction_error(e, context="save_corrected_transcript") from e

    def get_corrected_transcript(self, source_id: uuid.UUID) -> Optional[DomainCorrectedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(DomainCorrectedTranscriptModel).where(DomainCorrectedTranscriptModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # Deserialize
                domain_sentences = [ReconstructedSentence(**s) for s in db_model.sentences]
                domain_audit = [CorrectionAuditRecord(**a) for a in db_model.audit_trail]

                return DomainCorrectedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    sentences=domain_sentences,
                    audit_trail=domain_audit,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_asr_correction_error(e, context="get_corrected_transcript") from e