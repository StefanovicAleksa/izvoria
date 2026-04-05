# Canvas: visual_hallucination_repositories_v2
# Path: app/features/visual_hallucination_correction/data/repositories.py

import uuid
from typing import Optional
from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.visual_hallucination_correction.domain.interfaces import IVisualHallucinationRepository
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationAudit,
    VisualHallucinationEvaluatedTranscript
)
from app.features.visual_hallucination_correction.data.models import VisualHallucinationEvaluationModel
from app.features.visual_hallucination_correction.utils.error_mapper import map_hallucination_error


class VisualHallucinationRepository(IVisualHallucinationRepository):
    """
    PostgreSQL implementation for persisting the Hallucination Audit ledger.
    Handles serialization between robust Pydantic Domain Entities and PostgreSQL JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: VisualHallucinationEvaluatedTranscript) -> VisualHallucinationEvaluatedTranscript:
        try:
            serialized_audits = [audit.model_dump(mode='json') for audit in transcript.audits]

            db_model = VisualHallucinationEvaluationModel(
                id=transcript.id,
                source_id=transcript.source_id,
                audits=serialized_audits,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return transcript

        except Exception as e:
            # FIX: Included "transcript audit" in context so the error_mapper identifies it
            raise map_hallucination_error(
                e,
                context=f"VisualHallucinationRepository.save transcript audit for source {transcript.source_id}"
            ) from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[VisualHallucinationEvaluatedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(VisualHallucinationEvaluationModel).where(
                    VisualHallucinationEvaluationModel.source_id == source_id
                )
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                domain_audits = [VisualHallucinationAudit(**a) for a in db_model.audits]

                return VisualHallucinationEvaluatedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    audits=domain_audits,
                    created_at=db_model.created_at
                )

        except Exception as e:
            # FIX: Included "transcript audit" in context so the error_mapper identifies it
            raise map_hallucination_error(
                e,
                context=f"VisualHallucinationRepository.get_by_source_id transcript audit for {source_id}"
            ) from e