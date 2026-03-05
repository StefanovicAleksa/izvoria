# Canvas: sentence_repository_v3
# Path: app/features/sentence_reconstruction/data/repositories.py

import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.sentence_reconstruction.domain.interfaces import ISentenceRepository
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.sentence_reconstruction.data.models import ReconstructedTranscriptModel
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error


class PostgresSentenceRepository(ISentenceRepository):
    """
    PostgreSQL implementation of the Sentence Repository using SQLAlchemy and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: ReconstructedTranscript) -> ReconstructedTranscript:
        try:
            # Safely convert Pydantic models to JSON-compatible dictionaries
            serialized_sentences = [sentence.model_dump(mode='json') for sentence in transcript.sentences]

            db_model = ReconstructedTranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                sentences=serialized_sentences,
                created_at=transcript.created_at
            )

            # FIXED: Using the DatabaseManager context manager
            with self._db.session() as session:
                session.add(db_model)

            return transcript

        except Exception as e:
            raise map_sentence_error(e) from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[ReconstructedTranscript]:
        try:
            with self._db.session() as session:
                stmt = select(ReconstructedTranscriptModel).where(ReconstructedTranscriptModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # CRITICAL: Deserialize the JSONB while the session is STILL OPEN
                domain_sentences = [ReconstructedSentence(**s) for s in db_model.sentences]

                return ReconstructedTranscript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    sentences=domain_sentences,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_sentence_error(e) from e