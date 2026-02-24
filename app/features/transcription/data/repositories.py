# filename: transcription_repositories_v3.py
# path: app/features/transcription/data/repositories.py
import uuid
from typing import Optional

from sqlalchemy import select

from app.core.database.service.manager import DatabaseManager
from app.features.transcription.domain.interfaces import ITranscriptRepository
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.transcription.data.models import TranscriptModel
from app.features.transcription.utils.error_mapper import map_transcription_error


class TranscriptRepository(ITranscriptRepository):
    """
    PostgreSQL implementation of the Transcript Repository using SQLAlchemy and JSONB.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, transcript: Transcript) -> Transcript:
        try:
            serialized_segments = [segment.model_dump() for segment in transcript.segments]

            db_model = TranscriptModel(
                id=transcript.id,
                source_id=transcript.source_id,
                segments=serialized_segments,
                created_at=transcript.created_at
            )

            with self._db.session() as session:
                session.add(db_model)

            return transcript

        except Exception as e:
            raise map_transcription_error(e) from e

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[Transcript]:
        try:
            with self._db.session() as session:
                stmt = select(TranscriptModel).where(TranscriptModel.source_id == source_id)
                db_model = session.execute(stmt).scalar_one_or_none()

                if not db_model:
                    return None

                # CRITICAL FIX: Deserialize the JSONB while the session is STILL OPEN
                domain_segments = [TranscriptSegment(**seg) for seg in db_model.segments]

                return Transcript(
                    id=db_model.id,
                    source_id=db_model.source_id,
                    segments=domain_segments,
                    created_at=db_model.created_at
                )

        except Exception as e:
            raise map_transcription_error(e) from e