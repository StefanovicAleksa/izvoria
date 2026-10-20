# Path: app/features/source_profile/data/postgres_profile_repo.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from app.core.database.domain.interfaces import IDatabaseManager
from app.features.source_profile.data.models import SourceProfileModel
from app.features.source_profile.domain.entities import ProfiledSource, SourceIdentity, SourceProfile
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.domain.interfaces import ISourceProfileRepository
from app.features.source_profile.utils.error_mapper import map_source_profile_error

_IDENTITY_SQL = text("""
    SELECT s.id AS source_id, s.name, s.language, f.file_name
    FROM sources s
    LEFT JOIN files f ON f.id = s.file_id
    WHERE s.id = :source_id
""")

# Sources with indexed chunks, and their profiles. EXISTS rather than a join over vector_chunks: one index probe per
# source instead of a scan of every chunk.
_INDEXED_SQL = text("""
    SELECT s.id AS source_id, s.name, s.language, f.file_name,
           p.title, p.topic, p.origin, p.language AS profile_language
    FROM sources s
    LEFT JOIN files f ON f.id = s.file_id
    LEFT JOIN source_profiles p ON p.source_id = s.id
    WHERE EXISTS (SELECT 1 FROM vector_chunks v WHERE v.source_id = s.id)
    ORDER BY s.created_at, s.id
""")

def _identity(row) -> SourceIdentity:
    return SourceIdentity(
        source_id=row["source_id"], name=row["name"] or "", file_name=row["file_name"] or "", language=row["language"]
    )


def _to_profile(model: SourceProfileModel) -> SourceProfile:
    return SourceProfile(source_id=model.source_id, title=model.title, topic=model.topic or "",
                         origin=ProfileOrigin(model.origin), language=model.language)


class PostgresSourceProfileRepository(ISourceProfileRepository):

    def __init__(self, database: IDatabaseManager):
        self._db = database

    def get_identity(self, source_id: uuid.UUID) -> Optional[SourceIdentity]:
        try:
            with self._db.session() as session:
                row = session.execute(_IDENTITY_SQL, {"source_id": source_id}).mappings().first()
                return _identity(row) if row else None
        except Exception as e:
            raise map_source_profile_error(e, context=f"Reading source {source_id}") from e

    def get(self, source_id: uuid.UUID) -> Optional[SourceProfile]:
        try:
            with self._db.session() as session:
                model = session.execute(
                    select(SourceProfileModel).where(SourceProfileModel.source_id == source_id)
                ).scalar_one_or_none()
                return _to_profile(model) if model else None
        except Exception as e:
            raise map_source_profile_error(e, context=f"Reading the profile of source {source_id}") from e

    def save(self, profile: SourceProfile) -> SourceProfile:
        try:
            values = {
                "source_id": profile.source_id,
                "title": profile.title,
                "topic": profile.topic,
                "origin": profile.origin.value,
                "language": profile.language,
                "updated_at": datetime.now(timezone.utc),
            }
            statement = insert(SourceProfileModel).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=[SourceProfileModel.source_id],
                set_={key: statement.excluded[key] for key in values if key != "source_id"}
            )
            with self._db.session() as session:
                session.execute(statement)
            return profile
        except Exception as e:
            raise map_source_profile_error(e, context=f"Saving the profile of source {profile.source_id}") from e

    def list_indexed(self) -> List[ProfiledSource]:
        try:
            with self._db.session() as session:
                rows = session.execute(_INDEXED_SQL).mappings().all()
            return [
                ProfiledSource(
                    identity=_identity(row),
                    profile=SourceProfile(
                        source_id=row["source_id"], title=row["title"], topic=row["topic"] or "",
                        origin=ProfileOrigin(row["origin"]), language=row["profile_language"]
                    ) if row["title"] else None
                )
                for row in rows
            ]
        except Exception as e:
            raise map_source_profile_error(e, context="Listing indexed sources and their profiles") from e

    def list_unprofiled(self) -> List[SourceIdentity]:
        return [source.identity for source in self.list_indexed() if source.profile is None]

