from typing import Optional, List
import uuid
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.core.database.domain.interfaces import IDatabaseManager
from app.features.storage.domain.interfaces import IStorageRepository
from app.features.storage.domain.entities import Source
from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.utils.error_mapper import map_storage_error
from app.features.storage.exceptions.not_found import SourceNotFound

class StorageRepository(IStorageRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def create(self, source: Source) -> Source:
        """
        Saves a new Source and its associated File in a single transaction.
        """
        try:
            with self._db.session() as session:
                # 1. Create the Source Model
                source_model = SourceModel(
                    id=source.id,
                    name=source.name,
                    source_type=source.source_type.value,
                    content_hash=source.content_hash,
                    created_at=source.created_at,
                    updated_at=source.updated_at
                )

                # 2. Create the File Model (if present)
                if source.file:
                    file_model = FileModel(
                        id=source.file.id,
                        source_id=source.id,  # Link to parent
                        file_path=source.file.file_path,
                        file_name=source.file.file_name,
                        file_size_bytes=source.file.file_size_bytes,
                        mime_type=source.file.mime_type,
                        file_metadata=source.file.file_metadata,
                        is_processed=source.file.is_processed,
                        created_at=source.file.created_at
                    )
                    source_model.file = file_model

                # 3. Add to session and commit
                session.add(source_model)
                # Commit is handled by the context manager automatically on exit

                # 4. Refresh to get any DB-generated defaults (if needed)
                # But since we generate IDs in Python, we rarely need this.

                return source

        except SQLAlchemyError as e:
            raise map_storage_error(e)

    def get_by_id(self, source_id: uuid.UUID) -> Optional[Source]:
        """
        Fetch a source by ID. Eagerly loads the file info.
        """
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).where(SourceModel.id == source_id)
                model = session.execute(stmt).scalar_one_or_none()

                if not model:
                    return None

                return Source.model_validate(model)
        except SQLAlchemyError as e:
            raise map_storage_error(e)

    def get_by_hash(self, content_hash: str) -> Optional[Source]:
        """
        The Deduplication Engine.
        Used to check if a file already exists before processing.
        """
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).where(SourceModel.content_hash == content_hash)
                model = session.execute(stmt).scalar_one_or_none()

                if not model:
                    return None

                return Source.model_validate(model)
        except SQLAlchemyError as e:
            raise map_storage_error(e)

    def list_all(self) -> List[Source]:
        """
        Lists all sources.
        """
        try:
            with self._db.session() as session:
                # We order by created_at desc to show newest first
                stmt = select(SourceModel).order_by(SourceModel.created_at.desc())
                models = session.execute(stmt).scalars().all()

                return [Source.model_validate(m) for m in models]
        except SQLAlchemyError as e:
            raise map_storage_error(e)

    def mark_as_processed(self, source_id: uuid.UUID) -> Source:
        """
        Updates the is_processed flag on the associated file.
        Raises SourceNotFound if the source or file does not exist.
        """
        try:
            with self._db.session() as session:
                # 1. Fetch the model
                stmt = select(SourceModel).where(SourceModel.id == source_id)
                source_model = session.execute(stmt).scalar_one_or_none()

                # 2. Guard Clause: Check existence
                if not source_model:
                    raise SourceNotFound(f"Source with ID {source_id} not found.")

                if not source_model.file:
                    # Technically a data integrity error, but treated as Not Found for the caller
                    raise SourceNotFound(f"Source {source_id} exists but has no file record.")

                # 3. Update the flag (Now reachable!)
                source_model.file.is_processed = True

                # 4. Commit happens automatically on context manager exit

                return Source.model_validate(source_model)

        except SQLAlchemyError as e:
            raise map_storage_error(e)