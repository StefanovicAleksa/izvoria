# filename: storage_repositories_v2.py
# path: app/features/storage/data/repositories.py
from typing import Optional, List
import uuid
from sqlalchemy import select

from app.core.database.domain.interfaces import IDatabaseManager
from app.core.database import exceptions as db_exc
from app.features.storage.domain.interfaces import IStorageRepository
from app.features.storage.domain.entities import Source, File
from app.features.storage.data.models import SourceModel, FileModel
from app.features.storage.utils.error_mapper import map_storage_error
from app.features.storage.exceptions.not_found import SourceNotFound, FileNotFound

class StorageRepository(IStorageRepository):
    def __init__(self, db_manager: IDatabaseManager):
        self._db = db_manager

    def create_file(self, file: File) -> File:
        """Saves a standalone physical file record."""
        try:
            with self._db.session() as session:
                file_model = FileModel(
                    id=file.id,
                    file_path=file.file_path,
                    file_name=file.file_name,
                    file_size_bytes=file.file_size_bytes,
                    mime_type=file.mime_type,
                    file_metadata=file.file_metadata,
                    is_processed=file.is_processed,
                    created_at=file.created_at
                )
                session.add(file_model)
                return file
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def create(self, source: Source) -> Source:
        """Saves a new Source and its associated File in a single transaction."""
        try:
            with self._db.session() as session:
                # 1. Create File first
                if source.file:
                    file_model = FileModel(
                        id=source.file.id,
                        file_path=source.file.file_path,
                        file_name=source.file.file_name,
                        file_size_bytes=source.file.file_size_bytes,
                        mime_type=source.file.mime_type,
                        file_metadata=source.file.file_metadata,
                        is_processed=source.file.is_processed,
                        created_at=source.file.created_at
                    )
                    session.add(file_model)

                # 2. Create Source
                source_model = SourceModel(
                    id=source.id,
                    name=source.name,
                    source_type=source.source_type.value,
                    content_hash=source.content_hash,
                    language=source.language,
                    file_id=source.file_id,
                    created_at=source.created_at,
                    updated_at=source.updated_at
                )
                session.add(source_model)
                return source
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def get_by_id(self, source_id: uuid.UUID) -> Optional[Source]:
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).where(SourceModel.id == source_id)
                model = session.execute(stmt).scalar_one_or_none()
                return Source.model_validate(model) if model else None
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def get_file_by_id(self, file_id: uuid.UUID) -> Optional[File]:
        try:
            with self._db.session() as session:
                stmt = select(FileModel).where(FileModel.id == file_id)
                model = session.execute(stmt).scalar_one_or_none()
                return File.model_validate(model) if model else None
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def delete_file(self, file_id: uuid.UUID) -> Optional[File]:
        """Deletes a standalone physical file record."""
        try:
            with self._db.session() as session:
                model = session.get(FileModel, file_id)
                if not model:
                    return None
                deleted = File.model_validate(model)
                session.delete(model)
                return deleted
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def update_file(self, file: File) -> File:
        try:
            with self._db.session() as session:
                model = session.get(FileModel, file.id)
                if not model:
                    raise FileNotFound(f"File with ID {file.id} not found.")
                model.file_path = file.file_path
                model.file_size_bytes = file.file_size_bytes
                model.mime_type = file.mime_type
                return File.model_validate(model)
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def get_by_hash(self, content_hash: str) -> Optional[Source]:
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).where(SourceModel.content_hash == content_hash)
                model = session.execute(stmt).scalar_one_or_none()
                return Source.model_validate(model) if model else None
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def list_all(self) -> List[Source]:
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).order_by(SourceModel.created_at.desc())
                models = session.execute(stmt).scalars().all()
                return [Source.model_validate(m) for m in models]
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def mark_as_processed(self, source_id: uuid.UUID) -> Source:
        try:
            with self._db.session() as session:
                stmt = select(SourceModel).where(SourceModel.id == source_id)
                source_model = session.execute(stmt).scalar_one_or_none()
                if not source_model:
                    raise SourceNotFound(f"Source with ID {source_id} not found.")
                source_model.file.is_processed = True
                return Source.model_validate(source_model)
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)

    def set_language(self, source_id: uuid.UUID, language: Optional[str]) -> Source:
        try:
            with self._db.session() as session:
                source_model = session.get(SourceModel, source_id)
                if not source_model:
                    raise SourceNotFound(f"Source with ID {source_id} not found.")
                source_model.language = language
                return Source.model_validate(source_model)
        except db_exc.DatabaseError as e:
            raise map_storage_error(e)
