# Canvas: storage_api_v3
# Path: app/features/storage/service/storage_api.py
import os
import uuid
import mimetypes
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.features.storage.domain.entities import Source, File, SourceType
from app.features.storage.domain.interfaces import IStorageRepository
from app.features.storage.data.file_manager import FileManager
from app.features.storage.utils.hashing import calculate_file_hash
from app.features.storage.utils.upload_record import read_record
from app.features.storage.exceptions.base import StorageError
from app.features.storage.exceptions.conflict import SourceLanguageConflict
from app.features.storage.exceptions.not_found import FileNotFound


class StorageAPI:
    def __init__(self, repository: IStorageRepository, file_manager: Optional[FileManager] = None):
        self.repository = repository
        self.file_manager = file_manager or FileManager()

    def ingest_raw_file(self, file_path: str, delete_original: bool = False) -> File:
        """Vaults a standalone file without a Source (e.g., audio extractions)."""
        source_path_obj = Path(file_path).resolve()
        if not source_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            file_ext = source_path_obj.suffix
            dest_path = self.file_manager.generate_path(file_ext)
            file_size = self.file_manager.store_file(
                source_path=source_path_obj,
                dest_path=dest_path,
                move=delete_original
            )
            guessed_mime, _ = mimetypes.guess_type(dest_path)

            new_file = File(
                id=uuid.uuid4(),
                file_path=str(dest_path),
                file_name=source_path_obj.name,
                file_size_bytes=file_size,
                mime_type=guessed_mime
            )
            return self.repository.create_file(new_file)
        except Exception as e:
            raise StorageError(f"Raw ingestion failed: {str(e)}") from e

    def ingest_local_file(self, file_path: str, source_type: SourceType, name: Optional[str] = None,
                          delete_original: bool = False, language: Optional[str] = None) -> Source:
        """
        Deduplicated ingestion that creates a business Source linked to a File, stamped with the language the
        batch is ingested in (the active language unless given). A file already ingested in another language is
        refused (SourceLanguageConflict): every stage cached for it was produced in that language. A source
        nothing has bound yet is re-stamped.
        """
        language = language or settings.APPLIANCE_LANGUAGE
        source_path_obj = Path(file_path).resolve()
        if not source_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # A file prepared from an upload (the hosted 720p copy) is known by the upload's hash and name, so the
            # same upload still finds its source (utils/upload_record.py)
            record = read_record(source_path_obj)
            content_hash = record.sha256 if record else calculate_file_hash(str(source_path_obj))
            name = name or (record.original_name if record else None)
            existing = self.repository.get_by_hash(content_hash)
            if existing:
                if existing.language and existing.language != language:
                    raise SourceLanguageConflict(
                        f"{source_path_obj.name} was already ingested as '{existing.language}'. Ingest it with "
                        f"'{existing.language}' selected, or reset the index to change its language."
                    )
                if existing.language is None:
                    existing = self.repository.set_language(existing.id, language)
                if delete_original: os.remove(source_path_obj)
                return existing

            dest_path = self.file_manager.generate_path(source_path_obj.suffix)
            file_size = self.file_manager.store_file(source_path_obj, dest_path, delete_original)
            guessed_mime, _ = mimetypes.guess_type(dest_path)

            file_id = uuid.uuid4()
            new_file = File(
                id=file_id,
                file_path=str(dest_path),
                file_name=source_path_obj.name,
                file_size_bytes=file_size,
                mime_type=guessed_mime
            )

            new_source = Source(
                id=uuid.uuid4(),
                name=name or source_path_obj.name,
                source_type=source_type,
                content_hash=content_hash,
                language=language,
                file_id=file_id,
                file=new_file
            )
            return self.repository.create(new_source)
        except SourceLanguageConflict:
            raise
        except Exception as e:
            raise StorageError(f"Source ingestion failed: {str(e)}") from e

    def clear_language(self, source_id: uuid.UUID) -> Source:
        """
        Un-stamps a source whose speech failed the language check, so ingesting it again with the right language
        is accepted instead of refused as a conflict.
        """
        return self.repository.set_language(source_id, None)

    def get_source(self, source_id: uuid.UUID) -> Optional[Source]:
        return self.repository.get_by_id(source_id)

    def get_file(self, file_id: uuid.UUID) -> Optional[File]:
        """Fetches a standalone raw file directly by its UUID."""
        return self.repository.get_file_by_id(file_id)

    def delete_raw_file(self, file_id: uuid.UUID) -> bool:
        """
        Removes a standalone vaulted file (DB record + physical file).
        Used to roll back files vaulted by a feature whose run failed before recording them.
        """
        deleted = self.repository.delete_file(file_id)
        if not deleted:
            return False
        self.file_manager.delete(deleted.file_path)
        return True

    def replace_file(self, file_id: uuid.UUID, new_path: str | Path) -> File:
        """
        Moves `new_path` into the vault in place of a file's current bytes, then deletes the old bytes. The record
        keeps its id, so everything pointing at it follows; its path, size and type become the new file's. A
        source's content_hash stays the hash of what was uploaded, so the same upload still dedups.
        """
        record = self.repository.get_file_by_id(file_id)
        if not record:
            raise FileNotFound(f"Database record missing for file ID: {file_id}")
        new_src = Path(new_path)
        dest_path = self.file_manager.generate_path(new_src.suffix)
        try:
            size = self.file_manager.store_file(new_src, dest_path, move=True)
            guessed_mime, _ = mimetypes.guess_type(dest_path)
            updated = self.repository.update_file(record.model_copy(update={
                "file_path": str(dest_path), "file_size_bytes": size, "mime_type": guessed_mime,
            }))
        except Exception as e:
            self.file_manager.delete(dest_path)
            raise StorageError(f"Replacing file {file_id} failed: {str(e)}") from e
        self.file_manager.delete(record.file_path)
        return updated

    def get_file_path(self, file_id: uuid.UUID) -> Path:
        """
        Fetches a file record and strictly verifies the physical file exists in the vault.
        Used securely by internal features (like Visual Analysis) to locate JPEGs.
        """
        file_record = self.repository.get_file_by_id(file_id)
        if not file_record:
            raise FileNotFound(f"Database record missing for file ID: {file_id}")

        physical_path = Path(file_record.file_path)
        if not physical_path.exists():
            raise FileNotFoundError(f"Physical file missing from Izvoria Vault: {physical_path}")

        return physical_path

    def list_sources(self) -> List[Source]:
        return self.repository.list_all()