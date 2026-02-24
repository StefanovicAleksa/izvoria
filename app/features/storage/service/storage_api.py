# filename: storage_api_v2.py
# path: app/features/storage/service/storage_api.py
import os
import uuid
import mimetypes
from pathlib import Path
from typing import List, Optional

from app.features.storage.domain.entities import Source, File, SourceType
from app.features.storage.domain.interfaces import IStorageRepository
from app.features.storage.utils.file_manager import FileManager
from app.features.storage.utils.hashing import calculate_file_hash
from app.features.storage.exceptions.base import StorageError

class StorageAPI:
    def __init__(self, repository: IStorageRepository):
        self.repository = repository
        self.file_manager = FileManager()

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

    def ingest_local_file(self, file_path: str, source_type: SourceType,
                          name: Optional[str] = None, delete_original: bool = False) -> Source:
        """Deduplicated ingestion that creates a business Source linked to a File."""
        source_path_obj = Path(file_path).resolve()
        if not source_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content_hash = calculate_file_hash(str(source_path_obj))
            existing = self.repository.get_by_hash(content_hash)
            if existing:
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
                file_id=file_id,
                file=new_file
            )
            return self.repository.create(new_source)
        except Exception as e:
            raise StorageError(f"Source ingestion failed: {str(e)}") from e

    def get_source(self, source_id: uuid.UUID) -> Optional[Source]:
        return self.repository.get_by_id(source_id)

    def get_file(self, file_id: uuid.UUID) -> Optional[File]:
        """Fetches a standalone raw file directly by its UUID."""
        return self.repository.get_file_by_id(file_id)

    def list_sources(self) -> List[Source]:
        return self.repository.list_all()