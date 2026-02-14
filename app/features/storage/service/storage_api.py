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

    def ingest_local_file(self,
                          file_path: str,
                          source_type: SourceType,
                          name: Optional[str] = None,
                          delete_original: bool = False) -> Source:
        """
        Ingests a local file into the vault.
        """
        source_path_obj = Path(file_path).resolve()

        # 1. Validate existence
        if not source_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # 2. Calculate Hash (The fingerprint)
            content_hash = calculate_file_hash(str(source_path_obj))

            # 3. Check Deduplication
            existing_source = self.repository.get_by_hash(content_hash)
            if existing_source:
                if delete_original:
                    os.remove(source_path_obj)
                return existing_source

            # 4. Prepare Destination
            file_ext = source_path_obj.suffix
            dest_path = self.file_manager.generate_path(file_ext)

            # 5. Move/Copy file to Vault
            # Returns the actual size of the file on disk
            file_size = self.file_manager.store_file(
                source_path=source_path_obj,
                dest_path=dest_path,
                move=delete_original
            )

            # 6. Basic Metadata Detection
            # guess_type is fast (checks extension only), perfect for ingestion.
            guessed_mime, _ = mimetypes.guess_type(dest_path)

            # 7. Create Domain Entities
            # We explicitly generate UUIDs here so they are linked correctly.
            source_id = uuid.uuid4()
            file_id = uuid.uuid4()

            new_file = File(
                id=file_id,
                source_id=source_id, # Link to Parent
                file_path=str(dest_path),
                file_name=source_path_obj.name,
                file_size_bytes=file_size, # Use the confirmed size from disk
                mime_type=guessed_mime,    # e.g., "video/mp4"
                file_metadata={}           # Deep metadata (duration, resolution) is left for the Processor
            )

            new_source = Source(
                id=source_id,
                name=name or source_path_obj.name,
                source_type=source_type,
                content_hash=content_hash,
                file=new_file  # We attach the file here!
            )

            # 8. Persist Everything
            saved_source = self.repository.create(new_source)

            return saved_source

        except Exception as e:
            # Error Handling: Wrap unknown errors in StorageError
            if isinstance(e, StorageError):
                raise e
            raise StorageError(f"Ingestion failed: {str(e)}") from e

    def get_source(self, source_id: uuid.UUID) -> Optional[Source]:
        """Retrieves a source by its UUID."""
        return self.repository.get_by_id(source_id)

    def list_sources(self) -> List[Source]:
        """Lists all sources in the system."""
        return self.repository.list_all()