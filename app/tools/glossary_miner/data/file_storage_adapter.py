# Canvas: glossary_miner_file_storage_adapter_v1
# Path: app/features/glossary_miner/data/file_storage_adapter.py

from pathlib import Path

from app.tools.glossary_miner.domain.interfaces import IGlossaryStorageAdapter
from app.tools.glossary_miner.domain.entities import IzvoriaGlossaryManifest
from app.tools.glossary_miner.exceptions import DataProcessingError


class LocalFileStorageAdapter(IGlossaryStorageAdapter):
    """
    Concrete implementation for persisting the final Glossary Manifest to disk.
    Saves directly to the Izvoria appliance's data directory.
    """

    def __init__(self, base_dir: str = "data/glossaries"):
        # Resolve path relative to the project root
        self.base_path = Path(base_dir).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_manifest(self, file_name: str, manifest: IzvoriaGlossaryManifest) -> str:
        if not file_name.endswith(".json"):
            file_name += ".json"

        file_path = self.base_path / file_name

        try:
            # model_dump_json() ensures we get strict, valid JSON from Pydantic
            json_data = manifest.model_dump_json(indent=4)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_data)

            return str(file_path)

        except Exception as e:
            # Catch I/O errors (e.g., permission denied)
            raise DataProcessingError(
                message=f"Failed to write glossary manifest to disk at {file_path}",
                original_exception=e
            ) from e