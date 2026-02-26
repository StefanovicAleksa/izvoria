# filename: integration_tests/features/storage/test_storage_integration.py

import pytest
from pathlib import Path
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType, File


class TestStorageIntegration:

    @pytest.fixture
    def api(self, db_manager, tmp_path):
        """Creates a REAL StorageAPI connected to the REAL Test Database."""
        repo = StorageRepository(db_manager)
        service = StorageAPI(repo)
        service.file_manager.base_path = tmp_path / "vault"
        service.file_manager.base_path.mkdir()
        return service

    def test_ingest_raw_file_integration(self, api, tmp_path):
        """Tests storing a standalone derivative file completely independent of a Source."""
        # 1. Prepare
        source_file = tmp_path / "extracted_audio.wav"
        file_content = b"Fake audio byte stream"
        source_file.write_bytes(file_content)

        # 2. Action (Vault the pure file)
        vaulted_file = api.ingest_raw_file(
            file_path=str(source_file),
            delete_original=False
        )

        # 3. Assertions
        assert isinstance(vaulted_file, File)
        assert vaulted_file.id is not None
        assert vaulted_file.file_name == "extracted_audio.wav"
        assert vaulted_file.file_size_bytes == len(file_content)

        # Ensure physical persistence
        vault_path = Path(vaulted_file.file_path)
        assert vault_path.exists()
        assert vault_path.read_bytes() == file_content

    def test_end_to_end_ingestion(self, api, tmp_path):
        """Scenario: Ingest, Vault, and Deduplicate a Source File."""
        # Phase 1: Preparation
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        source_file = source_dir / "holiday_vlog.mp4"
        file_content = b"Content of the holiday vlog video file."
        source_file.write_bytes(file_content)

        # Phase 2: First Ingestion (Fresh Upload)
        source_1 = api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.VIDEO,
            name="Holiday Vlog 2024",
            delete_original=False
        )

        # Assertion 1: Database persistence
        assert source_1.id is not None
        assert source_1.name == "Holiday Vlog 2024"
        assert source_1.content_hash is not None
        assert source_1.file_id is not None  # Verify the new relationship constraint

        # Verify we can fetch it back
        fetched_source = api.get_source(source_1.id)
        assert fetched_source.id == source_1.id
        assert fetched_source.file.file_size_bytes == len(file_content)

        # Assertion 2: Physical Storage Check
        vault_path = Path(fetched_source.file.file_path)
        assert vault_path.exists()
        assert vault_path.read_bytes() == file_content
        assert source_file.exists()  # Ensure it was copied, not moved

        # Phase 3: Deduplication
        source_2 = api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.VIDEO,
            name="Duplicate Attempt"
        )

        # Assertion 3: Should return OLD record
        assert source_2.id == source_1.id
        assert source_2.name == "Holiday Vlog 2024"

        # Assertion 4: Disk check - exactly 1 file should exist
        vault_files = list(api.file_manager.base_path.rglob("*.*"))
        assert len(vault_files) == 1


    def test_ingest_move_mode(self, api, tmp_path):
        """Scenario: Test 'Move' mode where original file is deleted."""
        source_file = tmp_path / "tobe_moved.txt"
        source_file.write_text("Secret Data")

        api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.TEXT,
            delete_original=True
        )

        assert not source_file.exists()

        sources = api.list_sources()

        # FIXED: Look specifically for the file we just ingested
        # instead of assuming the database is completely empty.
        saved_sources = [s for s in sources if s.name == "tobe_moved.txt"]
        assert len(saved_sources) == 1

        saved_path = Path(saved_sources[0].file.file_path)
        assert saved_path.exists()
        assert saved_path.read_text() == "Secret Data"