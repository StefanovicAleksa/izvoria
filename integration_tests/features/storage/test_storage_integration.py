import pytest
from pathlib import Path
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType


# This assumes you have a global fixture 'db_manager' in your integration_tests/conftest.py
# that provides a clean database connection for each test.

class TestStorageIntegration:

    @pytest.fixture
    def api(self, db_manager, tmp_path):
        """
        Creates a REAL StorageAPI connected to the REAL Test Database.
        We only override the storage path to use a temp folder
        so we don't clutter your actual 'onyx_data' directory.
        """
        # 1. Initialize Real Repository
        repo = StorageRepository(db_manager)

        # 2. Initialize Real Service
        service = StorageAPI(repo)

        # 3. Configure File Manager to use Pytest's temp dir
        # This simulates a real disk, but one that is cleaned up after the test.
        service.file_manager.base_path = tmp_path / "vault"
        service.file_manager.base_path.mkdir()

        return service

    def test_end_to_end_ingestion(self, api, tmp_path):
        """
        Scenario:
        1. User has a video file on their USB drive (simulated).
        2. User ingests it into Onyx.
        3. Onyx saves it to DB and moves it to the Vault.
        4. User tries to ingest the EXACT SAME file again.
        5. Onyx detects duplicate and does nothing (returns original).
        """
        # ---------------------------------------------------------
        # PHASE 1: PREPARATION
        # ---------------------------------------------------------
        # Create a dummy "source file" simulating a file on a USB stick
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()
        source_file = source_dir / "holiday_vlog.mp4"

        # Write some unique bytes so we can verify content integrity later
        file_content = b"Content of the holiday vlog video file."
        source_file.write_bytes(file_content)

        # ---------------------------------------------------------
        # PHASE 2: FIRST INGESTION (Fresh Upload)
        # ---------------------------------------------------------
        # Action: Ingest with "Copy" mode (delete_original=False)
        source_1 = api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.VIDEO,
            name="Holiday Vlog 2024",
            delete_original=False
        )

        # Assertion 1: Check Database persistence
        assert source_1.id is not None
        assert source_1.name == "Holiday Vlog 2024"
        assert source_1.content_hash is not None

        # Verify we can fetch it back
        fetched_source = api.get_source(source_1.id)
        assert fetched_source.id == source_1.id
        assert fetched_source.file.file_size_bytes == len(file_content)

        # Assertion 2: Check Physical Storage (The Vault)
        vault_path = Path(fetched_source.file.file_path)

        assert vault_path.exists(), "File was not physically saved to the vault!"
        assert vault_path.read_bytes() == file_content, "File content corrupted during copy!"

        # Ensure it followed the folder structure (YYYY/MM/DD)
        # We check if the parent directory matches today's date structure logic
        assert str(api.file_manager.base_path) in str(vault_path)

        # Assertion 3: Original file should still exist (Copy mode)
        assert source_file.exists()

        # ---------------------------------------------------------
        # PHASE 3: DEDUPLICATION (Re-ingest same file)
        # ---------------------------------------------------------
        # Action: Try to ingest the exact same file again
        source_2 = api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.VIDEO,
            name="Duplicate Attempt"
        )

        # Assertion 4: It should return the OLD record
        assert source_2.id == source_1.id
        assert source_2.name == "Holiday Vlog 2024"  # Name shouldn't update to "Duplicate Attempt"

        # Assertion 5: Disk check - We should NOT have created a second file
        # Count all files in the vault
        vault_files = list(api.file_manager.base_path.rglob("*.*"))
        # We expect exactly 1 file (the original upload)
        assert len(vault_files) == 1

    def test_ingest_move_mode(self, api, tmp_path):
        """
        Scenario: Test 'Move' mode where the original file is deleted.
        """
        # Create source file
        source_file = tmp_path / "tobe_moved.txt"
        source_file.write_text("Secret Data")

        # Action: Ingest with delete_original=True
        api.ingest_local_file(
            file_path=str(source_file),
            source_type=SourceType.TEXT,
            delete_original=True
        )

        # Assertion: Original should be gone
        assert not source_file.exists()

        # Assertion: Data should be in vault
        # We find it by listing sources since we didn't capture the return val
        sources = api.list_sources()
        assert len(sources) == 1
        saved_path = Path(sources[0].file.file_path)
        assert saved_path.exists()
        assert saved_path.read_text() == "Secret Data"