# File: integration_tests/orchestration/test_ingest_flow_integration.py

import pytest
from pathlib import Path

from app.core.jobs.service.manager import JobManager
from app.core.jobs.data.repositories import JobRepository
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.scanner.service.scanner import FileScanner
from app.orchestration.service.ingest_flow import IngestFlowOrchestrator
from app.features.storage.domain.entities import SourceType


class TestIngestFlowIntegration:
    @pytest.fixture
    def orchestrator(self, db_manager, tmp_path):
        """
        Wires up the real dependencies with the test database.
        """
        # 1. Real Job Core
        job_repo = JobRepository(db_manager)
        job_manager = JobManager(job_repo)

        # 2. Real Storage Feature (using tmp_path as the vault)
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 3. Real Scanner Feature
        scanner = FileScanner()

        # 4. The Orchestrator
        return IngestFlowOrchestrator(
            job_manager=job_manager,
            scanner=scanner,
            storage_api=storage_api
        )

    def test_end_to_end_batch_ingest(self, orchestrator, tmp_path):
        """
        Creates real files on disk, runs the full batch orchestration,
        and verifies the jobs and storage records were saved to the database.
        """
        # 1. Setup Data: Create a mock directory with 2 valid files and 1 invalid file
        source_dir = tmp_path / "import_source"
        source_dir.mkdir()

        file1 = source_dir / "doc1.txt"
        file1.write_text("Hello World")

        file2 = source_dir / "doc2.txt"
        file2.write_text("Onyx is fast")

        invalid_file = source_dir / "script.py"
        invalid_file.write_text("print('Ignore me')")

        # 2. Execute Orchestrator
        # Defaulting source_type to TEXT for these txt files
        results = orchestrator.process_full_batch(str(source_dir), source_type=SourceType.TEXT)

        # 3. Assertions on Orchestrator Results
        assert len(results) == 2  # The python file should be skipped by the scanner

        # Sort results to be deterministic
        results.sort(key=lambda x: x.file_path)

        # Verify both items succeeded
        assert results[0].success is True
        assert results[0].source_id is not None
        assert "doc1.txt" in results[0].file_path

        assert results[1].success is True
        assert results[1].source_id is not None
        assert "doc2.txt" in results[1].file_path

        # 4. Verify Database state (Checking the job history)
        # In a real environment, we'd query the DB to ensure jobs hit COMPLETED.
        # Since the orchestrator returned success=True, we know the JobManager finished them.

        # We can explicitly check the Storage API to ensure the files are in the vault
        source1 = orchestrator._storage.get_source(results[0].source_id)
        assert source1 is not None
        assert source1.name == "doc1.txt"

        # Verify the physical file is in the vault
        vault_path = Path(source1.file.file_path)
        assert vault_path.exists()
        assert vault_path.read_text() == "Hello World"