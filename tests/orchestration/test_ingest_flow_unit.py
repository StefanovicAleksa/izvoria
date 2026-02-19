# File: tests/orchestration/test_ingest_flow_unit.py

import pytest
import uuid
from unittest.mock import Mock, MagicMock

from app.orchestration.service.ingest_flow import IngestFlowOrchestrator
from app.orchestration.domain.entities import ScanJobResult, IngestJobResult
from app.orchestration.exceptions.workflow import ScannerInitializationError
from app.features.storage.domain.entities import SourceType


class TestIngestFlowUnit:
    @pytest.fixture
    def mock_job_manager(self):
        return Mock()

    @pytest.fixture
    def mock_scanner(self):
        return Mock()

    @pytest.fixture
    def mock_storage_api(self):
        return Mock()

    @pytest.fixture
    def orchestrator(self, mock_job_manager, mock_scanner, mock_storage_api):
        return IngestFlowOrchestrator(
            job_manager=mock_job_manager,
            scanner=mock_scanner,
            storage_api=mock_storage_api
        )

    def test_run_directory_scan_success(self, orchestrator, mock_job_manager):
        """Phase 1: Verifies directory scan creates a job and returns expected paths."""
        # 1. Mock the JobManager's run_job to simulate a successful scanner wrapper execution
        mock_job_manager.run_job.return_value = {
            "total_found": 2,
            "file_paths": ["/mock/file1.txt", "/mock/file2.mp4"]
        }

        # 2. Execute
        result = orchestrator.run_directory_scan("/mock/target")

        # 3. Assertions
        assert isinstance(result, ScanJobResult)
        assert result.total_found == 2
        assert len(result.file_paths) == 2
        assert "/mock/file1.txt" in result.file_paths
        mock_job_manager.run_job.assert_called_once()

    def test_run_directory_scan_failure(self, orchestrator, mock_job_manager):
        """Phase 1: Verifies scanner errors bubble up as ScannerInitializationError."""
        # Mock run_job to simulate a crash during scanner initialization
        mock_job_manager.run_job.side_effect = ScannerInitializationError("Scanner crashed")

        with pytest.raises(ScannerInitializationError):
            orchestrator.run_directory_scan("/mock/bad/path")

    def test_run_file_ingest_success(self, orchestrator, mock_job_manager):
        """Phase 2: Verifies single file ingest creates a job and returns success."""
        # 1. Mock the returned source from JobManager executing StorageAPI
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_source

        # 2. Execute
        result = orchestrator.run_file_ingest("/mock/file1.txt", SourceType.TEXT)

        # 3. Assertions
        assert isinstance(result, IngestJobResult)
        assert result.success is True
        assert result.source_id == mock_source.id
        assert result.file_path == "/mock/file1.txt"

    def test_run_file_ingest_failure(self, orchestrator, mock_job_manager):
        """Phase 2: Verifies ingest failures are caught and returned as failed items."""
        # 1. Simulate the storage API crashing (JobManager bubbles this up)
        mock_job_manager.run_job.side_effect = Exception("Disk full")

        # 2. Execute
        result = orchestrator.run_file_ingest("/mock/file1.txt", SourceType.TEXT)

        # 3. Assertions (Should NOT raise exception, but return success=False)
        assert isinstance(result, IngestJobResult)
        assert result.success is False
        assert result.source_id is None
        assert "Disk full" in result.error_message

    def test_process_full_batch(self, orchestrator):
        """End-to-End: Verifies sequential execution of scan and ingest loop."""
        # 1. Mock the internal methods we just tested above
        orchestrator.run_directory_scan = MagicMock(return_value=ScanJobResult(
            target_directory="/mock",
            total_found=2,
            file_paths=["/mock/a.txt", "/mock/b.txt"]
        ))

        # Simulate alternating success and failure for the items
        orchestrator.run_file_ingest = MagicMock(side_effect=[
            IngestJobResult(file_path="/mock/a.txt", success=True, source_id=uuid.uuid4()),
            IngestJobResult(file_path="/mock/b.txt", success=False, error_message="Failed")
        ])

        # 2. Execute
        results = orchestrator.process_full_batch("/mock")

        # 3. Assertions
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False

        # Verify run_file_ingest was called for each path found by the scan
        assert orchestrator.run_file_ingest.call_count == 2