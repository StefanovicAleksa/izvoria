# filename: test_scan_phase_unit_v1.py
import pytest
from unittest.mock import Mock

from app.orchestration.service.phases.scan_phase import ScanPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult
from app.orchestration.exceptions.phase import ScannerInitializationError


class TestScanPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self):
        return Mock()

    @pytest.fixture
    def mock_scanner(self):
        return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_scanner):
        return ScanPhase(job_manager=mock_job_manager, scanner=mock_scanner)

    def test_execute_happy_path(self, phase, mock_job_manager):
        """Verifies successful scan translates to a COMPLETED BatchResult."""
        # Mock JobManager to simulate successful wrapper execution
        mock_job_manager.run_job.return_value = {
            "total_found": 2,
            "file_paths": ["/vault/a.mp4", "/vault/b.txt"]
        }

        result = phase.execute("/vault")

        assert isinstance(result, PhaseBatchResult)
        assert result.phase == PipelinePhase.SCAN
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 2
        assert len(result.items) == 2
        mock_job_manager.run_job.assert_called_once()

    def test_execute_empty_directory(self, phase, mock_job_manager):
        """Verifies finding zero files returns PARTIAL_SUCCESS so the pipeline knows it can stop gracefully."""
        mock_job_manager.run_job.return_value = {
            "total_found": 0,
            "file_paths": []
        }

        result = phase.execute("/empty")

        assert result.status == PhaseStatus.PARTIAL_SUCCESS
        assert result.successful_count == 0

    def test_execute_scanner_initialization_error(self, phase, mock_job_manager):
        """Verifies critical init errors (like invalid path) bubble up directly."""
        mock_job_manager.run_job.side_effect = ScannerInitializationError("Path not found")

        with pytest.raises(ScannerInitializationError):
            phase.execute("/ghost/dir")

    def test_execute_system_crash_caught(self, phase, mock_job_manager):
        """Verifies generic JobManager crashes are caught and return a FAILED batch status."""
        mock_job_manager.run_job.side_effect = Exception("Job Database Offline")

        result = phase.execute("/vault")

        assert result.status == PhaseStatus.FAILED
        assert result.failed_count == 1
        assert "Job Database Offline" in result.errors[0]