# filename: test_ingest_phase_unit_v1.py
import pytest
import uuid
from unittest.mock import Mock, MagicMock

from app.orchestration.service.phases.ingest_phase import IngestPhase
from app.orchestration.domain.enums import PhaseStatus
from app.features.storage.domain.entities import SourceType


class TestIngestPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self):
        return Mock()

    @pytest.fixture
    def mock_storage_api(self):
        return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_storage_api):
        return IngestPhase(job_manager=mock_job_manager, storage_api=mock_storage_api)

    def test_execute_perfect_batch(self, phase, mock_job_manager):
        """Verifies that if all files succeed, the batch is COMPLETED."""
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_source

        files = ["/a.txt", "/b.txt"]
        result = phase.execute(files, SourceType.TEXT)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 2
        assert result.failed_count == 0
        assert len(result.items) == 2
        assert result.items[0].success is True

    def test_execute_partial_success(self, phase, mock_job_manager):
        """Verifies the loop continues even if one file fails, marking batch PARTIAL_SUCCESS."""
        mock_source = MagicMock()
        mock_source.id = uuid.uuid4()

        # First call succeeds, second call crashes
        mock_job_manager.run_job.side_effect = [
            mock_source,
            Exception("Corrupted file")
        ]

        result = phase.execute(["/good.txt", "/bad.txt"], SourceType.TEXT)

        assert result.status == PhaseStatus.PARTIAL_SUCCESS
        assert result.successful_count == 1
        assert result.failed_count == 1

        # Verify specific item results
        assert result.items[0].success is True
        assert result.items[1].success is False
        assert "Corrupted file" in result.items[1].error_message
        assert len(result.errors) == 1

    def test_execute_total_failure(self, phase, mock_job_manager):
        """Verifies that if all files fail, the batch is marked FAILED."""
        mock_job_manager.run_job.side_effect = Exception("Storage Drive Full")

        result = phase.execute(["/a.txt", "/b.txt"], SourceType.TEXT)

        assert result.status == PhaseStatus.FAILED
        assert result.successful_count == 0
        assert result.failed_count == 2