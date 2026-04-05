# Canvas: test_hallucination_correction_phase_unit_v1
# Path: tests/orchestration/service/phases/test_hallucination_correction_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.orchestration.service.phases.hallucination_correction_phase import HallucinationCorrectionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import TranscribeJobResult
from app.features.transcription.domain.entities import Transcript


class TestHallucinationCorrectionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_correction_api(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_correction_api):
        return HallucinationCorrectionPhase(
            job_manager=mock_job_manager,
            correction_api=mock_correction_api
        )

    def test_execute_success(self, phase, mock_job_manager, mock_correction_api):
        """Scenario: Batch processes perfectly and scrubs hallucinations."""
        source_id = uuid.uuid4()

        # 1. Setup Input
        transcribe_results = [
            TranscribeJobResult(source_id=source_id, success=True, transcript_id=uuid.uuid4())
        ]

        # 2. Setup JobManager to return a fake cleaned Transcript and telemetry tuple
        fake_cleaned_transcript = Transcript(id=uuid.uuid4(), source_id=source_id, segments=[])
        fake_telemetry = Mock()
        mock_job_manager.run_job.return_value = (fake_cleaned_transcript, fake_telemetry)

        # Execute
        result = phase.execute(transcribe_results)

        # Assertions
        assert result.phase == PipelinePhase.HALLUCINATION_CORRECTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered
        mock_job_manager.run_job.assert_called_once()
        assert result.items[0].transcript_id == fake_cleaned_transcript.id

    def test_execute_skips_failed_transcriptions(self, phase, mock_job_manager):
        """Scenario: Items that failed Phase 5 should not be processed by Phase 5.5."""
        transcribe_results = [
            TranscribeJobResult(source_id=uuid.uuid4(), success=False, error_message="CUDA OOM")
        ]

        result = phase.execute(transcribe_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_job_manager.run_job.assert_not_called()