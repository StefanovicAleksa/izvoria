# filename: test_extract_audio_phase_unit_v1.py
import pytest
import uuid
from unittest.mock import Mock, MagicMock

from app.orchestration.service.phases.extract_audio_phase import ExtractAudioPhase
from app.orchestration.domain.enums import PhaseStatus


class TestExtractAudioPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self):
        return Mock()

    @pytest.fixture
    def mock_audio_extractor(self):
        return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_audio_extractor):
        return ExtractAudioPhase(job_manager=mock_job_manager, audio_extractor=mock_audio_extractor)

    def test_execute_happy_path(self, phase, mock_job_manager):
        """Verifies processing a list of Video Source IDs successfully."""
        mock_extraction = MagicMock()
        mock_extraction.extracted_file_id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_extraction

        source_ids = [uuid.uuid4(), uuid.uuid4()]
        result = phase.execute(source_ids)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 2
        assert result.items[0].extracted_file_id == mock_extraction.extracted_file_id
        assert mock_job_manager.run_job.call_count == 2

    def test_execute_captures_ffmpeg_crash(self, phase, mock_job_manager):
        """Verifies an FFmpeg crash on one video doesn't stop the rest from processing."""
        mock_extraction = MagicMock()
        mock_extraction.extracted_file_id = uuid.uuid4()

        # First video crashes, second succeeds
        mock_job_manager.run_job.side_effect = [
            Exception("FFmpeg timeout"),
            mock_extraction
        ]

        result = phase.execute([uuid.uuid4(), uuid.uuid4()])

        assert result.status == PhaseStatus.PARTIAL_SUCCESS
        assert result.failed_count == 1
        assert result.successful_count == 1
        assert "FFmpeg timeout" in result.errors[0]