# Canvas: test_domain_adaptation_phase_unit_v3
# Path: tests/orchestration/phases/test_domain_adaptation_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.features.audio_extractor.exceptions.base import AudioExtractorError
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.vad.domain.entities import VadMap
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import VadJobResult
from app.orchestration.service.phases.domain_adaptation_phase import DomainAdaptationPhase


class TestDomainAdaptationPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_adaptation_api(self): return Mock()

    @pytest.fixture
    def mock_audio_extractor(self): return Mock()

    @pytest.fixture
    def mock_vad_repo(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_adaptation_api, mock_audio_extractor, mock_vad_repo):
        return DomainAdaptationPhase(
            job_manager=mock_job_manager,
            adaptation_api=mock_adaptation_api,
            audio_extractor=mock_audio_extractor,
            vad_repo=mock_vad_repo
        )

    def test_execute_success_uses_extracted_wav(self, phase, mock_job_manager, mock_adaptation_api,
                                                mock_audio_extractor, mock_vad_repo):
        source_id = uuid.uuid4()
        vad_results = [VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())]

        mock_audio_extractor.get_extracted_audio_path.return_value = "/vault/extracted.wav"
        mock_vad_map = VadMap(id=uuid.uuid4(), source_id=source_id, segments=[])
        mock_vad_repo.get_map_by_source_id.return_value = mock_vad_map

        fake_glossary = DomainGlossary(
            id=uuid.uuid4(),
            source_id=source_id,
            industry="FINANCE",
            sub_domains=["OPTIONS_TRADING"],
            hotword_string="strike, theta",
            initial_prompt_string="This recording discusses Options Trading, including terms such as strike, theta."
        )
        mock_job_manager.run_job.return_value = fake_glossary

        result = phase.execute(vad_results)

        assert result.phase == PipelinePhase.DOMAIN_ADAPTATION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.items[0].glossary_id == fake_glossary.id

        request = mock_job_manager.run_job.call_args[0][0]
        assert request.input_params == {"audio_path": "/vault/extracted.wav"}
        mock_job_manager.run_job.assert_called_once_with(
            request,
            mock_adaptation_api.process_domain_adaptation,
            source_id=source_id,
            audio_path="/vault/extracted.wav",
            vad_map=mock_vad_map
        )

    def test_execute_fails_item_when_extracted_audio_is_missing(self, phase, mock_job_manager, mock_audio_extractor):
        mock_audio_extractor.get_extracted_audio_path.side_effect = AudioExtractorError("No extracted audio")

        result = phase.execute([VadJobResult(source_id=uuid.uuid4(), success=True, vad_map_id=uuid.uuid4())])

        assert result.status == PhaseStatus.FAILED
        assert "No extracted audio" in result.errors[0]
        mock_job_manager.run_job.assert_not_called()

    def test_execute_skips_failed_vads(self, phase, mock_audio_extractor, mock_job_manager):
        result = phase.execute([VadJobResult(source_id=uuid.uuid4(), success=False, error_message="CUDA OOM")])

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_audio_extractor.get_extracted_audio_path.assert_not_called()
        mock_job_manager.run_job.assert_not_called()
