# Canvas: test_domain_phonetic_correction_phase_unit_v7
# Path: tests/orchestration/phases/test_domain_phonetic_correction_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.orchestration.service.phases.domain_phonetic_correction_phase import DomainPhoneticCorrectionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import SentenceReconstructionJobResult, DomainPhoneticCorrectionJobResult
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript


class TestDomainPhoneticCorrectionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_correction_api(self): return Mock()

    @pytest.fixture
    def mock_sentence_repo(self): return Mock()

    @pytest.fixture
    def mock_domain_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_manifest(self):
        return ModelManifest(
            model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
            model_path="dummy_path",
            required_vram_mb=1000.0
        )

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_correction_api, mock_sentence_repo,
            mock_domain_repo, mock_async_service, mock_manifest, mock_adapter
    ):
        return DomainPhoneticCorrectionPhase(
            job_manager=mock_job_manager,
            correction_api=mock_correction_api,
            sentence_repo=mock_sentence_repo,
            domain_repo=mock_domain_repo,
            async_model_service=mock_async_service,
            manifest=mock_manifest,
            adapter=mock_adapter
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_manifest, mock_adapter):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(mock_manifest, mock_adapter)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_sentence_repo, mock_domain_repo, mock_correction_api):
        source_id = uuid.uuid4()
        upstream_results = [
            SentenceReconstructionJobResult(source_id=source_id, success=True, reconstructed_transcript_id=uuid.uuid4())
        ]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_sentence_repo.get_by_source_id.return_value = Mock(spec=ReconstructedTranscript)

        mock_glossary = Mock()
        mock_glossary.industry = "FINANCE"
        mock_glossary.sub_domains = ["OPTIONS_TRADING", "M_AND_A"]
        mock_domain_repo.get_glossary_by_source_id.return_value = mock_glossary

        mock_corrected_transcript = Mock(spec=DomainPhoneticCorrectedTranscript)
        mock_corrected_transcript.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_corrected_transcript

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.DOMAIN_PHONETIC_CORRECTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0
        assert isinstance(result.items[0], DomainPhoneticCorrectionJobResult)
        assert result.items[0].success is True

        # FIXED: Now asserts that the phase calls the synchronous release_model method on the API
        mock_correction_api.release_model.assert_called_once_with(mock_loaded_model)

    def test_execute_skips_failed_upstream(self, phase, mock_async_service):
        phase._preload_future = None

        upstream_results = [
            SentenceReconstructionJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]

        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert len(result.items) == 0

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future

        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Phonetic model" in result.errors[0]