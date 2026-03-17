# Canvas: test_domain_adaptation_phase_unit_v1
# Path: tests/orchestration/service/phases/test_domain_adaptation_phase_unit.py

import pytest
import uuid
from unittest.mock import Mock

from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.features.storage.domain.entities import Source, File, SourceType
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
    def mock_storage(self): return Mock()

    @pytest.fixture
    def mock_vad_repo(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_adaptation_api, mock_storage, mock_vad_repo):
        return DomainAdaptationPhase(
            job_manager=mock_job_manager,
            adaptation_api=mock_adaptation_api,
            storage_api=mock_storage,
            vad_repo=mock_vad_repo
        )

    def test_execute_success(self, phase, mock_job_manager, mock_storage, mock_vad_repo):
        """Scenario: Batch processes perfectly and returns glossary IDs."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Setup Input
        vad_results = [
            VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())
        ]

        # 2. Setup Storage (Returns a Source with a nested File)
        mock_file = File(id=file_id, file_path="/vault/audio.wav", file_name="a", file_size_bytes=1)
        mock_source = Source(id=source_id, name="a", source_type=SourceType.VIDEO, content_hash="123", file_id=file_id, file=mock_file)
        mock_storage.get_source.return_value = mock_source

        # 3. Setup VAD Repository
        mock_vad_map = VadMap(id=uuid.uuid4(), source_id=source_id, segments=[])
        mock_vad_repo.get_map_by_source_id.return_value = mock_vad_map

        # 4. Setup JobManager to return a fake DomainGlossary
        fake_glossary = DomainGlossary(id=uuid.uuid4(), source_id=source_id, industry="FINANCE", sub_domain="OPTIONS_TRADING", hotword_string="strike:15.0")
        mock_job_manager.run_job.return_value = fake_glossary

        # Execute
        result = phase.execute(vad_results)

        # Assertions
        assert result.phase == PipelinePhase.DOMAIN_ADAPTATION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered
        mock_job_manager.run_job.assert_called_once()
        assert result.items[0].glossary_id == fake_glossary.id

    def test_execute_skips_failed_vads(self, phase, mock_storage, mock_job_manager):
        """Scenario: Items that failed Phase 4 should not be processed by Phase 4.5."""
        vad_results = [
            VadJobResult(source_id=uuid.uuid4(), success=False, error_message="CUDA OOM")
        ]

        result = phase.execute(vad_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_storage.get_source.assert_not_called()
        mock_job_manager.run_job.assert_not_called()