# Canvas: test_hallucination_scrubber_phase_unit_v1
# Path: tests/orchestration/phases/test_hallucination_scrubber_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.hallucination_scrubber_phase import HallucinationScrubberPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import DiarizeJobResult
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.legacy_features.hallucination_scrubber.domain.entities import HallucinationScrubbedTranscript


class TestHallucinationScrubberPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_scrubber_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_manifest(self):
        return ModelManifest(
            model_type=ModelType.HALLUCINATION_SCRUBBER,
            model_path="dummy_path",
            required_vram_mb=800.0
        )

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_scrubber_api, mock_async_service,
            mock_manifest, mock_adapter
    ):
        return HallucinationScrubberPhase(
            job_manager=mock_job_manager,
            scrubber_api=mock_scrubber_api,
            async_model_service=mock_async_service,
            scrubber_manifest=mock_manifest,
            scrubber_adapter=mock_adapter
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_manifest, mock_adapter):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(mock_manifest, mock_adapter)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_scrubber_api, mock_adapter):
        source_id = uuid.uuid4()
        upstream_results = [
            DiarizeJobResult(source_id=source_id, success=True, diarization_map_id=uuid.uuid4())
        ]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_scrubbed_transcript = Mock(spec=HallucinationScrubbedTranscript)
        mock_scrubbed_transcript.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_scrubbed_transcript

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.HALLUCINATION_SCRUBBER
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Ensures the job logic passed release_after=False to hold VRAM during the loop
        mock_job_manager.run_job.assert_called_once_with(
            ANY,
            mock_scrubber_api.execute,
            source_id=source_id,
            release_after=False
        )

        mock_async_service.trigger_unload.assert_called_once_with(mock_loaded_model, mock_adapter, flush=True)

    def test_execute_skips_failed_upstream(self, phase):
        phase._preload_future = None
        upstream_results = [
            DiarizeJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]
        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        assert result.failed_count == 0

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future

        mock_async_service.resolve_preload.side_effect = ModelManagerError("OOM Exception")

        result = phase.execute([])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Scrubber model" in result.errors[0]