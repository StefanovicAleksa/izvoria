# Canvas: test_visual_cue_detection_phase_unit_v1
# Path: tests/orchestration/service/phases/test_visual_cue_detection_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.orchestration.service.phases.visual_cue_detection_phase import VisualCueDetectionPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import ContextualRetrievalJobResult, VisualCueDetectionJobResult
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.exceptions.base import ModelManagerError
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript


class TestVisualCueDetectionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_detection_api(self): return Mock()

    @pytest.fixture
    def mock_contextual_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_manifest(self):
        return ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path="dummy_slm_path",
            required_vram_mb=3500.0
        )

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_detection_api, mock_contextual_repo,
            mock_async_service, mock_manifest, mock_adapter
    ):
        return VisualCueDetectionPhase(
            job_manager=mock_job_manager,
            detection_api=mock_detection_api,
            contextual_repo=mock_contextual_repo,
            async_model_service=mock_async_service,
            manifest=mock_manifest,
            adapter=mock_adapter
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_manifest, mock_adapter):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(mock_manifest, mock_adapter)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_contextual_repo):
        source_id = uuid.uuid4()
        upstream_results = [
            ContextualRetrievalJobResult(source_id=source_id, success=True, contextual_transcript_id=uuid.uuid4())
        ]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_contextual_repo.get_by_source_id.return_value = Mock(spec=ContextualizedTranscript)

        mock_evaluated_transcript = Mock(spec=ContextualizedTranscript)
        mock_evaluated_transcript.id = uuid.uuid4()
        mock_job_manager.run_job.return_value = mock_evaluated_transcript

        result = phase.execute(upstream_results)

        assert result.phase == PipelinePhase.VISUAL_CUE_DETECTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0
        assert isinstance(result.items[0], VisualCueDetectionJobResult)
        assert result.items[0].success is True

        mock_async_service.trigger_unload.assert_called_once_with(mock_loaded_model, phase._adapter, flush=True)

    def test_execute_skips_failed_upstream(self, phase):
        phase._preload_future = None

        upstream_results = [
            ContextualRetrievalJobResult(source_id=uuid.uuid4(), success=False, error_message="Upstream fail")
        ]

        result = phase.execute(upstream_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        assert result.failed_count == 0

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future

        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([])

        assert result.status == PhaseStatus.FAILED
        assert "Critical failure preloading Visual Cue SLM" in result.errors[0]