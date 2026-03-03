# Canvas: test_intelligence_bridge_phase_unit_v1
# Path: tests/orchestration/service/phases/test_intelligence_bridge_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap
from app.features.storage.domain.entities import Source, File, SourceType
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import DiarizeJobResult
from app.orchestration.service.phases.intelligence_bridge_phase import IntelligenceBridgePhase


class TestIntelligenceBridgePhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_bridge_api(self): return Mock()

    @pytest.fixture
    def mock_storage(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.LLM, model_path="fake_phi3", required_vram_mb=3500)

    @pytest.fixture
    def phase(self, mock_job_manager, mock_bridge_api, mock_storage, mock_async_service, sample_manifest, mock_adapter):
        return IntelligenceBridgePhase(
            job_manager=mock_job_manager,
            bridge_api=mock_bridge_api,
            storage_api=mock_storage,
            async_model_service=mock_async_service,
            nlp_manifest=sample_manifest,
            nlp_adapter=mock_adapter
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        """Should trigger background load of the Phi-3.5 NLP model and store the future."""
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_success(self, phase, mock_job_manager, mock_storage, mock_async_service, mock_adapter):
        """Scenario: Batch processes perfectly, NLP model unloads at the end."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Setup Input (Requires Diarization phase output)
        diarize_results = [
            DiarizeJobResult(source_id=source_id, success=True, diarization_map_id=uuid.uuid4())
        ]

        # 2. Setup Async Model Future
        mock_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake_phi3")
        mock_future.set_result(fake_loaded_model)
        phase._preload_future = mock_future
        mock_async_service.resolve_preload.return_value = fake_loaded_model

        # 3. Setup Storage (Returns a Source with a nested Video File)
        mock_file = File(id=file_id, file_path="/vault/video.mp4", file_name="v", file_size_bytes=1000)
        mock_source = Source(id=source_id, name="v", source_type=SourceType.VIDEO, content_hash="123", file_id=file_id, file=mock_file)
        mock_storage.get_source.return_value = mock_source

        # 4. Setup JobManager to return a fake IntelligentSceneMap
        fake_map = IntelligentSceneMap(id=uuid.uuid4(), source_id=source_id, scenes=[])
        mock_job_manager.run_job.return_value = fake_map

        # Execute
        result = phase.execute(diarize_results)

        # Assertions
        assert result.phase == PipelinePhase.INTELLIGENCE_BRIDGE
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered
        mock_job_manager.run_job.assert_called_once()

        # Verify model was released after batch loop finished
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_skips_failed_diarizations(self, phase, mock_storage, mock_job_manager):
        """Scenario: Items that failed Diarization should not be processed by the Intelligence Bridge."""
        diarize_results = [
            DiarizeJobResult(source_id=uuid.uuid4(), success=False, error_message="Clustering Error")
        ]

        result = phase.execute(diarize_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_storage.get_source.assert_not_called()
        mock_job_manager.run_job.assert_not_called()