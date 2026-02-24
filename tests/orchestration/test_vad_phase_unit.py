# filename: test_vad_phase_unit_v1.py
# path: tests/orchestration/service/phases/test_vad_phase_unit.py
import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.domain.entities import VadMap
from app.features.storage.domain.entities import File
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import AudioExtractionJobResult
from app.orchestration.service.phases.vad_phase import VadPhase


class TestVadPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self):
        return Mock()

    @pytest.fixture
    def mock_vad_api(self):
        return Mock()

    @pytest.fixture
    def mock_storage(self):
        return Mock()

    @pytest.fixture
    def mock_async_service(self):
        return Mock()

    @pytest.fixture
    def mock_adapter(self):
        return Mock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.VAD, model_path="fake.nemo", required_vram_mb=100)

    @pytest.fixture
    def phase(self, mock_job_manager, mock_vad_api, mock_storage, mock_async_service, sample_manifest, mock_adapter):
        return VadPhase(
            job_manager=mock_job_manager,
            vad_api=mock_vad_api,
            storage_api=mock_storage,
            async_model_service=mock_async_service,
            vad_manifest=sample_manifest,
            vad_adapter=mock_adapter
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        """Should trigger background load and store the future."""
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_success(self, phase, mock_job_manager, mock_storage, mock_async_service, mock_adapter):
        """Scenario: Batch processes perfectly, model unloads at the end."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Setup Input
        extraction_results = [
            AudioExtractionJobResult(source_id=source_id, success=True, extracted_file_id=file_id)
        ]

        # 2. Setup Async Model Future
        mock_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake")
        mock_future.set_result(fake_loaded_model)
        phase._preload_future = mock_future
        mock_async_service.resolve_preload.return_value = fake_loaded_model

        # 3. Setup Storage
        mock_storage.get_file.return_value = File(id=file_id, file_path="/vault/audio.wav", file_name="a",
                                                  file_size_bytes=1)

        # 4. Setup JobManager to return a fake VadMap
        fake_map = VadMap(id=uuid.uuid4(), source_id=source_id, segments=[])
        mock_job_manager.run_job.return_value = fake_map

        # Execute
        result = phase.execute(extraction_results)

        # Assertions
        assert result.phase == PipelinePhase.VAD
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered
        mock_job_manager.run_job.assert_called_once()

        # Verify model was released after batch loop finished
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_skips_failed_extractions(self, phase, mock_storage, mock_job_manager):
        """Scenario: Items that failed Phase 3 should not be processed by Phase 4."""
        extraction_results = [
            AudioExtractionJobResult(source_id=uuid.uuid4(), success=False, error_message="FFmpeg crashed")
        ]

        result = phase.execute(extraction_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_storage.get_file.assert_not_called()
        mock_job_manager.run_job.assert_not_called()