# Canvas: test_diarize_phase_unit_v2
# Path: tests/orchestration/phases/test_diarize_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.audio_extractor.exceptions.base import AudioExtractorError
from app.features.diarization.domain.entities import DiarizationMap
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import TranscribeJobResult
from app.orchestration.service.phases.diarize_phase import DiarizePhase


class TestDiarizePhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_diarization_api(self): return Mock()

    @pytest.fixture
    def mock_audio_extractor(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.DIARIZATION, model_path="fake_titanet", required_vram_mb=100)

    @pytest.fixture
    def phase(self, mock_job_manager, mock_diarization_api, mock_audio_extractor, mock_async_service,
              sample_manifest, mock_adapter):
        return DiarizePhase(
            job_manager=mock_job_manager,
            diarization_api=mock_diarization_api,
            audio_extractor=mock_audio_extractor,
            async_model_service=mock_async_service,
            diarize_manifest=sample_manifest,
            diarize_adapter=mock_adapter
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_success_diarizes_extracted_wav(self, phase, mock_job_manager, mock_diarization_api,
                                                    mock_audio_extractor, mock_async_service, mock_adapter):
        source_id = uuid.uuid4()
        transcribe_results = [TranscribeJobResult(source_id=source_id, success=True, transcript_id=uuid.uuid4())]

        phase._preload_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake_titanet")
        mock_async_service.resolve_preload.return_value = fake_loaded_model
        mock_audio_extractor.get_extracted_audio_path.return_value = "/vault/extracted.wav"
        mock_job_manager.run_job.return_value = DiarizationMap(id=uuid.uuid4(), source_id=source_id, segments=[])

        result = phase.execute(transcribe_results)

        assert result.phase == PipelinePhase.DIARIZE
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        mock_job_manager.run_job.assert_called_once_with(
            mock_job_manager.run_job.call_args[0][0],
            mock_diarization_api.process_audio,
            source_id=source_id,
            audio_path="/vault/extracted.wav",
            release_after=False
        )
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_fails_item_when_extracted_audio_is_missing(self, phase, mock_job_manager, mock_audio_extractor):
        mock_audio_extractor.get_extracted_audio_path.side_effect = AudioExtractorError("No extracted audio")

        result = phase.execute([TranscribeJobResult(source_id=uuid.uuid4(), success=True, transcript_id=uuid.uuid4())])

        assert result.status == PhaseStatus.FAILED
        mock_job_manager.run_job.assert_not_called()

    def test_execute_skips_failed_transcripts(self, phase, mock_audio_extractor, mock_job_manager):
        result = phase.execute([TranscribeJobResult(source_id=uuid.uuid4(), success=False, error_message="Model Load Error")])

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_audio_extractor.get_extracted_audio_path.assert_not_called()
        mock_job_manager.run_job.assert_not_called()
