# Canvas: test_transcribe_phase_unit_v2
# Path: tests/orchestration/service/phases/test_transcribe_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.domain.entities import Transcript
from app.features.storage.domain.entities import Source, File, SourceType
from app.features.domain_adaptation.domain.entities import DomainGlossary
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import VadJobResult
from app.orchestration.service.phases.transcribe_phase import TranscribePhase


class TestTranscribePhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_transcription_api(self): return Mock()

    @pytest.fixture
    def mock_storage(self): return Mock()

    @pytest.fixture
    def mock_domain_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(model_type=ModelType.TRANSCRIPTION, model_path="fake_whisper", required_vram_mb=100)

    @pytest.fixture
    def phase(self, mock_job_manager, mock_transcription_api, mock_storage, mock_domain_repo, mock_async_service, sample_manifest, mock_adapter):
        return TranscribePhase(
            job_manager=mock_job_manager,
            transcription_api=mock_transcription_api,
            storage_api=mock_storage,
            domain_repo=mock_domain_repo,
            async_model_service=mock_async_service,
            transcribe_manifest=sample_manifest,
            transcribe_adapter=mock_adapter
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        """Should trigger background load and store the future."""
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_success_with_domain_hints(self, phase, mock_job_manager, mock_storage, mock_domain_repo, mock_async_service, mock_adapter, mock_transcription_api):
        """Scenario: Batch processes perfectly, fetches domain hints, unloads at the end."""
        source_id = uuid.uuid4()
        file_id = uuid.uuid4()

        # 1. Setup Input
        vad_results = [
            VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())
        ]

        # 2. Setup Async Model Future
        mock_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake_whisper")
        mock_future.set_result(fake_loaded_model)
        phase._preload_future = mock_future
        mock_async_service.resolve_preload.return_value = fake_loaded_model

        # 3. Setup Storage (Returns a Source with a nested File)
        mock_file = File(id=file_id, file_path="/vault/audio.wav", file_name="a", file_size_bytes=1)
        mock_source = Source(id=source_id, name="a", source_type=SourceType.VIDEO, content_hash="123", file_id=file_id, file=mock_file)
        mock_storage.get_source.return_value = mock_source

        # 4. Setup Domain Repo to return hints
        mock_glossary = DomainGlossary(id=uuid.uuid4(), source_id=source_id, industry="FINANCE", sub_domain="GENERAL", hotword_string="EBITDA:15.0")
        mock_domain_repo.get_glossary_by_source_id.return_value = mock_glossary

        # 5. Setup JobManager to return a fake Transcript
        fake_transcript = Transcript(id=uuid.uuid4(), source_id=source_id, segments=[])
        mock_job_manager.run_job.return_value = fake_transcript

        # Execute
        result = phase.execute(vad_results)

        # Assertions
        assert result.phase == PipelinePhase.TRANSCRIBE
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered and passed the domain hints exactly as expected
        mock_job_manager.run_job.assert_called_once_with(
            mock_job_manager.run_job.call_args[0][0], # request
            mock_transcription_api.process_audio,
            source_id=source_id,
            audio_path="/vault/audio.wav",
            language=settings.APPLIANCE_LANGUAGE,
            hotwords="EBITDA:15.0",
            release_after=False
        )

        # Verify model was released after batch loop finished
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_skips_failed_vads(self, phase, mock_storage, mock_job_manager):
        """Scenario: Items that failed Phase 4 should not be processed by Phase 5."""
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