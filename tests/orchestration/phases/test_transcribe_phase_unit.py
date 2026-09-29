# Canvas: test_transcribe_phase_unit_v4
# Path: tests/orchestration/phases/test_transcribe_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.audio_extractor.exceptions.base import AudioExtractorError
from app.features.transcription.domain.entities import Transcript
from app.features.transcription.exceptions import LanguageMismatchError, TranscriptionError
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
    def mock_audio_extractor(self): return Mock()

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
    def mock_storage_api(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_transcription_api, mock_audio_extractor, mock_domain_repo,
              mock_async_service, sample_manifest, mock_adapter, mock_storage_api):
        return TranscribePhase(
            job_manager=mock_job_manager,
            transcription_api=mock_transcription_api,
            audio_extractor=mock_audio_extractor,
            domain_repo=mock_domain_repo,
            async_model_service=mock_async_service,
            transcribe_manifest=sample_manifest,
            transcribe_adapter=mock_adapter,
            storage_api=mock_storage_api
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        """Should trigger background load and store the future."""
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_transcribes_extracted_wav_with_domain_hints(
            self, phase, mock_job_manager, mock_audio_extractor, mock_domain_repo, mock_async_service,
            mock_adapter, mock_transcription_api
    ):
        source_id = uuid.uuid4()
        vad_results = [VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())]

        mock_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake_whisper")
        phase._preload_future = mock_future
        mock_async_service.resolve_preload.return_value = fake_loaded_model

        mock_audio_extractor.get_extracted_audio_path.return_value = "/vault/extracted_16k_mono.wav"

        mock_domain_repo.get_glossary_by_source_id.return_value = DomainGlossary(
            id=uuid.uuid4(),
            source_id=source_id,
            industry="FINANCE",
            sub_domains=["GENERAL"],
            hotword_string="EBITDA, CapEx",
            initial_prompt_string="This recording discusses General."
        )

        mock_job_manager.run_job.return_value = Transcript(id=uuid.uuid4(), source_id=source_id, segments=[])

        result = phase.execute(vad_results)

        assert result.phase == PipelinePhase.TRANSCRIBE
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1

        mock_audio_extractor.get_extracted_audio_path.assert_called_once_with(source_id)
        mock_job_manager.run_job.assert_called_once_with(
            mock_job_manager.run_job.call_args[0][0],
            mock_transcription_api.process_audio,
            source_id=source_id,
            audio_path="/vault/extracted_16k_mono.wav",
            language=settings.APPLIANCE_LANGUAGE,
            hotwords="EBITDA, CapEx",
            initial_prompt="This recording discusses General.",
            release_after=False
        )
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_fails_item_when_extracted_audio_is_missing(self, phase, mock_job_manager, mock_audio_extractor):
        source_id = uuid.uuid4()
        mock_audio_extractor.get_extracted_audio_path.side_effect = AudioExtractorError("No extracted audio")

        result = phase.execute([VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())])

        assert result.status == PhaseStatus.FAILED
        assert result.items[0].success is False
        assert "No extracted audio" in result.items[0].error_message
        mock_job_manager.run_job.assert_not_called()

    def test_execute_skips_failed_vads(self, phase, mock_audio_extractor, mock_job_manager):
        """Scenario: Items that failed Phase 4 should not be processed by Phase 5."""
        result = phase.execute([VadJobResult(source_id=uuid.uuid4(), success=False, error_message="CUDA OOM")])

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_audio_extractor.get_extracted_audio_path.assert_not_called()
        mock_job_manager.run_job.assert_not_called()

    def test_a_failed_language_check_forgets_the_sources_language_and_glossary(
            self, phase, mock_job_manager, mock_audio_extractor, mock_domain_repo, mock_storage_api
    ):
        source_id = uuid.uuid4()
        mock_audio_extractor.get_extracted_audio_path.return_value = "/vault/a.wav"
        mock_job_manager.run_job.side_effect = LanguageMismatchError("spoken in 'en'")

        result = phase.execute([VadJobResult(source_id=source_id, success=True, vad_map_id=uuid.uuid4())])

        assert result.status == PhaseStatus.FAILED
        mock_domain_repo.delete_glossary_by_source_id.assert_called_once_with(source_id)
        mock_storage_api.clear_language.assert_called_once_with(source_id)

    def test_any_other_failure_keeps_the_sources_language(
            self, phase, mock_job_manager, mock_audio_extractor, mock_domain_repo, mock_storage_api
    ):
        mock_audio_extractor.get_extracted_audio_path.return_value = "/vault/a.wav"
        mock_job_manager.run_job.side_effect = TranscriptionError("CUDA OOM")

        phase.execute([VadJobResult(source_id=uuid.uuid4(), success=True, vad_map_id=uuid.uuid4())])

        mock_domain_repo.delete_glossary_by_source_id.assert_not_called()
        mock_storage_api.clear_language.assert_not_called()
