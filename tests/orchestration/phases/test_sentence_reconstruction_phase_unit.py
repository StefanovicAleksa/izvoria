# Canvas: test_sentence_reconstruction_phase_unit_v2
# Path: tests/orchestration/service/phases/test_sentence_reconstruction_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

# CRITICAL FIX: Pointing to the new feature directory
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript
from app.features.transcription.domain.entities import Transcript

from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import DiarizeJobResult
from app.orchestration.service.phases.sentence_reconstruction_phase import SentenceReconstructionPhase


class TestSentenceReconstructionPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_reconstruction_api(self): return Mock()

    @pytest.fixture
    def mock_transcription_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_adapter(self): return Mock()

    @pytest.fixture
    def sample_manifest(self):
        return ModelManifest(
            model_type=ModelType.SENTENCE_RECONSTRUCTION,
            model_path="fake_punctuation",  # Changed from fake_sat
            required_vram_mb=500.0          # Changed from 750
        )

    @pytest.fixture
    def phase(self, mock_job_manager, mock_reconstruction_api, mock_transcription_api, mock_async_service, sample_manifest, mock_adapter):
        return SentenceReconstructionPhase(
            job_manager=mock_job_manager,
            reconstruction_api=mock_reconstruction_api,
            transcription_api=mock_transcription_api,
            async_model_service=mock_async_service,
            punctuation_manifest=sample_manifest,  # Renamed
            punctuation_adapter=mock_adapter       # Renamed
        )

    def test_preload_model(self, phase, mock_async_service, sample_manifest, mock_adapter):
        """Should trigger background load of the Punctuation model and store the future."""
        mock_future = concurrent.futures.Future()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once_with(sample_manifest, mock_adapter)

    def test_execute_success(self, phase, mock_job_manager, mock_transcription_api, mock_async_service, mock_adapter):
        """Scenario: Batch processes perfectly, Punctuation model unloads at the end."""
        source_id = uuid.uuid4()

        # 1. Setup Input (Requires Diarization phase output)
        diarize_results = [
            DiarizeJobResult(source_id=source_id, success=True, diarization_map_id=uuid.uuid4())
        ]

        # 2. Setup Async Model Future
        mock_future = concurrent.futures.Future()
        fake_loaded_model = LoadedModel(manifest=phase._manifest, model_instance="fake_punctuation")
        mock_future.set_result(fake_loaded_model)
        phase._preload_future = mock_future
        mock_async_service.resolve_preload.return_value = fake_loaded_model

        # 3. Setup Transcription API (Returns a base Transcript)
        mock_transcript = Transcript(id=uuid.uuid4(), source_id=source_id, segments=[])
        mock_transcription_api.get_transcript.return_value = mock_transcript

        # 4. Setup JobManager to return a fake ReconstructedTranscript
        fake_reconstructed = ReconstructedTranscript(id=uuid.uuid4(), source_id=source_id, sentences=[])
        mock_job_manager.run_job.return_value = fake_reconstructed

        # Execute
        result = phase.execute(diarize_results)

        # Assertions
        assert result.phase == PipelinePhase.SENTENCE_RECONSTRUCTION
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        # Verify Job tracking triggered
        mock_job_manager.run_job.assert_called_once()

        # Verify model was released after batch loop finished
        mock_async_service.trigger_unload.assert_called_once_with(fake_loaded_model, mock_adapter, flush=True)

    def test_execute_skips_failed_diarizations(self, phase, mock_transcription_api, mock_job_manager):
        """Scenario: Items that failed Diarization should not be processed by Sentence Reconstruction."""
        diarize_results = [
            DiarizeJobResult(source_id=uuid.uuid4(), success=False, error_message="Clustering Error")
        ]

        result = phase.execute(diarize_results)

        assert result.status == PhaseStatus.COMPLETED
        assert result.total_processed == 1
        assert result.successful_count == 0
        assert result.failed_count == 0
        mock_transcription_api.get_transcript.assert_not_called()
        mock_job_manager.run_job.assert_not_called()