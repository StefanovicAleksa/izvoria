# Canvas: test_rag_indexing_phase_unit_v2
# Path: tests/orchestration/phases/test_rag_indexing_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock

from app.orchestration.service.phases.rag_indexing_phase import RagIndexingPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import ContextUnificationJobResult, RagIndexingJobResult
from app.core.model_manager.domain.entities import LoadedModel
from app.features.context_unification.domain.entities import UnifiedTranscript
from app.features.rag_storage.domain.entities import IngestionResult, IndexingTelemetry


class TestRagIndexingPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_rag_storage_service(self): return Mock()

    @pytest.fixture
    def mock_unified_repo(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(
            self, mock_job_manager, mock_rag_storage_service, mock_unified_repo,
            mock_async_service, mock_loader
    ):
        return RagIndexingPhase(
            job_manager=mock_job_manager,
            rag_storage_service=mock_rag_storage_service,
            unified_repo=mock_unified_repo,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service):
        """Verifies that preloading correctly registers a future in the phase state."""
        mock_future = Mock()
        mock_async_service.trigger_preload.return_value = mock_future

        phase.preload_model()

        assert phase._preload_future == mock_future
        mock_async_service.trigger_preload.assert_called_once()

    def test_execute_success_flow(self, phase, mock_job_manager, mock_async_service, mock_unified_repo):
        """Verifies the happy path processes correctly, registers hybrid indexing metrics, and clears VRAM."""
        source_id = uuid.uuid4()
        prev_results = [ContextUnificationJobResult(source_id=source_id, success=True)]

        # Mock Preload and Resolution
        mock_future = Mock()
        mock_async_service.trigger_preload.return_value = mock_future

        mock_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_model

        # Mock Transcript Retrieval
        mock_unified_repo.get_by_source_id.return_value = Mock(spec=UnifiedTranscript)

        # Mock Job Manager execution (Now atomic: dense and sparse counts should match)
        mock_ingestion_result = IngestionResult(
            success=True,
            telemetry=IndexingTelemetry(
                source_id=source_id,
                total_chunks_processed=5,
                dense_vectors_saved=5,
                sparse_vectors_saved=5,
                processing_time_seconds=1.0,
                embedding_model_used="test"
            )
        )
        mock_job_manager.run_job.return_value = mock_ingestion_result

        # Trigger Preload
        phase.preload_model()

        # Execute Phase
        result = phase.execute(prev_results)

        # Assertions
        assert result.phase == PipelinePhase.RAG_INDEXING
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        assert isinstance(result.items[0], RagIndexingJobResult)
        assert result.items[0].dense_vectors_saved == 5
        assert result.items[0].sparse_vectors_saved == 5  # FTS Index saved synchronously

        # Verify GPU lifecycle is handled
        mock_async_service.resolve_preload.assert_called_once_with(mock_future)
        mock_async_service.trigger_unload.assert_called_once_with(
            loaded_model=mock_model,
            loader=phase._loader,
            flush=True
        )

    def test_execute_skips_failed_upstream_items(self, phase, mock_job_manager):
        """Verifies that the phase ignores transcripts that failed in Phase 15."""
        source_id = uuid.uuid4()
        prev_results = [
            ContextUnificationJobResult(source_id=source_id, success=False, error_message="Unification failed")]

        result = phase.execute(prev_results)

        assert result.successful_count == 0
        mock_job_manager.run_job.assert_not_called()

    def test_execute_fails_if_preload_fails(self, phase, mock_async_service):
        """Verifies failure bubbling if the embedding model cannot load into VRAM."""
        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future

        from app.core.model_manager.exceptions.base import ModelManagerError
        mock_async_service.resolve_preload.side_effect = ModelManagerError("Out of VRAM")

        result = phase.execute([])
        assert result.status == PhaseStatus.FAILED

    def test_execute_clears_resolved_preload_future(self, phase, mock_async_service, mock_unified_repo):
        phase._preload_future = Mock()
        mock_async_service.resolve_preload.return_value = Mock(spec=LoadedModel)

        phase.execute([])

        assert phase._preload_future is None

    def test_discard_preload_hands_unresolved_future_to_async_service(self, phase, mock_async_service):
        mock_future = Mock()
        phase._preload_future = mock_future

        phase.discard_preload()

        mock_async_service.discard_preload.assert_called_once_with(mock_future, phase._loader)
        assert phase._preload_future is None

    def test_discard_preload_is_noop_without_future(self, phase, mock_async_service):
        phase.discard_preload()

        mock_async_service.discard_preload.assert_not_called()
