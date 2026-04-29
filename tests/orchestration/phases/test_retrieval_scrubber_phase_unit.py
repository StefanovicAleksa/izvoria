# Canvas: test_retrieval_scrubber_phase_unit_v1
# Path: tests/orchestration/phases/test_retrieval_scrubber_phase_unit.py

import pytest
import uuid
import concurrent.futures
from unittest.mock import Mock, ANY

from app.orchestration.service.phases.retrieval_scrubber_phase import RetrievalScrubberPhase
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import RagRetrievalJobResult, RetrievalScrubbingJobResult
from app.features.rag_retrieval.domain.entities import SearchResult, RetrievedChunk
from app.core.model_manager.domain.entities import LoadedModel


class TestRetrievalScrubberPhaseUnit:
    @pytest.fixture
    def mock_job_manager(self): return Mock()

    @pytest.fixture
    def mock_scrubber_api(self): return Mock()

    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_loader(self): return Mock()

    @pytest.fixture
    def phase(self, mock_job_manager, mock_scrubber_api, mock_async_service, mock_loader):
        return RetrievalScrubberPhase(
            job_manager=mock_job_manager,
            scrubber_api=mock_scrubber_api,
            async_model_service=mock_async_service,
            loader=mock_loader
        )

    def test_preload_model_triggers_async_service(self, phase, mock_async_service, mock_loader):
        phase.preload_model()
        mock_async_service.trigger_preload.assert_called_once_with(manifest=ANY, loader=mock_loader)

    def test_execute_happy_path(self, phase, mock_async_service, mock_job_manager, mock_loader):
        search_id = uuid.uuid4()
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Test", start_sec=0,
                               end_sec=1, relevance_score=1.0)

        job_result = RagRetrievalJobResult(search_id=search_id, success=True, retrieved_chunks_count=1)
        search_result = SearchResult(query="Test Query", results=[chunk], execution_time_ms=10)
        inputs = [(job_result, search_result)]

        mock_future = Mock(spec=concurrent.futures.Future)
        phase._preload_future = mock_future
        mock_loaded_model = Mock(spec=LoadedModel)
        mock_async_service.resolve_preload.return_value = mock_loaded_model

        mock_job_manager.run_job.return_value = {"surviving_chunks_count": 1, "_transient_chunks": [chunk]}

        result = phase.execute(inputs)

        assert result.phase == PipelinePhase.RETRIEVAL_SCRUBBING
        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 1
        assert result.failed_count == 0

        scrub_res, chunks = result.items[0]
        assert isinstance(scrub_res, RetrievalScrubbingJobResult)
        assert scrub_res.success is True
        assert len(chunks) == 1

        mock_job_manager.run_job.assert_called_once_with(ANY, ANY, prompt="Test Query", chunks=[chunk])
        mock_async_service.trigger_unload.assert_called_once_with(loaded_model=mock_loaded_model, loader=mock_loader,
                                                                  flush=True)

    def test_execute_skips_failed_upstream(self, phase):
        phase._preload_future = None
        job_result = RagRetrievalJobResult(search_id=uuid.uuid4(), success=False, error_message="Upstream fail")

        result = phase.execute([(job_result, None)])

        assert result.status == PhaseStatus.COMPLETED
        assert result.successful_count == 0
        assert result.failed_count == 0