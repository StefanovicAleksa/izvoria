# Canvas: test_onyx_engine_unit_v3
# Path: tests/engine/service/test_onyx_engine_unit.py

import pytest
import uuid
from unittest.mock import Mock, patch

from app.engine.service.onyx_engine import OnyxEngine
from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.engine.exceptions.lifecycle import EngineOfflineError
from app.features.rag_retrieval.domain.entities import SearchQuery
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult
from app.features.storage.domain.entities import Source, SourceType


class TestOnyxEngineUnit:
    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_media_pipeline(self): return Mock()

    @pytest.fixture
    def mock_audio_pipeline(self): return Mock()

    @pytest.fixture
    def mock_retrieval_api(self): return Mock()

    @pytest.fixture
    def mock_storage_api(self): return Mock()

    @pytest.fixture
    def engine(self, mock_async_service, mock_media_pipeline, mock_audio_pipeline, mock_retrieval_api,
               mock_storage_api):
        return OnyxEngine(
            async_model_service=mock_async_service,
            media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline,
            retrieval_api=mock_retrieval_api,
            storage_api=mock_storage_api
        )

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_start_and_stop_lifecycle(self, mock_db_manager, engine, mock_async_service):
        assert engine.get_state() == EngineState.OFFLINE

        # Test Start
        engine.start()
        assert engine.get_state() == EngineState.ONLINE
        mock_db_manager.session.assert_called_once()

        # Test Stop
        engine.stop()
        assert engine.get_state() == EngineState.OFFLINE
        mock_async_service.shutdown.assert_called_once()
        mock_db_manager.dispose.assert_called_once()

    def test_ingest_raises_if_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.ingest("/dummy/path")

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_ingest_happy_path_aggregation(self, _, engine, mock_media_pipeline, mock_audio_pipeline):
        # Boot the engine
        engine.start()

        # Mock Pipeline Results
        mock_audio_pipeline.process_directory.return_value = {
            PipelinePhase.SCAN: PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                 successful_count=2, items=[]),
            PipelinePhase.RAG_INDEXING: PhaseBatchResult(phase=PipelinePhase.RAG_INDEXING, status=PhaseStatus.COMPLETED,
                                                         successful_count=2, failed_count=0,
                                                         items=[Mock(success=True, dense_vectors_saved=10)])
        }

        mock_media_pipeline.process_directory.return_value = {
            PipelinePhase.SCAN: PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                 successful_count=1, items=[]),
            PipelinePhase.CLIP_EXTRACTION: PhaseBatchResult(phase=PipelinePhase.CLIP_EXTRACTION,
                                                            status=PhaseStatus.COMPLETED,
                                                            items=[Mock(success=True, extracted_clips_count=5)])
        }

        report = engine.ingest("/dummy/path")

        assert isinstance(report, IngestionReport)
        assert report.total_files_scanned == 3  # 2 audio + 1 video
        assert report.successful_ingestions == 2
        assert report.vectors_created == 10
        assert report.clips_extracted == 5
        assert engine.get_state() == EngineState.ONLINE

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_search_resolves_physical_paths(self, _, engine, mock_retrieval_api, mock_storage_api):
        engine.start()

        # Pure Mocks avoid strict Pydantic validation errors
        mock_chunk_video = Mock(
            source_id=uuid.uuid4(),
            extracted_file_id=uuid.uuid4(),
            text_payload="video text",
            start_sec=0,
            end_sec=5,
            relevance_score=0.9
        )
        mock_chunk_audio = Mock(
            source_id=uuid.uuid4(),
            extracted_file_id=None,
            text_payload="audio text",
            start_sec=0,
            end_sec=5,
            relevance_score=0.8
        )

        # Mock the entire SearchResult object instead of instantiating the Pydantic model
        mock_search_result = Mock()
        mock_search_result.results = [mock_chunk_video, mock_chunk_audio]
        mock_search_result.execution_time_ms = 100

        mock_retrieval_api.search.return_value = mock_search_result

        mock_storage_api.get_file_path.return_value = "/vault/video_clip.mp4"

        mock_source = Source(
            id=mock_chunk_audio.source_id,
            name="audio",
            source_type=SourceType.AUDIO,
            content_hash="hash",
            file_id=uuid.uuid4()
        )
        mock_source.file = Mock(file_path="/vault/audio_source.wav")
        mock_storage_api.get_source.return_value = mock_source

        result = engine.search(SearchQuery(prompt="test"))

        assert len(result.results) == 2
        assert getattr(result.results[0], 'physical_path') == "/vault/video_clip.mp4"
        assert getattr(result.results[1], 'physical_path') == "/vault/audio_source.wav"